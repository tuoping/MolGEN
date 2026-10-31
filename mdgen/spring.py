import torch


class HarmonicOverlapWall:
    """
    Independent short-range repulsive overlap potential.

    Energy for each atom pair:

        E_ij = 1/2 * k_wall * max(r_inner - r_ij, 0)^2

    Force on atom j:

        F_j = k_wall * max(r_inner - r_ij, 0) * n_ij

    where

        n_ij = (r_j - r_i) / r_ij

    and r_inner depends smoothly on atomic number.

    Default behavior:
        pair containing H       -> r_inner = 0.90 A
        progressively heavier   -> larger r_inner
        Z_min >= inner_z_max    -> r_inner = 1.25 A

    Inputs
    ------
    cell : (3, 3) or (B, 3, 3)
        Cell matrix in ASE convention:

            r_cart = r_frac @ cell

    atomic_numbers : (N,) or (B, N)
        Atomic numbers.

    frac passed to build_energy/build_force:
        (N, 3) or (B, N, 3)

    Units
    -----
    distance : Angstrom
    energy   : eV
    force    : eV / Angstrom
    """

    def __init__(
        self,
        inner_cutoff_h=0.90,
        inner_cutoff_other=1.25,
        inner_z_max=8.0,
        k_wall=100.0,
        dtype=torch.float32,
        device=None,
    ):
        if device is None:
            device = "cpu"

        self.device = torch.device(device)
        self.dtype = dtype

        self.inner_cutoff_h = inner_cutoff_h
        self.inner_cutoff_other = inner_cutoff_other
        self.inner_z_max = inner_z_max
        self.k_wall = k_wall

    # ============================================================
    # Batch handling
    # ============================================================

    def _prepare_inputs(self, frac, cell, atomic_numbers):
        frac = torch.as_tensor(
            frac,
            dtype=self.dtype,
            device=self.device,
        )

        unbatched = frac.ndim == 2

        if unbatched:
            frac = frac.unsqueeze(0)

        if frac.ndim != 3:
            raise ValueError(
                f"frac must have shape (N,3) or (B,N,3), "
                f"got {frac.shape}"
            )

        B, N, _ = frac.shape

        if cell.ndim == 2:
            cell = cell.unsqueeze(0)

        if cell.shape[0] == 1 and B > 1:
            cell = cell.expand(B, -1, -1)

        if cell.shape[0] != B:
            raise ValueError(
                f"cell batch size {cell.shape[0]} "
                f"does not match frac batch size {B}"
            )

        # atomic numbers

        if atomic_numbers.ndim == 1:
            atomic_numbers = atomic_numbers.unsqueeze(0)

        if atomic_numbers.shape[0] == 1 and B > 1:
            atomic_numbers = atomic_numbers.expand(B, -1)

        if atomic_numbers.shape[0] != B:
            raise ValueError(
                f"atomic_numbers batch size {atomic_numbers.shape[0]} "
                f"does not match frac batch size {B}"
            )

        if atomic_numbers.shape[1] != N:
            raise ValueError(
                f"atomic_numbers contains {atomic_numbers.shape[1]} atoms, "
                f"but frac contains {N}"
            )

        return frac, cell, atomic_numbers, unbatched

    # ============================================================
    # Pair-dependent inner cutoff
    # ============================================================

    def _inner_cutoff(self, zi, zj, dtype):
        """
        Smooth interpolation:

            Zmin = 1
                -> inner_cutoff_h

            Zmin >= inner_z_max
                -> inner_cutoff_other

        Uses cubic smoothstep:

            s(x) = x^2 (3 - 2x)
        """

        zmin = torch.minimum(zi, zj).to(dtype)

        x = (
            (zmin - 1.0)
            / (self.inner_z_max - 1.0)
        )

        x = torch.clamp(
            x,
            min=0.0,
            max=1.0,
        )

        s = x * x * (3.0 - 2.0 * x)

        return (
            self.inner_cutoff_h
            + (
                self.inner_cutoff_other
                - self.inner_cutoff_h
            )
            * s
        )

    # ============================================================
    # Pair geometry
    # ============================================================

    def _pair_geometry(
        self,
        frac,
        cell,
        atomic_numbers,
    ):
        B, N, _ = frac.shape

        # all unique pairs
        ij = torch.triu_indices(
            N,
            N,
            offset=1,
            device=frac.device,
        )

        i = ij[0]
        j = ij[1]

        # fractional displacement
        dfrac = (
            frac[:, j, :]
            - frac[:, i, :]
        )

        # minimum-image convention
        dfrac = dfrac - torch.round(dfrac)

        # Cartesian displacement
        dr = torch.einsum(
            "bpi,bij->bpj",
            dfrac,
            cell,
        )

        r = torch.linalg.norm(
            dr,
            dim=-1,
        )

        # pair atomic numbers
        zi = atomic_numbers[:, i]
        zj = atomic_numbers[:, j]

        # pair-dependent wall distance
        r_inner = self._inner_cutoff(
            zi,
            zj,
            frac.dtype,
        )

        return i, j, dr, r, r_inner

    # ============================================================
    # Energy
    # ============================================================

    @torch.no_grad()
    def build_energy(self, frac, cell, atomic_numbers):

        (
            frac,
            cell,
            atomic_numbers,
            unbatched,
        ) = self._prepare_inputs(frac, cell, atomic_numbers)

        (
            i,
            j,
            dr,
            r,
            r_inner,
        ) = self._pair_geometry(
            frac,
            cell,
            atomic_numbers,
        )

        penetration = torch.clamp(
            r_inner - r,
            min=0.0,
        )

        E_pair = (
            0.5
            * self.k_wall
            * penetration**2
        )

        E = E_pair.sum(dim=-1)

        if unbatched:
            E = E.squeeze(0)

        return E

    # ============================================================
    # Force
    # ============================================================

    @torch.no_grad()
    def build_force(self, frac, cell, atomic_numbers):

        (
            frac,
            cell,
            atomic_numbers,
            unbatched,
        ) = self._prepare_inputs(frac, cell, atomic_numbers)

        B, N, _ = frac.shape

        (
            i,
            j,
            dr,
            r,
            r_inner,
        ) = self._pair_geometry(
            frac,
            cell,
            atomic_numbers,
        )

        # direction i -> j
        n = (
            dr
            / r.clamp_min(1e-12)[..., None]
        )

        penetration = torch.clamp(
            r_inner - r,
            min=0.0,
        )

        # force acting on atom j
        F_pair = (
            self.k_wall
            * penetration[..., None]
            * n
        )

        # pair forces -> atomic forces
        F = torch.zeros(
            B,
            N,
            3,
            dtype=frac.dtype,
            device=frac.device,
        )

        # atom i gets -F_pair
        F.scatter_add_(
            1,
            i[None, :, None].expand(B, -1, 3),
            -F_pair,
        )

        # atom j gets +F_pair
        F.scatter_add_(
            1,
            j[None, :, None].expand(B, -1, 3),
            F_pair,
        )

        if unbatched:
            F = F.squeeze(0)

        return F

import torch
import torch.nn as nn
from ase.data import covalent_radii


def _zbl_pair_parameters(
    zi,
    zj,
    *,
    zbl_scale=1.0,
    r_soft_scale=0.20,
    r_on_scale=0.60,
    r_off_scale=0.80,
    exclusion_k=0.0,
    exclusion_r_on=None,
    exclusion_r_off=None,
):
    """Build one entry in the pair-parameter dictionary."""

    radius_sum = float(covalent_radii[zi] + covalent_radii[zj])
    return {
        "zbl_scale": float(zbl_scale),
        "r_soft_scale": float(r_soft_scale),
        "r_on_scale": float(r_on_scale),
        "r_off_scale": float(r_off_scale),
        "r_soft": round(float(r_soft_scale * radius_sum), 6),
        "r_on": round(float(r_on_scale * radius_sum), 6),
        "r_off": round(float(r_off_scale * radius_sum), 6),
        "exclusion_k": float(exclusion_k),
        "exclusion_r_on": exclusion_r_on,
        "exclusion_r_off": exclusion_r_off,
    }


# Unordered element-pair keys are canonicalized as (min(Zi, Zj), max(Zi, Zj)).
# Elements found below experiments/MOF/lattpath_noTime/Stage2.1 are
# H (1), C (6), N (7), Zn (30), and I (53).
ZBL_PAIR_PARAMETERS = {
    (1, 1): _zbl_pair_parameters(1, 1),
    (1, 6): _zbl_pair_parameters(1, 6),
    (1, 7): _zbl_pair_parameters(1, 7),
    (1, 30): _zbl_pair_parameters(1, 30),
    (1, 53): _zbl_pair_parameters(1, 53),
    (6, 6): _zbl_pair_parameters(6, 6),
    (6, 7): _zbl_pair_parameters(6, 7),
    (6, 30): _zbl_pair_parameters(6, 30),
    (6, 53): _zbl_pair_parameters(6, 53),
    (7, 7): _zbl_pair_parameters(7, 7),
    (7, 30): _zbl_pair_parameters(7, 30),
    (7, 53): _zbl_pair_parameters(7, 53),
    (30, 30): _zbl_pair_parameters(
        30,
        30,
        exclusion_k=10.0,
        exclusion_r_on=4.5,
        exclusion_r_off=5.0,
    ),
    (30, 53): _zbl_pair_parameters(30, 53),
    (53, 53): _zbl_pair_parameters(53, 53),
}

_DEFAULT_ZBL_PAIR_PARAMETERS = ZBL_PAIR_PARAMETERS[(1, 1)]


class ZBLRepulsiveWall(nn.Module):
    """
    Short-range ZBL repulsive wall for stabilizing ML potentials.

    Total potential can be constructed as

        E_total = E_ML + E_wall

    or, if forces are computed separately,

        F_total = F_ML + F_wall

    The ZBL interaction is smoothly switched off between

        r_on  = r_on_scale  * (Rcov_i + Rcov_j)
        r_off = r_off_scale * (Rcov_i + Rcov_j)

    Default:
        r_soft_scale = 0.20
        r_on_scale  = 0.60
        r_off_scale = 0.80

    Below ``r_soft_scale * (Rcov_i + Rcov_j)`` the singular ZBL potential is
    continued linearly.  This preserves the repulsive force at the soft-core
    boundary while preventing a nearly coincident pair from producing an
    unbounded force and destabilizing an integration step.

    Zn-Zn pairs additionally receive a MOF-specific exclusion wall:

        E_ZnZn = 1/2 k_ZnZn max(r_off_ZnZn - r, 0)^2 s(r)

    where the pair dictionary defines ``zbl_scale``, the exclusion spring
    constant ``exclusion_k``, and the switching distances.  This term prevents
    collapsed Zn nodes while leaving the 6.17--6.73 Angstrom Zn-Zn separations
    observed in the selected trajectories untouched.

    The universal ZBL component is exactly zero once atoms are at >=80% of
    their approximate normal covalent-contact distance.  The separate Zn-Zn
    component is exactly zero at and beyond its configured exclusion cutoff.

    Coordinates
    -----------
    frac : (N, 3) or (B, N, 3)
        Fractional coordinates.

    cell : (3, 3) or (B, 3, 3)
        ASE convention:

            r_cart = r_frac @ cell

    atomic_numbers : (N,) or (B, N)

    Units
    -----
    distance : Angstrom
    energy   : eV
    force    : eV / Angstrom
    """

    # Coulomb constant e^2 / (4 pi epsilon_0), eV Angstrom
    COULOMB = 14.3996454784255

    # Bohr radius, Angstrom
    A0 = 0.529177210903

    # Standard ZBL screening coefficients
    ZBL_C = (0.1818, 0.5099, 0.2802, 0.02817)
    ZBL_D = (3.2, 0.9423, 0.4029, 0.2016)

    def __init__(
        self,
        atomic_numbers=None,
        pair_parameters=None,
        eps=1e-6,
    ):
        super().__init__()

        self.eps = eps

        if pair_parameters is None:
            pair_parameters = ZBL_PAIR_PARAMETERS
        self.pair_parameters = {}
        required_parameters = set(_DEFAULT_ZBL_PAIR_PARAMETERS)
        for pair, parameters in pair_parameters.items():
            if len(pair) != 2:
                raise ValueError(f"Invalid element-pair key: {pair}")
            zi, zj = (int(pair[0]), int(pair[1]))
            if zi <= 0 or zi > zj:
                raise ValueError(
                    "Pair keys must be canonical positive atomic-number "
                    f"tuples, got {pair}."
                )
            parameters = dict(parameters)
            missing = required_parameters - set(parameters)
            if missing:
                raise ValueError(
                    f"Pair {(zi, zj)} is missing parameters: {sorted(missing)}"
                )
            if not (
                0.0
                < parameters["r_soft_scale"]
                < parameters["r_on_scale"]
                < parameters["r_off_scale"]
            ):
                raise ValueError(
                    f"Invalid ZBL scales for pair {(zi, zj)}."
                )
            if not (
                0.0
                < parameters["r_soft"]
                < parameters["r_on"]
                < parameters["r_off"]
            ):
                raise ValueError(
                    f"Invalid ZBL radii for pair {(zi, zj)}."
                )
            if parameters["zbl_scale"] < 0.0:
                raise ValueError(
                    f"ZBL scale must be non-negative for pair {(zi, zj)}."
                )
            exclusion_k = parameters["exclusion_k"]
            if exclusion_k < 0.0:
                raise ValueError(
                    "Exclusion spring constant must be non-negative for pair "
                    f"{(zi, zj)}."
                )
            if exclusion_k > 0.0 and not (
                parameters["exclusion_r_on"] is not None
                and parameters["exclusion_r_off"] is not None
                and 0.0
                < parameters["exclusion_r_on"]
                < parameters["exclusion_r_off"]
            ):
                raise ValueError(
                    f"Invalid exclusion radii for pair {(zi, zj)}."
                )
            self.pair_parameters[(zi, zj)] = parameters

        if atomic_numbers is not None:
            atomic_numbers = torch.as_tensor(
                atomic_numbers,
                dtype=torch.long,
            )

        self.register_buffer(
            "atomic_numbers",
            atomic_numbers,
        )

        # ASE covalent radii for the periodic table.
        self.register_buffer(
            "rcov",
            torch.as_tensor(
                covalent_radii,
                dtype=torch.float64,
            ),
        )

    # ============================================================
    # Input preparation
    # ============================================================

    def _prepare(
        self,
        frac,
        cell,
        atomic_numbers=None,
    ):
        unbatched = frac.ndim == 2

        if unbatched:
            frac_b = frac.unsqueeze(0)
        else:
            frac_b = frac

        if frac_b.ndim != 3:
            raise ValueError(
                f"frac must be (N,3) or (B,N,3), got {frac.shape}"
            )

        B, N, _ = frac_b.shape

        # --------------------------------------------------------
        # Cell
        # --------------------------------------------------------
        cell = torch.as_tensor(
            cell,
            dtype=frac_b.dtype,
            device=frac_b.device,
        )

        if cell.ndim == 2:
            cell = cell.unsqueeze(0)

        if cell.shape[0] == 1 and B > 1:
            cell = cell.expand(B, -1, -1)

        if cell.shape[0] != B:
            raise ValueError(
                f"cell batch size {cell.shape[0]} != {B}"
            )

        # --------------------------------------------------------
        # Atomic numbers
        # --------------------------------------------------------
        if atomic_numbers is None:
            atomic_numbers = self.atomic_numbers

        if atomic_numbers is None:
            raise ValueError(
                "atomic_numbers must either be supplied when "
                "constructing the wall or passed to forward()."
            )

        atomic_numbers = torch.as_tensor(
            atomic_numbers,
            dtype=torch.long,
            device=frac_b.device,
        )

        if atomic_numbers.ndim == 1:
            atomic_numbers = atomic_numbers.unsqueeze(0)

        if (
            atomic_numbers.shape[0] == 1
            and B > 1
        ):
            atomic_numbers = atomic_numbers.expand(
                B, -1
            )

        if atomic_numbers.shape != (B, N):
            raise ValueError(
                f"atomic_numbers has shape "
                f"{atomic_numbers.shape}, expected {(B, N)}"
            )

        return (
            frac_b,
            cell,
            atomic_numbers,
            unbatched,
        )

    # ============================================================
    # Smooth cutoff
    # ============================================================

    def _switch(
        self,
        r,
        r_on,
        r_off,
    ):
        """
        C2 smooth switching function.

        s = 1                       r <= r_on
        s = 0                       r >= r_off

        In between:

            t = (r-r_on)/(r_off-r_on)

            s(t) = 1 - 10t^3 + 15t^4 - 6t^5

        Both first and second derivatives vanish at the endpoints.
        """

        t = (
            (r - r_on)
            / (r_off - r_on)
        )

        t = torch.clamp(
            t,
            min=0.0,
            max=1.0,
        )

        return (
            1.0
            - 10.0 * t**3
            + 15.0 * t**4
            - 6.0 * t**5
        )

    # ============================================================
    # ZBL pair potential
    # ============================================================

    def _zbl(
        self,
        r,
        zi,
        zj,
        r_soft,
    ):
        """
        Standard universal ZBL screened nuclear repulsion.

        V(r) =
            ke * Zi * Zj / r * phi(r/a)

        with

            a = 0.8854 a0 /
                (Zi^0.23 + Zj^0.23)
        """

        dtype = r.dtype

        zi_f = zi.to(dtype)
        zj_f = zj.to(dtype)

        # ZBL screening length
        a = (
            0.8854
            * self.A0
            / (
                zi_f.pow(0.23)
                + zj_f.pow(0.23)
            )
        )

        x = r / a

        phi = (
            self.ZBL_C[0]
            * torch.exp(-self.ZBL_D[0] * x)
            +
            self.ZBL_C[1]
            * torch.exp(-self.ZBL_D[1] * x)
            +
            self.ZBL_C[2]
            * torch.exp(-self.ZBL_D[2] * x)
            +
            self.ZBL_C[3]
            * torch.exp(-self.ZBL_D[3] * x)
        )

        prefactor = (
            self.COULOMB
            * zi_f
            * zj_f
        )
        V = prefactor / r * phi

        # A linear continuation is C1 at r_soft and caps the pair-force
        # magnitude.  The successful Stage2.1 trajectories used to calibrate
        # this boundary remain above 0.20 * (Rcov_i + Rcov_j) during the
        # guided, late-time portion of their paths, so their ZBL curve is
        # unchanged.
        x_soft = r_soft / a
        exp_soft = tuple(
            torch.exp(-coefficient * x_soft)
            for coefficient in self.ZBL_D
        )
        phi_soft = sum(
            coefficient * exponential
            for coefficient, exponential in zip(self.ZBL_C, exp_soft)
        )
        minus_dphi_dx_soft = sum(
            coefficient * decay * exponential
            for coefficient, decay, exponential in zip(
                self.ZBL_C,
                self.ZBL_D,
                exp_soft,
            )
        )
        V_soft = prefactor / r_soft * phi_soft
        force_soft = prefactor * (
            phi_soft / r_soft.square()
            + minus_dphi_dx_soft / (a * r_soft)
        )
        V = torch.where(
            r < r_soft,
            V_soft + force_soft * (r_soft - r),
            V,
        )

        return V

    # ============================================================
    # Energy
    # ============================================================

    def forward(
        self,
        frac,
        cell,
        atomic_numbers=None,
    ):
        """
        Return wall energy.

        Output:
            scalar for (N,3)
            (B,) for (B,N,3)
        """

        (
            frac,
            cell,
            z,
            unbatched,
        ) = self._prepare(
            frac,
            cell,
            atomic_numbers,
        )

        B, N, _ = frac.shape

        # --------------------------------------------------------
        # All unique atom pairs
        # --------------------------------------------------------
        ij = torch.triu_indices(
            N,
            N,
            offset=1,
            device=frac.device,
        )

        i = ij[0]
        j = ij[1]

        # --------------------------------------------------------
        # Pair displacement under PBC
        # --------------------------------------------------------
        dfrac = (
            frac[:, j, :]
            - frac[:, i, :]
        )

        dfrac = (
            dfrac
            - torch.round(dfrac)
        )

        dr = torch.einsum(
            "bpi,bij->bpj",
            dfrac,
            cell,
        )

        r2 = torch.sum(
            dr * dr,
            dim=-1,
        )

        # Soft numerical protection against exact r=0.
        r = torch.sqrt(
            r2 + self.eps**2
        )

        # --------------------------------------------------------
        # Species
        # --------------------------------------------------------
        zi = z[:, i]
        zj = z[:, j]

        # --------------------------------------------------------
        # Pair-specific covalent contact distance
        # --------------------------------------------------------
        rcov = self.rcov.to(
            dtype=frac.dtype,
            device=frac.device,
        )

        rij_cov = (
            rcov[zi]
            + rcov[zj]
        )

        # Begin with universal fallback values for elements absent from the
        # configured dictionary, then apply the matching unordered-pair entry.
        zbl_scale = torch.full_like(
            r, _DEFAULT_ZBL_PAIR_PARAMETERS["zbl_scale"]
        )
        r_soft = _DEFAULT_ZBL_PAIR_PARAMETERS["r_soft_scale"] * rij_cov
        r_on = _DEFAULT_ZBL_PAIR_PARAMETERS["r_on_scale"] * rij_cov
        r_off = _DEFAULT_ZBL_PAIR_PARAMETERS["r_off_scale"] * rij_cov

        for (pair_zi, pair_zj), parameters in self.pair_parameters.items():
            pair_mask = (
                ((zi == pair_zi) & (zj == pair_zj))
                | ((zi == pair_zj) & (zj == pair_zi))
            )
            zbl_scale = torch.where(
                pair_mask,
                torch.as_tensor(
                    parameters["zbl_scale"], dtype=r.dtype, device=r.device
                ),
                zbl_scale,
            )
            r_soft = torch.where(
                pair_mask,
                torch.as_tensor(
                    parameters["r_soft"], dtype=r.dtype, device=r.device
                ),
                r_soft,
            )
            r_on = torch.where(
                pair_mask,
                torch.as_tensor(
                    parameters["r_on"], dtype=r.dtype, device=r.device
                ),
                r_on,
            )
            r_off = torch.where(
                pair_mask,
                torch.as_tensor(
                    parameters["r_off"], dtype=r.dtype, device=r.device
                ),
                r_off,
            )

        # --------------------------------------------------------
        # ZBL repulsion
        # --------------------------------------------------------
        V_zbl = self._zbl(
            r,
            zi,
            zj,
            r_soft,
        )

        # --------------------------------------------------------
        # Smoothly remove ZBL before normal bonding distances
        # --------------------------------------------------------
        switch = self._switch(
            r,
            r_on,
            r_off,
        )

        E_pair = (
            V_zbl
            * switch
            * zbl_scale
        )

        # Pair-specific longer-range exclusion terms are intentionally
        # separate from the universal short-range ZBL curve.
        for (pair_zi, pair_zj), parameters in self.pair_parameters.items():
            exclusion_k = parameters["exclusion_k"]
            if exclusion_k == 0.0:
                continue
            pair_mask = (
                ((zi == pair_zi) & (zj == pair_zj))
                | ((zi == pair_zj) & (zj == pair_zi))
            )
            exclusion_r_on = parameters["exclusion_r_on"]
            exclusion_r_off = parameters["exclusion_r_off"]
            exclusion_switch = self._switch(
                r,
                torch.full_like(r, exclusion_r_on),
                torch.full_like(r, exclusion_r_off),
            )
            penetration = torch.clamp(
                exclusion_r_off - r,
                min=0.0,
            )
            E_pair = E_pair + (
                0.5
                * exclusion_k
                * penetration.square()
                * exclusion_switch
                * pair_mask.to(r.dtype)
            )

        E = E_pair.sum(dim=-1)

        if unbatched:
            E = E.squeeze(0)

        return E

    # ============================================================
    # Compatibility name
    # ============================================================

    def build_energy(
        self,
        frac,
        cell,
        atomic_numbers=None,
    ):
        return self.forward(
            frac,
            cell,
            atomic_numbers,
        )

    # ============================================================
    # Cartesian force
    # ============================================================

    def build_force(
        self,
        frac,
        cell,
        atomic_numbers=None,
        create_graph=False,
    ):
        """
        Compute Cartesian force from the wall.

        frac is differentiated in fractional coordinates first:

            grad_frac = dE / d frac

        Since

            r_cart = frac @ cell

        we have

            grad_frac = grad_cart @ cell.T

        therefore

            grad_cart = grad_frac @ cell^{-T}

        and

            F_cart = -grad_cart
        """

        unbatched = frac.ndim == 2

        x = frac.detach().clone()
        x.requires_grad_(True)

        E = self.forward(
            x,
            cell,
            atomic_numbers,
        )

        grad_frac = torch.autograd.grad(
            E.sum(),
            x,
            create_graph=create_graph,
        )[0]

        if unbatched:
            grad_frac_b = grad_frac.unsqueeze(0)
        else:
            grad_frac_b = grad_frac

        cell_b = torch.as_tensor(
            cell,
            dtype=grad_frac.dtype,
            device=grad_frac.device,
        )

        if cell_b.ndim == 2:
            cell_b = cell_b.unsqueeze(0)

        if (
            cell_b.shape[0] == 1
            and grad_frac_b.shape[0] > 1
        ):
            cell_b = cell_b.expand(
                grad_frac_b.shape[0],
                -1,
                -1,
            )

        # Solve
        #
        #     cell @ grad_cart_column
        #         = grad_frac_column
        #
        grad_cart = torch.linalg.solve(
            cell_b[:, None, :, :],
            grad_frac_b[..., None],
        ).squeeze(-1)

        force = -grad_cart

        if unbatched:
            force = force.squeeze(0)

        return force

class NNSpring:
    """
    Fast correlated Gaussian noise sampler for SiO2.

    Prior:
        p(u | kBT) ∝ exp[-u^T H u / (2 kBT)]

    so
        Cov(u) = kBT * H^{-1}

    H is a Si-O spring-network Hessian:

        K_ij = k_perp I
             + (k_parallel - k_perp) n_ij n_ij^T

    with an additional weak onsite pinning term

        H <- H + k_pin I

    to remove the three translational zero modes and make the
    Gaussian reference normalizable.

    Inputs
    ------
    ref_frac : (N, 3)
        Reference fractional coordinates.

    cell : (3, 3)
        Cell matrix in ASE convention:
            r_cart = r_frac @ cell

    atomic_numbers : (N,)
        Atomic numbers. Si=14, O=8.

    kBT passed to sample() can have arbitrary batch dimensions:
        kBT.shape == (B,)
        kBT.shape == (B, T)
        etc.

    Output
    ------
    du_frac.shape = (*kBT.shape, N, 3)

    Units
    -----
    coordinates : Angstrom
    H           : eV / Angstrom^2
    kBT         : eV
    """

    def __init__(
        self,
        ref_frac,
        cell,
        atomic_numbers = None,
        cutoff=3.5,
        k_parallel=1.0,
        k_perp=1.0,
        k_pin=1.0,
        dtype=torch.float32,
        device=None,
        reference_cell=None,
    ):
        if device is None:
            device = cell.device

        self.device = torch.device(device)
        self.dtype = dtype

        self.ref_frac = ref_frac
        self.cell = torch.as_tensor(
            cell,
            device=self.device,
            dtype=dtype,
        )
        if reference_cell is None:
            reference_cell = cell
        self.reference_cell = torch.as_tensor(
            reference_cell,
            device=self.device,
            dtype=dtype,
        )
        if atomic_numbers is not None:
            self.atomic_numbers = torch.as_tensor(
                atomic_numbers,
                device=self.device,
                dtype=torch.long,
            )
        else:
            self.atomic_numbers = None

        self.cutoff = cutoff
        self.k_parallel = k_parallel
        self.k_perp = k_perp
        self.k_pin = k_pin

    @torch.no_grad()
    def build_energy(self, frac):

        B, N, _ = frac.shape
        device = frac.device

        # --------------------------------------------------------
        # Unique atom pairs
        # --------------------------------------------------------
        ij = torch.triu_indices(
            N, N,
            offset=1,
            device=device,
        )

        i = ij[0]   # [P]
        j = ij[1]   # [P]

        # --------------------------------------------------------
        # Current pair distances
        # --------------------------------------------------------
        dfrac = frac[:, j, :] - frac[:, i, :]           # [B, P, 3]
        dfrac = dfrac - torch.round(dfrac)

        dr = torch.einsum(
            "bpi,bij->bpj",
            dfrac,
            self.cell,
        )                                                # [B, P, 3]

        r = torch.linalg.norm(dr, dim=-1)                # [B, P]

        # --------------------------------------------------------
        # Reference pair distances
        # --------------------------------------------------------
        ref_dfrac = (
            self.ref_frac[:, j, :]
            - self.ref_frac[:, i, :]
        )                                                # [B, P, 3]

        ref_dfrac = ref_dfrac - torch.round(ref_dfrac)

        ref_dr = torch.einsum(
            "bpi,bij->bpj",
            ref_dfrac,
            self.reference_cell,
        )

        ref_r = torch.linalg.norm(ref_dr, dim=-1)        # [B, P]

        # bonds defined by reference structure
        bond_mask = ref_r < self.cutoff                  # [B, P]

        # --------------------------------------------------------
        # Pair spring energy
        # --------------------------------------------------------
        zprod = (
            self.atomic_numbers[:, i]
            * self.atomic_numbers[:, j]
        )                                                # [B, P]

        E_pair = (
            0.5
            * self.k_parallel
            * zprod
            * (r - ref_r)**2
        )                                                # [B, P]

        E_pair = E_pair.masked_fill(
            ~bond_mask,
            0.0,
        )

        # total energy for each structure
        E = E_pair.sum(dim=-1)                           # [B]

        return E

    @torch.no_grad()
    def build_force(self, frac):
    
        B, N, _ = frac.shape
        device = frac.device
    
        # --------------------------------------------------------
        # Unique atom pairs
        # --------------------------------------------------------
        ij = torch.triu_indices(
            N, N,
            offset=1,
            device=device,
        )
    
        i = ij[0]   # [P]
        j = ij[1]   # [P]
    
        # --------------------------------------------------------
        # Current pair displacement
        # --------------------------------------------------------
        dfrac = frac[:, j, :] - frac[:, i, :]       # [B, P, 3]
        dfrac = dfrac - torch.round(dfrac)
    
        # if self.cell is [B, 3, 3]
        dr = torch.einsum(
            "bpi,bij->bpj",
            dfrac,
            self.cell,
        )                                             # [B, P, 3]
    
        r2 = torch.sum(dr**2, dim=-1)                 # [B, P]
        r = torch.sqrt(r2)
    
        # --------------------------------------------------------
        # Reference pair displacement
        # --------------------------------------------------------
        ref_dfrac = (
            self.ref_frac[:, j, :]
            - self.ref_frac[:, i, :]
        )                                             # [B, P, 3]
    
        ref_dfrac = ref_dfrac - torch.round(ref_dfrac)
    
        ref_dr = torch.einsum(
            "bpi,bij->bpj",
            ref_dfrac,
            self.cell,
        )
    
        ref_r = torch.linalg.norm(ref_dr, dim=-1)     # [B, P]
    
        # bonds defined by reference structure
        bond_mask = ref_r < self.cutoff               # [B, P]
    
        # --------------------------------------------------------
        # Unit vector
        # --------------------------------------------------------
        n = dr / r.clamp_min(1e-12)[..., None]        # [B, P, 3]
    
        # --------------------------------------------------------
        # Pair force
        #
        # dr = r_j - r_i
        # This is force acting on atom j
        # --------------------------------------------------------
        F_pair = (
            -self.k_parallel
            * ((r - ref_r)*(self.atomic_numbers[:,i]*self.atomic_numbers[:,j]))[..., None]
            * n
        )                                             # [B, P, 3]
    
        F_pair = F_pair.masked_fill(
            ~bond_mask[..., None],
            0.0,
        )
    
        # --------------------------------------------------------
        # Convert pair forces -> atomic forces
        # --------------------------------------------------------
        F = torch.zeros(
            B, N, 3,
            device=frac.device,
            dtype=frac.dtype,
        )
    
        # F_pair is force on j
        # opposite force acts on i
        F.scatter_add_(
            1,
            i[None, :, None].expand(B, -1, 3),
            -F_pair,
        )
    
        F.scatter_add_(
            1,
            j[None, :, None].expand(B, -1, 3),
            F_pair,
        )
    
        return F
