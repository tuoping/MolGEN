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
        r_on_scale  = 0.60
        r_off_scale = 0.80

    Therefore the wall is exactly zero once atoms are at >=80%
    of their approximate normal covalent-contact distance.

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
        r_on_scale=0.60,
        r_off_scale=0.80,
        strength=1.0,
        eps=1e-6,
    ):
        super().__init__()

        if not (0.0 < r_on_scale < r_off_scale):
            raise ValueError(
                "Require 0 < r_on_scale < r_off_scale."
            )

        self.r_on_scale = r_on_scale
        self.r_off_scale = r_off_scale
        self.strength = strength
        self.eps = eps

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

        V = (
            self.COULOMB
            * zi_f
            * zj_f
            / r
            * phi
        )

        return self.strength * V

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

        r_on = (
            self.r_on_scale
            * rij_cov
        )

        r_off = (
            self.r_off_scale
            * rij_cov
        )

        # --------------------------------------------------------
        # ZBL repulsion
        # --------------------------------------------------------
        V_zbl = self._zbl(
            r,
            zi,
            zj,
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
            self.cell,
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
