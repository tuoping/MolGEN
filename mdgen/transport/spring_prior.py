import torch


class SiO2SpringNoise:
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
        atomic_numbers,
        cutoff=2.0,
        k_parallel=30.0,
        k_perp=3.0,
        k_pin=1.0,
        dtype=torch.float32,
        device=None,
    ):
        if device is None:
            device = ref_frac.device

        self.device = torch.device(device)
        self.dtype = dtype

        self.ref_frac = torch.as_tensor(
            ref_frac,
            device=self.device,
            dtype=dtype,
        )

        self.cell = torch.as_tensor(
            cell,
            device=self.device,
            dtype=dtype,
        )

        self.atomic_numbers = torch.as_tensor(
            atomic_numbers,
            device=self.device,
            dtype=torch.long,
        )

        self.N = self.ref_frac.shape[-2]
        self.D = 3 * self.N

        self.cutoff = cutoff
        self.k_parallel = k_parallel
        self.k_perp = k_perp
        self.k_pin = k_pin

        self.inv_cell = torch.linalg.inv(self.cell)

        # --------------------------------------------------------
        # Build H once.
        # --------------------------------------------------------
        H, bond_index = self._build_hessian()

        self.bond_index = bond_index
        self.n_bonds = bond_index.shape[1]

        # Weak pinning removes translational zero modes.
        H.diagonal().add_(k_pin)

        # Ensure perfect symmetry numerically.
        H = 0.5 * (H + H.T)

        self.H = H

        # --------------------------------------------------------
        # Cholesky:
        #
        #       H = L L^T
        #
        # Sampling:
        #
        #       u = sqrt(kBT) L^{-T} z
        #
        # gives
        #
        #       Cov(u) = kBT H^{-1}
        # --------------------------------------------------------
        L, info = torch.linalg.cholesky_ex(H)

        if torch.any(info != 0):
            raise RuntimeError(
                "Spring Hessian is not positive definite. "
                "Increase k_pin."
            )

        self.L = L.contiguous()
        self.LT = L.T.contiguous()

        # Useful if evaluating normalized reference density.
        self.logdetH = (
            2.0
            * torch.log(torch.diagonal(self.L)).sum()
        )

    # ============================================================
    # Hessian construction
    # ============================================================

    @torch.no_grad()
    def _build_hessian(self):
        """
        Pure-torch periodic Si-O neighbor finding + Hessian assembly.

        Assumes the simulation cell is sufficiently large that
        minimum-image fractional wrapping is appropriate.
        """

        N = self.N
        device = self.device
        dtype = self.dtype

        # --------------------------------------------------------
        # Generate unique atom pairs i < j
        # --------------------------------------------------------
        ij = torch.triu_indices(
            N,
            N,
            offset=1,
            device=device,
        )

        i = ij[0]
        j = ij[1]

        Zi = self.atomic_numbers[i]
        Zj = self.atomic_numbers[j]

        # --------------------------------------------------------
        # Only Si-O pairs
        # --------------------------------------------------------
        is_SiO = (
            ((Zi == 14) & (Zj == 8))
            |
            ((Zi == 8) & (Zj == 14))
        )

        i = i[is_SiO]
        j = j[is_SiO]

        # --------------------------------------------------------
        # Minimum-image displacement in fractional coordinates
        # --------------------------------------------------------
        dfrac = (
            self.ref_frac[j]
            - self.ref_frac[i]
        )

        dfrac = dfrac - torch.round(dfrac)

        # fractional -> Cartesian
        dr = dfrac @ self.cell

        r2 = torch.sum(dr * dr, dim=-1)

        within_cutoff = (
            r2 < self.cutoff ** 2
        )

        i = i[within_cutoff]
        j = j[within_cutoff]
        dr = dr[within_cutoff]
        r2 = r2[within_cutoff]

        # --------------------------------------------------------
        # Unit bond vectors
        # --------------------------------------------------------
        r = torch.sqrt(r2)

        n = dr / r[:, None]

        # n n^T
        nn = (
            n[:, :, None]
            * n[:, None, :]
        )

        E = len(i)

        eye3 = torch.eye(
            3,
            dtype=dtype,
            device=device,
        )[None]

        # --------------------------------------------------------
        # Bond stiffness tensor
        #
        # K =
        #     k_perp I
        #   + (k_parallel-k_perp) nn^T
        # --------------------------------------------------------
        K = (
            self.k_perp * eye3
            +
            (self.k_parallel - self.k_perp)
            * nn
        )

        # --------------------------------------------------------
        # Global coordinate indices
        # --------------------------------------------------------
        xyz = torch.arange(
            3,
            device=device,
        )

        ii = 3 * i[:, None] + xyz
        jj = 3 * j[:, None] + xyz

        def block_indices(row, col):
            rows = (
                row[:, :, None]
                .expand(-1, 3, 3)
                .reshape(-1)
            )

            cols = (
                col[:, None, :]
                .expand(-1, 3, 3)
                .reshape(-1)
            )

            return rows, cols

        rii, cii = block_indices(ii, ii)
        rjj, cjj = block_indices(jj, jj)
        rij, cij = block_indices(ii, jj)
        rji, cji = block_indices(jj, ii)

        # Hessian contribution from each bond:
        #
        #       [ K  -K ]
        #       [-K   K ]
        rows = torch.cat([
            rii,
            rjj,
            rij,
            rji,
        ])

        cols = torch.cat([
            cii,
            cjj,
            cij,
            cji,
        ])

        kvals = K.reshape(-1)

        values = torch.cat([
             kvals,
             kvals,
            -kvals,
            -kvals,
        ])

        # Sparse assembly first because number of bond terms
        # is tiny compared with D^2.
        H = torch.sparse_coo_tensor(
            torch.stack([rows, cols]),
            values,
            size=(self.D, self.D),
            dtype=dtype,
            device=device,
        )

        H = H.coalesce().to_dense()

        bond_index = torch.stack([i, j])

        return H, bond_index

    # ============================================================
    # Sampling
    # ============================================================

    @torch.no_grad()
    def sample_cartesian(
        self,
        kBT,
        generator=None,
    ):
        """
        Sample Cartesian correlated displacement.

        kBT can be:
            scalar
            (B,)
            (B,T)
            ...

        Returns:
            (*kBT.shape, N, 3)
        """

        kBT = torch.as_tensor(
            kBT,
            device=self.device,
            dtype=self.dtype,
        )

        if torch.any(kBT <= 0):
            raise ValueError("kBT must be positive.")

        batch_shape = kBT.shape
        B = kBT.numel()

        # --------------------------------------------------------
        # z ~ N(0, I)
        #
        # Arrange RHS as (D, B), because triangular solve with
        # multiple RHS is very efficient on GPU.
        # --------------------------------------------------------
        z = torch.randn(
            self.D,
            B,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )

        z.mul_(
            torch.sqrt(kBT.reshape(1, B))
        )

        # --------------------------------------------------------
        # Solve
        #
        #       L^T u = sqrt(kBT) z
        #
        # => u = sqrt(kBT) L^{-T} z
        #
        # Cov(u) = kBT H^{-1}
        # --------------------------------------------------------
        u = torch.linalg.solve_triangular(
            self.LT,
            z,
            upper=True,
        )

        # (D,B) -> (B,N,3)
        u = (
            u.T
            .reshape(B, self.N, 3)
        )

        return u.reshape(
            *batch_shape,
            self.N,
            3,
        )

    @torch.no_grad()
    def sample(
        self,
        kBT,
        generator=None,
    ):
        """
        Return fractional-coordinate noise.

        Output:
            (*kBT.shape, N, 3)
        """

        u_cart = self.sample_cartesian(
            kBT,
            generator=generator,
        )

        return u_cart @ self.inv_cell

    @torch.no_grad()
    def sample_with_logq(self, kBT, generator=None):
    
        kBT = torch.as_tensor(
            kBT,
            device=self.device,
            dtype=self.dtype,
        )
    
        batch_shape = kBT.shape
        B = kBT.numel()
    
        z = torch.randn(
            self.D,
            B,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )
    
        # Sample-dependent reduced prior energy
        beta_U0 = 0.5 * (z * z).sum(dim=0)
    
        rhs = z * torch.sqrt(
            kBT.reshape(1, B)
        )
    
        u = torch.linalg.solve_triangular(
            self.LT,
            rhs,
            upper=True,
        )
    
        u_cart = (
            u.T
            .reshape(B, self.N, 3)
            .reshape(*batch_shape, self.N, 3)
        )
    
        # Normalization of Gaussian prior
        beta_F0 = (
            0.5 * self.logdetH
            - 0.5 * self.D
            * torch.log(
                2.0 * torch.pi * kBT
            )
        )
    
        beta_U0 = beta_U0.reshape(batch_shape)
    
        logq0 = beta_F0 - beta_U0
    
        du_frac = u_cart @ self.inv_cell
    
        return du_frac, beta_U0, logq0

    @torch.no_grad()
    def sample_positions(
        self,
        kBT,
        generator=None,
    ):
        """
        Return reference + correlated fractional noise.
        """

        return (
            self.ref_frac
            + self.sample(
                kBT,
                generator=generator,
            )
        )

    # ============================================================
    # Reference energy / density
    # ============================================================

    def reduced_energy_cartesian(
        self,
        u_cart,
        kBT,
    ):
        """
        beta U0 = u^T H u / (2 kBT)

        u_cart:
            (*batch, N, 3)

        kBT:
            matching batch dimensions
        """

        kBT = torch.as_tensor(
            kBT,
            device=self.device,
            dtype=self.dtype,
        )

        batch_shape = u_cart.shape[:-2]
        B = u_cart.numel() // self.D

        u = u_cart.reshape(B, self.D)

        # Since H = L L^T:
        #
        # u^T H u = ||u^T L||^2
        #
        # avoids explicit H multiplication.
        Lu = u @ self.L

        energy = 0.5 * torch.sum(
            Lu * Lu,
            dim=-1,
        )

        return (
            energy
            / kBT.reshape(-1)
        ).reshape(batch_shape)

    def log_prob_cartesian(
        self,
        u_cart,
        kBT,
    ):
        """
        Normalized log q(u) with respect to Cartesian measure.
        """

        kBT = torch.as_tensor(
            kBT,
            device=self.device,
            dtype=self.dtype,
        )

        beta_U = self.reduced_energy_cartesian(
            u_cart,
            kBT,
        )

        log_norm = (
            0.5 * self.logdetH
            - 0.5 * self.D
            * torch.log(
                2.0
                * torch.pi
                * kBT
            )
        )

        return log_norm - beta_U

    def diagnostics(self):
        print(f"N atoms       : {self.N}")
        print(f"DOF           : {self.D}")
        print(f"Si-O bonds    : {self.n_bonds}")
        print(
            f"k_parallel    : "
            f"{self.k_parallel:g} eV/A^2"
        )
        print(
            f"k_perp        : "
            f"{self.k_perp:g} eV/A^2"
        )
        print(
            f"k_pin         : "
            f"{self.k_pin:g} eV/A^2"
        )
        print(
            f"log det H     : "
            f"{self.logdetH.item():.6f}"
        )
