import torch


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
            * (r - ref_r)[..., None]
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
