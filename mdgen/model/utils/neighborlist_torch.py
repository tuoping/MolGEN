# %%
import torch
import numpy as np
from mace.data import AtomicData
from typing import List, Sequence, Iterable, Dict, Any


####################################################################################################
# ======================================================================
# GPU-friendly neighbour list (minimum image)
# ======================================================================

def torch_neighbour_list(
    positions: torch.Tensor,      # [N, 3], float (Å)
    cell: torch.Tensor,           # [3, 3], float
    pbc: torch.Tensor,            # [3], bool
    cutoff: float,                # scalar cutoff (Å)
    quantities: str = "ijS",      # subset of 'i', 'j', 'D', 'd', 'S'
    device=None,
    dtype: torch.dtype = torch.float32,
):
    """
    Build a minimum-image neighbour list using pure PyTorch tensor operations.

    - Complexity: O(N^2), but for N ~ 1000 this is still very fast on GPU.
    - PBC is handled via standard minimum image convention in fractional coords.
    - No Python-level triple for loop.

    Args
    ----
    positions:
        Atomic positions in Cartesian coordinates, shape [N, 3].
    cell:
        Simulation cell matrix, shape [3, 3].
    pbc:
        Periodic boundary conditions in each direction, shape [3], bool.
    cutoff:
        Cutoff radius (Å).
    quantities:
        String specifying which quantities to return (some subset of "ijSdD").

        Supported characters:
            'i' : source indices (shape [E], int64)
            'j' : target indices (shape [E], int64)
            'S' : integer cell shifts in each direction (shape [E, 3], int64)
            'D' : displacement vectors dr_ij (shape [E, 3], float)
            'd' : distances |dr_ij| (shape [E], float)

        The return value is in the same order as characters in `quantities`.
    device:
        Torch device. If None, uses positions.device.
    dtype:
        Floating dtype for positions / cell.

    Returns
    -------
    If len(quantities) == 1:
        Single tensor.
    Else:
        Tuple of tensors in the same order as `quantities`.
    """
    if device is None:
        device = positions.device

    # Ensure everything is on the same device / dtype
    positions = positions.to(device=device, dtype=dtype)    # [N, 3]
    cell = cell.to(device=device, dtype=dtype)              # [3, 3]
    pbc = pbc.to(device=device, dtype=torch.bool)           # [3]

    N = positions.size(0)

    # Inverse cell for fractional coordinates
    cell_inv = torch.inverse(cell)                          # [3, 3]

    # ------------------------------------------------------------------
    # Pairwise displacements: dr_ij[i, j, :] = pos[j, :] - pos[i, :]
    # ------------------------------------------------------------------
    pos_i = positions.unsqueeze(1)                          # [N, 1, 3]
    pos_j = positions.unsqueeze(0)                          # [1, N, 3]
    dr_ij = pos_j - pos_i                                   # [N, N, 3]

    # Convert to fractional coordinates
    # dr_frac[i, j, :] = dr_ij[i, j, :] @ cell_inv
    dr_frac = torch.einsum("nij,jk->nik", dr_ij, cell_inv)  # [N, N, 3]

    # ------------------------------------------------------------------
    # Apply minimum image convention in fractional coordinates
    # ------------------------------------------------------------------
    # For pbc=True directions: s -> s - round(s)
    # For pbc=False directions: leave unchanged
    pbc_float = pbc.to(dtype=dtype).view(1, 1, 3)           # [1, 1, 3]
    shift_frac = torch.round(dr_frac) * pbc_float           # [N, N, 3]

    dr_frac_min = dr_frac - shift_frac                      # [N, N, 3]

    # Back to Cartesian coordinates
    dr_min = torch.einsum("nij,jk->nik", dr_frac_min, cell) # [N, N, 3]
    dist2 = (dr_min ** 2).sum(-1)                           # [N, N]

    # ------------------------------------------------------------------
    # Cutoff mask + remove self edges
    # ------------------------------------------------------------------
    cutoff_sq = float(cutoff) * float(cutoff)
    mask = dist2 < cutoff_sq

    eye = torch.eye(N, dtype=torch.bool, device=device)
    mask = mask & (~eye)

    # Indices of all edges
    i_idx, j_idx = mask.nonzero(as_tuple=True)              # [E], [E]

    # ------------------------------------------------------------------
    # Integer shifts S (unit_shifts)
    # ------------------------------------------------------------------
    # shift_frac is the integer part we subtracted; only non-zero where pbc=True
    # We define:
    #   unit_shifts = -shift_frac
    # so that: dr_min = dr_ij + (unit_shifts @ cell)
    unit_shifts_all = -shift_frac                           # [N, N, 3]
    unit_shifts_edge = unit_shifts_all[i_idx, j_idx]        # [E, 3]

    # For verification: (not used further)
    # shifts_edge = torch.einsum("ek,kj->ej", unit_shifts_edge, cell)

    dr_edge = dr_min[i_idx, j_idx]                          # [E, 3]
    dist_edge = torch.sqrt((dr_edge ** 2).sum(-1))          # [E]

    i_t = i_idx.to(torch.int64)
    j_t = j_idx.to(torch.int64)
    S_t = unit_shifts_edge.to(torch.int64)
    D_t = dr_edge
    d_t = dist_edge

    # ------------------------------------------------------------------
    # Pack outputs according to `quantities`
    # ------------------------------------------------------------------
    out = []
    for q in quantities:
        if q == "i":
            out.append(i_t)
        elif q == "j":
            out.append(j_t)
        elif q == "S":
            out.append(S_t)
        elif q == "D":
            out.append(D_t)
        elif q == "d":
            out.append(d_t)
        else:
            raise ValueError(f"Unsupported quantity: {q!r}")

    if len(out) == 1:
        return out[0]
    return tuple(out)


####################################################################################################
# ======================================================================
# Build cueq-compatible AtomicData using the GPU neighbour list
# ======================================================================

def build_atomic_data_gpu(
    configs: Sequence[Any],
    cutoff: float,
    atomic_numbers: Iterable[int],
    device: str = "cuda",
) -> List[AtomicData]:
    """
    Build a list of cueq-compatible `AtomicData` objects on GPU.

    This:
      - Mirrors the structure of `AtomicData.from_config` used in MACE/cueq.
      - Uses `torch_neighbour_list` to construct edge_index + shifts + unit_shifts
        entirely on GPU.

    Args
    ----
    configs:
        List of `Configuration` objects, typically from `mace.data.config_from_atoms`.
    cutoff:
        Cutoff radius used for neighbour list (Å).
    atomic_numbers:
        Sequence of element atomic numbers defining species order, e.g. [1, 8].
        The order MUST match what was used in training the MACE model.
    device:
        Torch device for all created tensors ("cuda" or "cpu").

    Returns
    -------
    data_list:
        List of `AtomicData` instances, one per configuration.
    """
    data_list: List[AtomicData] = []

    # Map Z -> species index according to the given atomic_numbers order
    atomic_numbers = list(int(z) for z in atomic_numbers)
    Z_to_idx: Dict[int, int] = {z: i for i, z in enumerate(atomic_numbers)}
    num_species = len(atomic_numbers)

    for conf in configs:
        # --------------------------------------------------------------
        # Basic fields moved to GPU
        # --------------------------------------------------------------
        pos = torch.tensor(
            conf.positions,
            dtype=torch.float32,
            device=device,
        )                                                   # [N, 3]

        Z = torch.tensor(
            conf.atomic_numbers,
            dtype=torch.long,
            device=device,
        )                                                   # [N]

        cell = torch.tensor(
            conf.cell,
            dtype=torch.float32,
            device=device,
        )                                                   # [3, 3]

        # AtomicData expects pbc shape [1, 3]
        pbc = torch.tensor(
            conf.pbc,
            dtype=torch.bool,
            device=device,
        ).view(1, 3)                                        # [1, 3]

        N = pos.size(0)

        # --------------------------------------------------------------
        # Node attributes: one-hot species according to atomic_numbers
        # --------------------------------------------------------------
        # Map atomic numbers to species indices 0..num_species-1
        Z_cpu = Z.cpu().tolist()
        species_indices = [Z_to_idx[int(z)] for z in Z_cpu]

        species_idx = torch.tensor(
            species_indices,
            dtype=torch.long,
            device=device,
        )                                                   # [N]

        node_attrs = torch.nn.functional.one_hot(
            species_idx,
            num_classes=num_species,
        ).float()                                           # [N, num_species]

        # --------------------------------------------------------------
        # Neighbour list on GPU
        # --------------------------------------------------------------
        i_idx, j_idx, S = torch_neighbour_list(
            positions=pos,
            cell=cell,
            pbc=pbc[0],             # [1, 3] -> [3]
            cutoff=float(cutoff),
            quantities="ijS",
            device=device,
            dtype=torch.float32,
        )
        # i_idx, j_idx: [E], int64
        # S: [E, 3], int64

        edge_index = torch.stack([i_idx, j_idx], dim=0)     # [2, E], int64
        unit_shifts_edge = S.to(torch.float32)              # [E, 3], float32
        shifts_edge = unit_shifts_edge @ cell               # [E, 3], float32

        # --------------------------------------------------------------
        # Other cueq-required fields (shapes and types only)
        # These are dummy values; they can be overwritten later.
        # --------------------------------------------------------------
        charges = torch.zeros(N, device=device)
        charges_weight = torch.tensor(1.0, device=device)

        dipole = torch.zeros(1, 3, device=device)
        dipole_weight = torch.ones(1, 3, device=device)

        polarizability = torch.zeros(1, 3, 3, device=device)
        polarizability_weight = torch.ones(1, 3, 3, device=device)

        stress = torch.zeros(1, 3, 3, device=device)
        stress_weight = torch.tensor(1.0, device=device)

        virials = torch.zeros(1, 3, 3, device=device)
        virials_weight = torch.tensor(1.0, device=device)

        forces = torch.zeros(N, 3, device=device)
        forces_weight = torch.tensor(1.0, device=device)

        energy = torch.tensor(0.0, device=device)
        energy_weight = torch.tensor(1.0, device=device)

        elec_temp = torch.tensor(0.0, device=device)

        head = torch.tensor(0, device=device)
        total_charge = torch.tensor(0.0, device=device)
        total_spin = torch.tensor(1.0, device=device)
        weight = torch.tensor(1.0, device=device)

        # --------------------------------------------------------------
        # Construct AtomicData
        # --------------------------------------------------------------
        item = AtomicData(
            positions=pos,
            node_attrs=node_attrs,
            cell=cell,
            pbc=pbc,

            edge_index=edge_index,
            shifts=shifts_edge,
            unit_shifts=unit_shifts_edge,

            charges=charges,
            charges_weight=charges_weight,

            dipole=dipole,
            dipole_weight=dipole_weight,

            polarizability=polarizability,
            polarizability_weight=polarizability_weight,

            stress=stress,
            stress_weight=stress_weight,

            virials=virials,
            virials_weight=virials_weight,

            forces=forces,
            forces_weight=forces_weight,

            elec_temp=elec_temp,

            energy=energy,
            energy_weight=energy_weight,

            head=head,
            total_charge=total_charge,
            total_spin=total_spin,
            weight=weight,
        )

        data_list.append(item)

    return data_list