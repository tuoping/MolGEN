import torch
import torch.nn as nn

from .utils.data_utils import (
    get_pbc_distances,
)

# from .utils.neighborhood import get_neighborhood
from .utils.neighborlist_torch import torch_neighbour_list
from torch_scatter import scatter_mean, scatter_add

class PolynomialRepulsiveEnergy(nn.Module):
    """
    Polynomial short-range repulsive energy:

        E_ij = prefactor * (1 - r_ij^2 / cutoff^2)^n_pow,  r_ij < cutoff

    Assumptions
    -----------
    x:
        Fractional coordinates.
        Shape can be [B, T, N, 3] or [M, N, 3].

    cell:
        Lattice matrix.
        Shape can be [B, T, 3, 3] or [M, 3, 3].

    torch_neighbour_list:
        Expected to return edge_index_i, edge_index_j, to_jimages,
        where edge_index_i and edge_index_j index flattened atoms.

    get_pbc_distances:
        Your function, with edge_index ordered as [j_index, i_index].
    """

    def __init__(
        self,
        cutoff: float,
        prefactor: float = 1.0,
        n_pow: int = 2,
        halve_if_directed: bool = True,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.cutoff = float(cutoff)
        self.prefactor = float(prefactor)
        self.n_pow = int(n_pow)
        self.halve_if_directed = halve_if_directed
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        cell: torch.Tensor,
        num_atoms: torch.Tensor,
        idx_config: torch.Tensor | None = None,
        return_per_config: bool = False,
    ):
        device = x.device
        dtype = x.dtype

        # Flatten possible [B, T, N, 3] input to [M_all, N, 3]
        if x.ndim == 4:
            B, T, N, _ = x.shape
            x_flat = x.reshape(B * T, N, 3)
            cell_flat = cell.reshape(B * T, 3, 3)
        elif x.ndim == 3:
            M_all, N, _ = x.shape
            x_flat = x
            cell_flat = cell
        else:
            raise ValueError("x must have shape [B, T, N, 3] or [M, N, 3].")

        if idx_config is None:
            idx_config = torch.arange(x_flat.shape[0], device=device)

        x_sel = x_flat[idx_config]          # [M, N, 3], fractional
        cell_sel = cell_flat[idx_config]    # [M, 3, 3]
        M = x_sel.shape[0]

        # Cartesian positions for neighbor list construction
        pos_cart = torch.einsum("mni,mij->mnj", x_sel, cell_sel)
        edge_index = []
        to_jimages = []
        num_bonds = torch.zeros(M, device=x.device).to(int)
        for idx_config in range(M):
            _edge_i, _edge_j, _to_jimages = torch_neighbour_list(
                positions=pos_cart[idx_config],
                cell=cell_sel[idx_config],
                pbc=torch.tensor([True, True, True], device=device),
                cutoff=self.cutoff,
                dtype=dtype,
            )
            _edge_index = torch.stack([_edge_i, _edge_j])
            assert _edge_index.shape[0] == 2
            edge_index.append(_edge_index + idx_config*N)
            to_jimages.append(_to_jimages)
            num_bonds[idx_config] = _edge_index.shape[1]
        edge_index = torch.cat(edge_index, dim=1)
        to_jimages = torch.cat(to_jimages, dim=0)
        # No edges
        if edge_index.numel() == 0:
            zero = x.sum() * 0.0
            if return_per_config:
                return zero, torch.zeros(M, device=device, dtype=dtype)
            return zero

        dist_out = get_pbc_distances(
            coords=x_sel.reshape(M * N, 3),
            edge_index=edge_index,
            lattice=cell_sel,
            to_jimages=to_jimages,
            num_atoms=num_atoms.view(-1),
            num_bonds=num_bonds,
            coord_is_cart=False,
            return_distance_vec=False,
            return_offsets=False,
        )

        distances = dist_out["distances"]

        r2 = distances ** 2

        x_poly = 1.0 - r2 / (self.cutoff ** 2)
        x_poly = torch.clamp(x_poly, min=0.0)

        e_edge = self.prefactor * x_poly ** self.n_pow

        # Many neighbor lists return both i -> j and j -> i.
        # If so, divide by 2 to avoid double counting pair energy.
        if self.halve_if_directed:
            e_edge = 0.5 * e_edge

        e_per_config = scatter_add(e_edge, index=edge_index[1], dim=0, dim_size = M*N).view(M,N).sum(-1)

        if return_per_config:
            return e_per_config
        
        e_total = e_per_config.sum()
        return e_total