import torch
import torch.nn as nn

from .utils.data_utils import get_pbc_distances
from .utils.neighborlist_torch import torch_neighbour_list
from torch_scatter import scatter_add


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

    Output
    ------
    If return_forces=True:

        energy, force_cart

    where:

        force_cart = - dE / d r_cart

    The returned energy and force are detached.
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
        self.halve_if_directed = bool(halve_if_directed)
        self.eps = float(eps)

    def _prepare_num_atoms(
        self,
        num_atoms,
        idx_config: torch.Tensor,
        M_all: int,
        M: int,
        N: int,
        device,
    ):
        if not torch.is_tensor(num_atoms):
            num_atoms = torch.tensor(num_atoms, device=device)

        num_atoms = num_atoms.to(device=device).long().view(-1)

        if num_atoms.numel() == 1:
            return num_atoms.expand(M)

        if num_atoms.numel() == M_all:
            return num_atoms[idx_config]

        if num_atoms.numel() == M:
            return num_atoms

        # Fallback for fixed-N systems
        return torch.full(
            (M,),
            fill_value=N,
            device=device,
            dtype=torch.long,
        )

    def _energy(
        self,
        pos_cart: torch.Tensor,
        cell: torch.Tensor,
        num_atoms: torch.Tensor,
        return_per_config: bool = False,
        return_forces: bool = True,
    ):
        """
        pos_cart:
            Cartesian coordinates, shape [M, N, 3].
            Should require grad if return_forces=True.

        cell:
            Shape [M, 3, 3].

        num_atoms:
            Shape [M].
        """

        M, N, _ = pos_cart.shape
        device = pos_cart.device
        dtype = pos_cart.dtype

        edge_index_list = []
        to_jimages_list = []

        num_bonds = torch.zeros(M, device=device, dtype=torch.long)

        pbc = torch.tensor([True, True, True], device=device)

        for m in range(M):
            edge_i, edge_j, to_jimages = torch_neighbour_list(
                positions=pos_cart[m],
                cell=cell[m],
                pbc=pbc,
                cutoff=self.cutoff,
                dtype=dtype,
            )

            local_edge_index = torch.stack([edge_i, edge_j], dim=0)
            assert local_edge_index.shape[0] == 2

            edge_index_list.append(local_edge_index + m * N)
            to_jimages_list.append(to_jimages)

            num_bonds[m] = local_edge_index.shape[1]

        # No configs or no edge containers
        if len(edge_index_list) == 0:
            zero_total = pos_cart.sum() * 0.0
            zero_per_config = pos_cart.sum(dim=(1, 2)) * 0.0
            zero_force = pos_cart * 0.0

            if return_forces:
                energy_out = zero_per_config if return_per_config else zero_total
                return energy_out.detach(), zero_force.detach()

            if return_per_config:
                return zero_per_config.detach()

            return zero_total.detach()

        edge_index = torch.cat(edge_index_list, dim=1)

        # No actual neighbor edges
        if edge_index.numel() == 0:
            zero_total = pos_cart.sum() * 0.0
            zero_per_config = pos_cart.sum(dim=(1, 2)) * 0.0
            zero_force = pos_cart * 0.0

            if return_forces:
                energy_out = zero_per_config if return_per_config else zero_total
                return energy_out.detach(), zero_force.detach()

            if return_per_config:
                return zero_per_config.detach()

            return zero_total.detach()

        to_jimages = torch.cat(to_jimages_list, dim=0)

        dist_out = get_pbc_distances(
            coords=pos_cart.reshape(M * N, 3),
            edge_index=edge_index,
            lattice=cell,
            to_jimages=to_jimages,
            num_atoms=num_atoms.view(-1),
            num_bonds=num_bonds,
            coord_is_cart=True,
            return_distance_vec=False,
            return_offsets=False,
        )

        distances = dist_out["distances"]

        r2 = distances ** 2

        x_poly = 1.0 - r2 / (self.cutoff ** 2)
        x_poly = torch.clamp(x_poly, min=0.0)

        e_edge = self.prefactor * x_poly ** self.n_pow

        # Many neighbor lists return both i -> j and j -> i.
        # If so, divide by 2 to avoid double-counting pair energy.
        if self.halve_if_directed:
            e_edge = 0.5 * e_edge

        e_per_config = scatter_add(
            e_edge,
            index=edge_index[1],
            dim=0,
            dim_size=M * N,
        ).view(M, N).sum(dim=-1)

        e_total = e_per_config.sum()

        if not return_forces:
            if return_per_config:
                return e_per_config.detach()
            return e_total.detach()

        if not e_total.requires_grad:
            zero_force = pos_cart * 0.0
            energy_out = e_per_config if return_per_config else e_total
            return energy_out.detach(), zero_force.detach()

        grad_cart = torch.autograd.grad(
            e_total,
            pos_cart,
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )[0]

        force_cart = -grad_cart

        energy_out = e_per_config if return_per_config else e_total

        return energy_out.detach(), force_cart.detach()

    def forward(
        self,
        x: torch.Tensor,
        cell: torch.Tensor,
        num_atoms: torch.Tensor,
        idx_config: torch.Tensor | None = None,
        return_per_config: bool = False,
        return_forces: bool = True,
    ):
        device = x.device

        # Flatten possible [B, T, N, 3] input to [M_all, N, 3]
        if x.ndim == 4:
            B, T, N, _ = x.shape
            x_flat = x.reshape(B * T, N, 3)
            cell_flat = cell.reshape(B * T, 3, 3)
            input_was_4d = True

        elif x.ndim == 3:
            M_all, N, _ = x.shape
            x_flat = x
            cell_flat = cell
            input_was_4d = False

        else:
            raise ValueError(
                f"x must have shape [B, T, N, 3] or [M, N, 3], got {x.shape}"
            )

        M_all = x_flat.shape[0]

        user_provided_idx_config = idx_config is not None

        if idx_config is None:
            idx_config_work = torch.arange(M_all, device=device)
        else:
            idx_config_work = idx_config.to(device=device).long().view(-1)

        x_sel = x_flat[idx_config_work]          # [M, N, 3], fractional
        cell_sel = cell_flat[idx_config_work]    # [M, 3, 3]

        M = x_sel.shape[0]

        num_atoms_sel = self._prepare_num_atoms(
            num_atoms=num_atoms,
            idx_config=idx_config_work,
            M_all=M_all,
            M=M,
            N=N,
            device=device,
        )

        if return_forces:
            # Important for PyTorch Lightning validation:
            #
            # torch.enable_grad() handles no_grad().
            # torch.inference_mode(False) helps when Lightning uses inference mode.
            #
            # If this still fails in validation, also set:
            #     Trainer(..., inference_mode=False)
            with torch.inference_mode(False):
                with torch.enable_grad():
                    # We detach x/cell here intentionally.
                    # This makes pos_cart an independent leaf variable.
                    # Force can be computed, but no graph flows back to x/model.
                    pos_cart = torch.einsum(
                        "mni,mij->mnj",
                        x_sel.detach(),
                        cell_sel.detach(),
                    ).clone().requires_grad_(True)

                    energy_out, force_sel = self._energy(
                        pos_cart=pos_cart,
                        cell=cell_sel.detach(),
                        num_atoms=num_atoms_sel,
                        return_per_config=return_per_config,
                        return_forces=True,
                    )

        else:
            # No force needed, so no autograd needed.
            pos_cart = torch.einsum(
                "mni,mij->mnj",
                x_sel.detach(),
                cell_sel.detach(),
            )

            energy_out = self._energy(
                pos_cart=pos_cart,
                cell=cell_sel.detach(),
                num_atoms=num_atoms_sel,
                return_per_config=return_per_config,
                return_forces=False,
            )

            return energy_out.detach()

        # If user selected only some configs, return selected force shape [M, N, 3].
        if user_provided_idx_config:
            return energy_out.detach(), force_sel.detach()

        # Otherwise restore the original input shape.
        if input_was_4d:
            force_out = force_sel.reshape(B, T, N, 3)
        else:
            force_out = force_sel.reshape(x.shape)

        return energy_out.detach(), force_out.detach()