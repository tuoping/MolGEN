import torch
import torch.nn.functional as F
from ase.data import covalent_radii

from mdgen.model.utils.neighborlist_torch import torch_neighbour_list


def _line_graph_edge_pairs(center_index, num_atoms):
    """Return pairs of edge indices that share the same source atom."""

    if center_index.numel() == 0:
        return center_index.new_empty((0, 2))

    counts = torch.bincount(center_index, minlength=num_atoms)
    max_degree = int(counts.max().item())
    if max_degree < 2:
        return center_index.new_empty((0, 2))

    # Place the original edge indices into a dense [atom, neighbor-slot]
    # table. This avoids one GPU synchronization and one full edge scan per
    # central atom while scaling with the actual maximum coordination.
    edge_order = torch.argsort(center_index)
    sorted_centers = center_index[edge_order]
    starts = torch.cumsum(counts, dim=0) - counts
    slot = torch.arange(
        center_index.numel(), device=center_index.device
    ) - starts[sorted_centers]
    edge_table = center_index.new_full((num_atoms, max_degree), -1)
    edge_table[sorted_centers, slot] = edge_order

    slot_pairs = torch.triu_indices(
        max_degree,
        max_degree,
        offset=1,
        device=center_index.device,
    )
    edge_a = edge_table[:, slot_pairs[0]].reshape(-1)
    edge_b = edge_table[:, slot_pairs[1]].reshape(-1)
    valid = (edge_a >= 0) & (edge_b >= 0)
    return torch.stack((edge_a[valid], edge_b[valid]), dim=-1)


def graph_loss(
    x_frac,
    x1_frac,
    cell,
    cutoff,
    pbc=None,
    reference_cell=None,
    distance_scale=None,
    structure_weight=None,
    use_reference_image=False,
    compute_line_graph=False,
):
    """
    Graph-distance loss using the reference neighbor graph.

    Parameters
    ----------
    x_frac : (..., N, 3)
        Generated fractional coordinates.

    x1_frac : (..., N, 3)
        Reference fractional coordinates.

    cell : (..., 3, 3) or (3, 3)
        Cell matrix for ``x_frac`` in ASE convention:
            x_cart = x_frac @ cell

    cutoff : float
        Neighbor cutoff in Cartesian units.

    pbc : (3,), (..., 3), or None
        Periodicity. Defaults to [True, True, True].

    reference_cell : (..., 3, 3), (3, 3), or None
        Cell matrix for ``x1_frac``. Defaults to ``cell``. This allows a
        one-step graph loss to compare predicted and teacher lattice steps.

    distance_scale : scalar, (...) tensor, or None
        Divide each structure's edge-distance errors by this value before
        squaring. Passing the Euler step size makes the loss compare distance
        change rates and prevents it from vanishing as ``dt**2``.

    structure_weight : scalar, (...) tensor, or None
        Multiply each structure's squared edge errors by this value. The
        reduction denominator is unchanged, so values below one suppress the
        absolute contribution rather than being normalized away.

    use_reference_image : bool
        Reuse the periodic image selected by the reference neighbor graph.
        This reproduces the original graph-loss convention. When false, the
        generated coordinates select their own minimum image.

    compute_line_graph : bool
        Also compare the bond angles formed by unordered pairs of outgoing
        reference edges sharing a central atom. The differentiable angle loss
        is returned as ``info["loss_line_graph"]``.

    Returns
    -------
    loss : scalar tensor
        Mean squared edge-distance error.

    info : dict
        Diagnostics.
    """

    assert x_frac.shape == x1_frac.shape
    assert x_frac.shape[-1] == 3

    leading_shape = x_frac.shape[:-2]
    N = x_frac.shape[-2]

    # Flatten all batch/time/etc. dimensions:
    #
    # (B, T, N, 3) -> (B*T, N, 3)
    x = x_frac.reshape(-1, N, 3)
    x1 = x1_frac.reshape(-1, N, 3)

    M = x.shape[0]

    # ---------------------------------------------------------
    # Cell handling
    # ---------------------------------------------------------

    if cell.ndim == 2:
        # one common cell for every structure
        cells = cell.unsqueeze(0).expand(M, -1, -1)

    else:
        cells = cell.reshape(-1, 3, 3)

        if cells.shape[0] == 1:
            cells = cells.expand(M, -1, -1)

        elif cells.shape[0] != M:
            raise ValueError(
                f"Incompatible cell shape {cell.shape} "
                f"for coordinates {x_frac.shape}"
            )

    if reference_cell is None:
        reference_cells = cells
    elif reference_cell.ndim == 2:
        reference_cells = reference_cell.unsqueeze(0).expand(M, -1, -1)
    else:
        reference_cells = reference_cell.reshape(-1, 3, 3)
        if reference_cells.shape[0] == 1:
            reference_cells = reference_cells.expand(M, -1, -1)
        elif reference_cells.shape[0] != M:
            raise ValueError(
                f"Incompatible reference_cell shape {reference_cell.shape} "
                f"for coordinates {x1_frac.shape}"
            )

    if distance_scale is None:
        distance_scales = torch.ones(M, dtype=x.dtype, device=x.device)
    else:
        distance_scales = torch.as_tensor(
            distance_scale,
            dtype=x.dtype,
            device=x.device,
        ).reshape(-1)
        if distance_scales.numel() == 1:
            distance_scales = distance_scales.expand(M)
        elif distance_scales.numel() != M:
            raise ValueError(
                "distance_scale must provide one value per structure"
            )
        if torch.any(distance_scales <= 0):
            raise ValueError("distance_scale must be positive")

    if structure_weight is None:
        structure_weights = torch.ones(M, dtype=x.dtype, device=x.device)
    else:
        structure_weights = torch.as_tensor(
            structure_weight,
            dtype=x.dtype,
            device=x.device,
        ).reshape(-1)
        if structure_weights.numel() == 1:
            structure_weights = structure_weights.expand(M)
        elif structure_weights.numel() != M:
            raise ValueError(
                "structure_weight must provide one value per structure"
            )
        if torch.any(structure_weights < 0):
            raise ValueError("structure_weight must be non-negative")

    # ---------------------------------------------------------
    # PBC handling
    # ---------------------------------------------------------

    if pbc is None:
        pbcs = torch.ones(
            M,
            3,
            dtype=torch.bool,
            device=x_frac.device,
        )

    elif pbc.ndim == 1:
        pbcs = pbc.to(x_frac.device).unsqueeze(0).expand(M, -1)

    else:
        pbcs = pbc.to(x_frac.device).reshape(-1, 3)

        if pbcs.shape[0] == 1:
            pbcs = pbcs.expand(M, -1)

    # ---------------------------------------------------------
    # Process each structure independently
    #
    # This is intentional:
    # every structure can have a different number of edges.
    # ---------------------------------------------------------

    squared_errors = []
    abs_errors = []
    line_squared_errors = []
    line_abs_errors = []

    total_edges = 0
    total_line_edges = 0

    for b in range(M):

        xb = x[b]
        x1b = x1[b]
        cellb = cells[b]
        reference_cellb = reference_cells[b]
        pbcb = pbcs[b]

        # -----------------------------------------------------
        # Build reference graph.
        #
        # torch_neighbour_list wants Cartesian coordinates.
        # -----------------------------------------------------

        with torch.no_grad():

            x1_cart = x1b @ reference_cellb

            i, j, S, r1 = torch_neighbour_list(
                quantities="ijSd",
                positions=x1_cart,
                cell=reference_cellb,
                pbc=pbcb,
                cutoff=cutoff,
            )

        if i.numel() == 0:
            continue

        # Make sure all tensors are on the correct device
        i = i.to(xb.device)
        j = j.to(xb.device)
        S = S.to(device=xb.device, dtype=xb.dtype)

        r1 = r1.to(
            device=xb.device,
            dtype=xb.dtype,
        )

        # -----------------------------------------------------
        # Evaluate generated coordinates on the same edge topology. Select
        # the minimum image from the generated coordinates themselves: the
        # predicted and reference representations may differ by an integer
        # lattice translation even when they describe identical positions.
        #
        # The rounding operation only selects a discrete image. Its derivative
        # is intentionally excluded while the distance remains differentiable.
        # -----------------------------------------------------

        dr_frac = xb[j] - xb[i]
        if use_reference_image:
            dr_frac = dr_frac + S
        else:
            with torch.no_grad():
                image_shift = -torch.round(dr_frac) * pbcb.to(xb.dtype)
            dr_frac = dr_frac + image_shift

        dr_cart = dr_frac @ cellb

        r = torch.linalg.vector_norm(
            dr_cart,
            dim=-1,
        )

        error = (r - r1) / distance_scales[b]

        squared_errors.append(error.square() * structure_weights[b])
        abs_errors.append(error.abs())

        total_edges += error.numel()

        if compute_line_graph:
            # A line-graph edge is an unordered pair of bonds sharing the
            # source/central atom: (j <- i -> k). The reference neighbor graph
            # fixes which triplets are supervised, while the generated bond
            # vectors retain gradients with respect to generated coordinates.
            line_edge_pairs = _line_graph_edge_pairs(i, N)

            if line_edge_pairs.numel() > 0:
                edge_a = line_edge_pairs[:, 0]
                edge_b = line_edge_pairs[:, 1]

                reference_dr_frac = x1b[j] - x1b[i] + S
                reference_dr_cart = reference_dr_frac @ reference_cellb

                def angle_cosine(v1, v2):
                    eps = torch.finfo(v1.dtype).eps
                    u1 = v1 / torch.linalg.vector_norm(
                        v1, dim=-1, keepdim=True
                    ).clamp_min(eps)
                    u2 = v2 / torch.linalg.vector_norm(
                        v2, dim=-1, keepdim=True
                    ).clamp_min(eps)
                    return (u1 * u2).sum(dim=-1).clamp(-1.0, 1.0)

                generated_cos_angle = angle_cosine(
                    dr_cart[edge_a], dr_cart[edge_b]
                )
                with torch.no_grad():
                    reference_cos_angle = angle_cosine(
                        reference_dr_cart[edge_a],
                        reference_dr_cart[edge_b],
                    )

                # Cosine supervision avoids the singular gradient of acos at
                # linear bond angles.
                line_error_squared = (
                    generated_cos_angle - reference_cos_angle
                ).square() / distance_scales[b].square()
                line_squared_errors.append(
                    line_error_squared * structure_weights[b]
                )
                line_abs_errors.append(
                    torch.sqrt(line_error_squared.clamp_min(0.0))
                )
                total_line_edges += line_error_squared.numel()

    # ---------------------------------------------------------
    # Combine all edges from the entire batch
    # ---------------------------------------------------------

    if total_edges == 0:

        # Maintain autograd connection
        loss = x_frac.sum() * 0.0

        info = {
            "loss": loss.detach(),
            "num_edges": 0,
            "mae_distance": torch.tensor(
                0.0,
                device=x_frac.device,
            ),
            "rmse_distance": torch.tensor(
                0.0,
                device=x_frac.device,
            ),
        }
        if compute_line_graph:
            info.update({
                "loss_line_graph": loss,
                "num_line_edges": 0,
                "mae_cos_angle": torch.tensor(
                    0.0, device=x_frac.device
                ),
                "rmse_cos_angle": torch.tensor(
                    0.0, device=x_frac.device
                ),
            })
        return loss, info

    squared_errors = torch.cat(squared_errors)
    abs_errors = torch.cat(abs_errors)

    loss = squared_errors.mean()

    info = {
        "loss": loss.detach(),
        "num_edges": total_edges,
        "mae_distance": abs_errors.mean().detach(),
        "rmse_distance": torch.sqrt(
            squared_errors.mean()
        ).detach(),
    }

    if compute_line_graph:
        if total_line_edges == 0:
            line_loss = x_frac.sum() * 0.0
            line_mae = torch.zeros(
                (), dtype=x_frac.dtype, device=x_frac.device
            )
            line_rmse = line_mae
        else:
            line_squared_errors = torch.cat(line_squared_errors)
            line_abs_errors = torch.cat(line_abs_errors)
            line_loss = line_squared_errors.mean()
            line_mae = line_abs_errors.mean().detach()
            line_rmse = torch.sqrt(line_loss).detach()

        info.update({
            # Keep the loss attached to autograd; the remaining info entries
            # are diagnostics only.
            "loss_line_graph": line_loss,
            "num_line_edges": total_line_edges,
            "mae_cos_angle": line_mae,
            "rmse_cos_angle": line_rmse,
        })

    return loss, info


def collision_loss(
    x_frac,
    cell,
    min_distance=0.8,
    softness=0.05,
    pbc=None,
    atom_mask=None,
    structure_weight=None,
    atomic_numbers=None,
    radius_scale=0.8,
):
    """Smooth collision penalty on nearby generated atom pairs.

    Each pair contributes

        [softness * softplus((pair_cutoff - distance) / softness)]**2.

    When ``atomic_numbers`` is supplied, the cutoff is species-dependent:

        pair_cutoff = radius_scale * (Rcov_i + Rcov_j).

    This uses the same outer-cutoff convention as ``ZBLRepulsiveWall``. The
    scalar ``min_distance`` remains as a backwards-compatible fallback for
    callers without species information.

    Candidate pairs are built with ``torch_neighbour_list`` using a cutoff
    beyond the collision distance. At the search cutoff the omitted softplus
    tail is negligible. Losses are normalized per atom rather than per pair,
    so a small number of severe overlaps is not diluted by a large graph.
    """

    if x_frac.shape[-1] != 3:
        raise ValueError("x_frac must have shape (..., N, 3)")
    if min_distance <= 0:
        raise ValueError("min_distance must be positive")
    if radius_scale <= 0:
        raise ValueError("radius_scale must be positive")
    if softness <= 0:
        raise ValueError("softness must be positive")

    N = x_frac.shape[-2]
    x = x_frac.reshape(-1, N, 3)
    M = x.shape[0]

    if cell.ndim == 2:
        cells = cell.unsqueeze(0).expand(M, -1, -1)
    else:
        cells = cell.reshape(-1, 3, 3)
        if cells.shape[0] == 1:
            cells = cells.expand(M, -1, -1)
        elif cells.shape[0] != M:
            raise ValueError(
                f"Incompatible cell shape {cell.shape} "
                f"for coordinates {x_frac.shape}"
            )

    if pbc is None:
        pbcs = torch.ones(M, 3, dtype=torch.bool, device=x.device)
    else:
        pbc = torch.as_tensor(pbc, dtype=torch.bool, device=x.device)
        if pbc.ndim == 1:
            pbcs = pbc.unsqueeze(0).expand(M, -1)
        else:
            pbcs = pbc.reshape(-1, 3)
            if pbcs.shape[0] == 1:
                pbcs = pbcs.expand(M, -1)
            elif pbcs.shape[0] != M:
                raise ValueError(
                    f"Incompatible pbc shape {pbc.shape} "
                    f"for coordinates {x_frac.shape}"
                )

    if atom_mask is None:
        masks = torch.ones(M, N, dtype=torch.bool, device=x.device)
    else:
        masks = torch.as_tensor(atom_mask, device=x.device)
        if masks.shape[-1] == 1:
            masks = masks.squeeze(-1)
        masks = masks.reshape(-1, N).bool()
        if masks.shape[0] == 1:
            masks = masks.expand(M, -1)
        elif masks.shape[0] != M:
            raise ValueError(
                f"Incompatible atom_mask shape {atom_mask.shape} "
                f"for coordinates {x_frac.shape}"
            )

    if atomic_numbers is None:
        atomic_numbers_flat = None
        covalent_radii_tensor = None
    else:
        atomic_numbers_flat = torch.as_tensor(
            atomic_numbers,
            dtype=torch.long,
            device=x.device,
        ).reshape(-1, N)
        if atomic_numbers_flat.shape[0] == 1:
            atomic_numbers_flat = atomic_numbers_flat.expand(M, -1)
        elif atomic_numbers_flat.shape[0] != M:
            raise ValueError(
                f"Incompatible atomic_numbers shape {atomic_numbers.shape} "
                f"for coordinates {x_frac.shape}"
            )

        valid_atomic_numbers = atomic_numbers_flat[masks]
        max_atomic_number = len(covalent_radii) - 1
        if torch.any(valid_atomic_numbers <= 0) or torch.any(
            valid_atomic_numbers > max_atomic_number
        ):
            raise ValueError(
                "atomic_numbers for unmasked atoms must be positive and no "
                f"greater than {max_atomic_number}"
            )

        covalent_radii_tensor = torch.as_tensor(
            covalent_radii,
            dtype=x.dtype,
            device=x.device,
        )

    if structure_weight is None:
        weights = torch.ones(M, dtype=x.dtype, device=x.device)
    else:
        weights = torch.as_tensor(
            structure_weight,
            dtype=x.dtype,
            device=x.device,
        ).reshape(-1)
        if weights.numel() == 1:
            weights = weights.expand(M)
        elif weights.numel() != M:
            raise ValueError(
                "structure_weight must provide one value per structure"
            )

    if N < 2:
        loss = x_frac.sum() * 0.0
        return loss, {
            "loss": loss.detach(),
            "num_pairs": 0,
            "num_collisions": 0,
            "min_distance": torch.full(
                (), float("inf"), device=x.device, dtype=x.dtype
            ),
        }

    per_structure = []
    all_valid_distances = []
    total_pairs = 0
    total_collisions = 0

    for b in range(M):
        if not torch.any(masks[b]):
            per_structure.append(x[b].sum() * 0.0)
            continue

        if atomic_numbers_flat is None:
            search_cutoff = min_distance + 8.0 * softness
        else:
            radii_b = covalent_radii_tensor[
                atomic_numbers_flat[b, masks[b]]
            ]
            search_cutoff = (
                2.0 * radius_scale * radii_b.max().item()
                + 8.0 * softness
            )

        # The edge topology is discrete, but distances are recomputed below
        # from x so that the collision penalty remains differentiable.
        with torch.no_grad():
            i, j, S = torch_neighbour_list(
                quantities="ijS",
                positions=x[b] @ cells[b],
                cell=cells[b],
                pbc=pbcs[b],
                cutoff=search_cutoff,
                dtype=x.dtype,
            )

        i = i.to(x.device)
        j = j.to(x.device)
        S = S.to(device=x.device, dtype=x.dtype)

        # torch_neighbour_list returns both i -> j and j -> i. Keep one
        # orientation and remove any edges involving padded atoms.
        valid_pair = (i < j) & masks[b, i] & masks[b, j]
        if not torch.any(valid_pair):
            per_structure.append(x[b].sum() * 0.0)
            continue

        ib = i[valid_pair]
        jb = j[valid_pair]
        Sb = S[valid_pair]
        dr_frac = x[b, jb] - x[b, ib] + Sb
        distance = torch.linalg.vector_norm(dr_frac @ cells[b], dim=-1)

        if atomic_numbers_flat is None:
            pair_cutoff = torch.full_like(distance, min_distance)
        else:
            zi = atomic_numbers_flat[b, ib]
            zj = atomic_numbers_flat[b, jb]
            pair_cutoff = radius_scale * (
                covalent_radii_tensor[zi] + covalent_radii_tensor[zj]
            )

        penetration = softness * F.softplus(
            (pair_cutoff - distance) / softness
        )
        num_atoms = masks[b].sum().clamp_min(1).to(x.dtype)
        per_structure.append(penetration.square().sum() / num_atoms)

        all_valid_distances.append(distance.detach())
        total_pairs += distance.numel()
        total_collisions += int(
            (distance.detach() < pair_cutoff.detach()).sum()
        )

    per_structure = torch.stack(per_structure)
    weight_sum = weights.sum()
    if bool(weight_sum > 0):
        loss = (per_structure * weights).sum() / weight_sum
    else:
        loss = per_structure.sum() * 0.0

    if all_valid_distances:
        observed_min_distance = torch.cat(all_valid_distances).min()
    else:
        observed_min_distance = torch.full(
            (), float("inf"), device=x.device, dtype=x.dtype
        )

    return loss, {
        "loss": loss.detach(),
        "num_pairs": total_pairs,
        "num_collisions": total_collisions,
        "min_distance": observed_min_distance,
    }
