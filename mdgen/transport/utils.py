import torch

from mdgen.model.utils.neighborlist_torch import torch_neighbour_list


def graph_loss(
    x_frac,
    x1_frac,
    cell,
    cutoff,
    pbc=None,
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
        Cell matrix in ASE convention:
            x_cart = x_frac @ cell

    cutoff : float
        Neighbor cutoff in Cartesian units.

    pbc : (3,), (..., 3), or None
        Periodicity. Defaults to [True, True, True].

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

    total_edges = 0

    for b in range(M):

        xb = x[b]
        x1b = x1[b]
        cellb = cells[b]
        pbcb = pbcs[b]

        # -----------------------------------------------------
        # Build reference graph.
        #
        # torch_neighbour_list wants Cartesian coordinates.
        # -----------------------------------------------------

        with torch.no_grad():

            x1_cart = x1b @ cellb

            i, j, S, r1 = torch_neighbour_list(
                quantities="ijSd",
                positions=x1_cart,
                cell=cellb,
                pbc=pbcb,
                cutoff=cutoff,
            )

        if i.numel() == 0:
            continue

        # Make sure all tensors are on the correct device
        i = i.to(xb.device)
        j = j.to(xb.device)

        S = S.to(
            device=xb.device,
            dtype=xb.dtype,
        )

        r1 = r1.to(
            device=xb.device,
            dtype=xb.dtype,
        )

        # -----------------------------------------------------
        # Evaluate generated coordinates on SAME edges/images
        #
        # S is the fractional lattice translation:
        #
        # dr_cart =
        #     (x_j - x_i + S) @ cell
        # -----------------------------------------------------

        dr_frac = (
            xb[j]
            - xb[i]
            + S
        )

        dr_cart = dr_frac @ cellb

        r = torch.linalg.vector_norm(
            dr_cart,
            dim=-1,
        )

        error = r - r1

        squared_errors.append(error.square())
        abs_errors.append(error.abs())

        total_edges += error.numel()

    # ---------------------------------------------------------
    # Combine all edges from the entire batch
    # ---------------------------------------------------------

    if total_edges == 0:

        # Maintain autograd connection
        loss = x_frac.sum() * 0.0

        return loss, {
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

    return loss, info