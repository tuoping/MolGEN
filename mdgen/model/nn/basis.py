import torch
from torch import nn

# Typing
from torch import Tensor
from typing import List, Optional, Tuple


def bessel(x: Tensor, start: float = 0.0, end: float = 1.0, num_basis: int = 8, eps: float = 1e-5) -> Tensor:
    """Expand scalar features into (radial) Bessel basis function values.
    """
    x = x[..., None] - start + eps
    c = end - start
    n = torch.arange(1, num_basis+1, dtype=x.dtype, device=x.device)
    return ((2/c)**0.5) * torch.sin(n*pi*x / c) / x


def gaussian(x: Tensor, start: float = 0.0, end: float = 1.0, num_basis: int = 8) -> Tensor:
    """Expand scalar features into Gaussian basis function values.
    """
    mu = torch.linspace(start, end, num_basis, dtype=x.dtype, device=x.device)
    step = mu[1] - mu[0]
    diff = (x[..., None] - mu) / step
    return diff.pow(2).neg().exp().div(1.12) # division by 1.12 so that sum of square is roughly 1


def scalar2basis(x: Tensor, start: float, end: float, num_basis: int, basis: str = 'gaussian'):
    """Expand scalar features into basis function values.
    Reference: https://docs.e3nn.org/en/stable/api/math/math.html#e3nn.math.soft_one_hot_linspace.
    """
    funcs = {
        'gaussian': gaussian,
        'bessel': bessel,
    }
    return funcs[basis](x, start, end, num_basis)


class Bessel(nn.Module):
    def __init__(self, start: float = 0.0, end: float = 1.0, num_basis: int = 8, eps: float = 1e-5) -> None:
        super().__init__()
        self.start     = start
        self.end       = end
        self.num_basis = num_basis
        self.eps       = eps
        self.register_buffer('n', torch.arange(1, num_basis+1, dtype=torch.float))

    def forward(self, x: Tensor) -> Tensor:
        x = x[..., None] - self.start + self.eps
        c = self.end - self.start
        return ((2/c)**0.5) * torch.sin(self.n*pi*x / c) / x

    def extra_repr(self) -> str:
        return f'start={self.start}, end={self.end}, num_basis={self.num_basis}, eps={self.eps}'


class Gaussian(nn.Module):
    def __init__(self, start: float = 0.0, end: float = 1.0, num_basis: int = 8) -> None:
        super().__init__()
        self.start     = start
        self.end       = end
        self.num_basis = num_basis
        self.register_buffer('mu', torch.linspace(start, end, num_basis))
    
    def forward(self, x: Tensor) -> Tensor:
        step = self.mu[1] - self.mu[0]
        diff = (x[..., None] - self.mu) / step
        return diff.pow(2).neg().exp().div(1.12) # division by 1.12 so that sum of square is roughly 1

    def extra_repr(self) -> str:
        return f'start={self.start}, end={self.end}, num_basis={self.num_basis}'

from math import pi
class GaussianRandomFourierFeatures(nn.Module):
    """Gaussian random Fourier features.

    Reference: https://arxiv.org/abs/2006.10739
    """
    def __init__(self, embed_dim: int, input_dim: int = 1, sigma: float = 1.0) -> None:
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.embed_dim = embed_dim
        if embed_dim > 1:
            self.register_buffer('B', torch.randn(input_dim, embed_dim//2) * sigma)
        else:
            self.register_buffer('B', torch.randn(1, 16) * sigma)
            self.proj = nn.Linear(16*2, 1)  # Project back to 1D

    def forward(self, v: Tensor) -> Tensor:
        if self.embed_dim > 1:
            v_proj =  2 * pi * v @ self.B
            return torch.cat([torch.cos(v_proj), torch.sin(v_proj)], dim=-1)
        else:
            v_proj = 2 * pi * v @ self.B
            features = torch.cat([torch.cos(v_proj), torch.sin(v_proj)], dim=-1)
            return self.proj(features)


class RBFEmb(nn.Module):
    r"""
    radial basis function to embed distances
    modified: delete cutoff with r
    """

    def __init__(self, num_rbf, rbound_upper, rbf_trainable=False):
        super().__init__()
        self.rbound_upper = rbound_upper
        self.rbound_lower = 0
        self.num_rbf = num_rbf
        self.rbf_trainable = rbf_trainable
        means, betas = self._initial_params()

        self.register_buffer("means", means)
        self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.rbound_upper))
        end_value = torch.exp(torch.scalar_tensor(-self.rbound_lower))
        means = torch.linspace(start_value, end_value, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (end_value - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        rbounds = 0.5 * (torch.cos(dist * pi / self.rbound_upper) + 1.0)
        rbounds = rbounds * (dist < self.rbound_upper).float()
        return rbounds * torch.exp(
            -self.betas * torch.square((torch.exp(-dist) - self.means))
        )

import math
import torch
from torch import nn, Tensor
from torch_scatter import scatter
from e3nn import o3
from e3nn.o3 import Irreps
from e3nn.o3 import FullyConnectedTensorProduct as FCTP

def _sorted_irreps(irreps_like: str | o3.Irreps):
    """Return an o3.Irreps, regardless of e3nn version."""
    ir = o3.Irreps(irreps_like)
    s = ir.sort()
    # Old e3nn: sort() -> namedtuple(sort)(irreps, p, inv)
    # New e3nn: sort() -> Irreps
    return getattr(s, "irreps", s)

class EdgeCGBlock(nn.Module):
    """
    Build edge irreps features phi(r)*Y_lm(r_hat) and contract with node features
    via Clebsch–Gordan rules using e3nn's FullyConnectedTensorProduct.

    Tensors in, tensors out. Irreps are stored on the module.
    """
    def __init__(self, node_irreps: list, msg_irreps: list,
                 lmax: int = 2, num_rbf: int = 32, cutoff: float = 5.0,
                 mul_per_l: int = 4):
        super().__init__()
        # Parse node/msg irreps from STRINGS (do not pass lists/tuples of strings)
        self.node_irreps = _sorted_irreps(node_irreps)
        self.msg_irreps  = _sorted_irreps(msg_irreps)

        parts = [f"{mul_per_l}x{l}{('e' if l % 2 == 0 else 'o')}" for l in range(int(lmax)+1)]
        edge_irreps_str = " + ".join(parts)
        self.edge_irreps = _sorted_irreps(edge_irreps_str)


        # ---- the rest of your init unchanged ----
        self.lmax = int(lmax)
        self.num_rbf = int(num_rbf)
        self.cutoff = float(cutoff)

        centers = torch.linspace(0.0, cutoff, self.num_rbf)
        self.register_buffer("rbf_centers", centers)
        delta = float(centers[1] - centers[0]) if self.num_rbf > 1 else max(cutoff, 1.0)
        self.register_buffer("rbf_gamma", torch.tensor(1.0 / (delta ** 2)))

        self.rbf_to_mul = nn.ModuleList([
            nn.Sequential(nn.Linear(self.num_rbf, 32), nn.SiLU(), nn.Linear(32, mul_per_l))
            for _ in range(self.lmax + 1)
        ])
        print("node:", self.node_irreps)  # e.g. "32x0e"
        print("edge:", self.edge_irreps)  # e.g. "4x0e + 4x1o + 4x2e"
        print("msg :", self.msg_irreps)   # e.g. "64x0e + 16x1o"
        print("dims:", self.node_irreps.dim, self.edge_irreps.dim, self.msg_irreps.dim)

        # Now this will parse cleanly
        self.tp = FCTP(self.node_irreps, self.edge_irreps, self.msg_irreps)
        self._rbf = RBFEmb(num_rbf, self.cutoff)

    @property
    def node_dim(self): return self.node_irreps.dim
    @property
    def msg_dim(self):  return self.msg_irreps.dim
    @property
    def edge_dim(self): return self.edge_irreps.dim


    def _edge_features(self, edge_vec: Tensor) -> Tensor:
        """
        Build flat edge feature tensor with irreps = self.edge_irreps.
        Returns: [E, edge_dim]
        """
        eps = 1e-12
        E = edge_vec.size(0)
        r = edge_vec.norm(dim=-1, keepdim=True).clamp_min(eps)  # [E,1]
        r_hat = edge_vec / r                                    # [E,3]
        
        rbf = self._rbf(r)                                      # [E,num_rbf]
        rbounds = 0.5 * (torch.cos(edge_vec.norm(-1) * pi / self.cutoff) + 1.0)
        rbf = rbounds.unsqueeze(-1) * rbf

        # Y_lm up to lmax
        Ys = [o3.spherical_harmonics([l], r_hat, normalize=True, normalization='component')  # [E, 2l+1]
              for l in range(self.lmax + 1)]

        blocks = []
        for l, Y_l in enumerate(Ys):
            w_l = self.rbf_to_mul[l](rbf)                 # [E, mul_per_l]
            block = w_l.unsqueeze(-1) * Y_l.unsqueeze(1)  # [E, mul_per_l, 2l+1]
            blocks.append(block.reshape(E, -1))           # flatten per-l block
        return torch.cat(blocks, dim=-1)                  # [E, edge_dim]

    def forward(
        self,
        node_feat: Tensor,     # [N, node_dim]  (treat as irreps=self.node_irreps)
        edge_index: Tensor,    # [2, E]
        edge_vec: Tensor,      # [E, 3]
        mask: Tensor | None = None  # [E] (1 keep, 0 drop) applied AFTER tp
    ):
        """
        Returns:
          msg_e: [E, msg_dim] per-edge message (equivariant per msg_irreps)
          msg_n: [N, msg_dim] node-aggregated message (same irreps)
        """
        i, j = edge_index
        E = edge_vec.size(0)
        if mask is None:
            mask = edge_vec.new_ones(E)

        # Build edge features (tensor) with irreps=self.edge_irreps
        Y_edge = self._edge_features(edge_vec)            # [E, edge_dim]

        # Gather sender node features
        f_i = node_feat[i]                                # [E, node_dim]

        # CG contraction (equivariant): (node_irreps ⊗ edge_irreps) -> msg_irreps
        msg_e = self.tp(f_i, Y_edge)                      # [E, msg_dim]

        # Gate after TP (scalar mask)
        msg_e = msg_e * mask[:, None]

        # Aggregate to receivers
        # msg_n = scatter(msg_e, index=j, dim=0, dim_size=node_feat.size(0))  # [N, msg_dim]

        return msg_e
