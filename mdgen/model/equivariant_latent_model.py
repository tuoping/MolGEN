# --------------------------------------------------------
# Adapted from: https://github.com/LLNL/graphite
# --------------------------------------------------------

import torch
from torch import nn
from torch_geometric.utils import scatter
from torch_scatter import scatter_mean, scatter_add
import copy

from .nn.mlp import MLP
from .nn.convs import EquivariantTransformerLayer
from .nn.basis import GaussianRandomFourierFeatures, EdgeCGBlock

# Typing
from torch import Tensor
from typing import List, Optional, Tuple


class Encoder(nn.Module):
    def __init__(self, num_species: int, node_dim: int, init_edge_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.num_species   = num_species
        self.node_dim      = node_dim
        self.init_edge_dim = init_edge_dim
        self.edge_dim      = edge_dim
        
        # self.embed_atom = nn.Embedding(num_species, node_dim)
        self.embed_atom = nn.Linear(self.num_species, node_dim)
        self.embed_bond = MLP([64+48+8, edge_dim, node_dim], act=nn.SiLU())
        self.phi_s = MLP([node_dim*2 + node_dim, edge_dim, node_dim], act=nn.SiLU())
        self.phi_h = MLP([node_dim*2,            edge_dim, node_dim], act=nn.SiLU())
        self.phi_v = MLP([node_dim*2 + node_dim, edge_dim, node_dim], act=nn.SiLU())

    def forward(self, species: Tensor, edge_index: Tensor, edge_attr: Tensor, edge_vec: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        num_nodes = len(species)
        i = edge_index[0]
        j = edge_index[1]
        
        # Embed node and edge features
        f         = self.embed_atom(species)
        edge_attr = self.embed_bond(edge_attr)

        # Convolve node features
        e  = torch.cat([f[i], f[j], edge_attr], dim=-1)
        h0 = self.phi_h(torch.cat([
            f, scatter(self.phi_s(e) * f[i], index=j, dim=0, dim_size=num_nodes)
        ], dim=-1))

        # Initialize vector features
        v0 = scatter(edge_vec[:, None, :] * self.phi_v(e)[:, :, None], index=j, dim=0, dim_size=num_nodes)
        return h0, v0, edge_attr

from e3nn import o3

def msg_to_invariants_from_irreps(msg_e, irreps_like):
    irreps = o3.Irreps(irreps_like)
    E, C = msg_e.shape
    out, off = [], 0
    for mul, ir in irreps:  # ir.dim = 2ℓ+1
        d = mul * ir.dim
        if d == 0: continue
        blk = msg_e[:, off:off+d].reshape(E, mul, ir.dim)
        off += d
        inv = blk.squeeze(-1) if ir.l == 0 else blk.norm(dim=-1)   # [E, mul]
        if inv.dim() == 1: inv = inv.unsqueeze(-1)
        out.append(inv)
    if off != C:
        raise ValueError(f"msg_e channels {C} != irreps total {off}")
    return msg_e.new_zeros((E,0)) if not out else torch.cat(out, dim=1)  # [E, Σ mul]

class Encoder_dpm(Encoder):
    def __init__(self, num_species: int, node_dim: int, init_edge_dim: int, edge_dim: int, input_dim: int, object_aware: bool=False) -> None:
        super().__init__(num_species, node_dim, init_edge_dim, edge_dim)
        self.embed_time = nn.Sequential(
            GaussianRandomFourierFeatures(node_dim, input_dim=input_dim),
            MLP([node_dim, edge_dim, node_dim], act=nn.SiLU()),
        )
        if object_aware:
            self.phi_s_cross = MLP([node_dim*2, edge_dim, node_dim], act=nn.SiLU())
            self.phi_h_cross = MLP([node_dim*2, edge_dim, node_dim], act=nn.SiLU())
            self.phi_fuse = MLP([node_dim*2, edge_dim, node_dim], act=nn.SiLU())
        self.object_aware = object_aware

    def forward(self, species: Tensor, edge_index: Tensor, edge_attr: Tensor, edge_vec: Tensor, t: Tensor, sub_graph_mask=None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if not self.object_aware:
            assert sub_graph_mask is None
            return self.forward_generic(species, edge_index, edge_attr, edge_vec, t)
        else:
            assert sub_graph_mask is not None
            return self.forward_object_aware(species, edge_index, edge_attr, edge_vec, t, sub_graph_mask)


    def forward_object_aware(self, species: Tensor, edge_index: Tensor, edge_attr: Tensor, edge_vec: Tensor, t: Tensor, sub_graph_mask=None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # h_cross = self.cross_graph(species, edge_index, edge_attr[:,3:], edge_vec, sub_graph_mask == 0)
        sub_graph_mask = (sub_graph_mask > 0).to(species.dtype)  
        num_nodes = len(species)
        i = edge_index[0]
        j = edge_index[1]

        # Embed node and edge features
        f         = self.embed_atom(species)
        edge_attr = self.embed_bond(edge_attr)

        # Convolve node features
        e  = torch.cat([f[i]*sub_graph_mask[:,None], f[j]*sub_graph_mask[:,None], edge_attr*sub_graph_mask[:,None]], dim=-1)

        h0 = self.phi_h(torch.cat([
            f, scatter(self.phi_s(e) * f[i]*sub_graph_mask[:,None], index=j, dim=0, dim_size=num_nodes)
        ], dim=-1))

        # Initialize vector features
        v0 = scatter((edge_vec * sub_graph_mask[:, None] )[:, None, :] * self.phi_v(e)[:, :, None], index=j, dim=0, dim_size=num_nodes)

        # ---------------- 2) CROSS-OBJECT SCALAR PASS (NO GEOMETRY) ----------------
        h0_i, h0_j = h0[i], h0[j]
        m_cross = self.phi_s_cross(torch.cat([h0_i, h0_j], dim=1)) * (1-sub_graph_mask[:,None])

        h0_cross= self.phi_h_cross(
            torch.cat([h0, scatter(m_cross, index=j, dim=0, dim_size=num_nodes)], dim=1)

        )
        h_out = self.phi_fuse(torch.cat([h0, h0_cross], dim=-1))
        # Add time embedding to node features
        h_out = h_out + self.embed_time(t)
        return h_out, v0, edge_attr


    def forward_generic(self, species: Tensor, edge_index: Tensor, edge_attr: Tensor, edge_vec: Tensor, t: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        num_nodes = len(species)
        i = edge_index[0]
        j = edge_index[1]

        # Embed node and edge features
        f         = self.embed_atom(species)
        edge_attr = self.embed_bond(edge_attr)

        # Convolve node features
        e  = torch.cat([f[i], f[j], edge_attr], dim=-1)

        h0 = self.phi_h(torch.cat([
            f, scatter(self.phi_s(e) * f[i], index=j, dim=0, dim_size=num_nodes)
        ], dim=-1))

        # Initialize vector features
        v0 = scatter(edge_vec[:, None, :] * self.phi_v(e)[:, :, None], index=j, dim=0, dim_size=num_nodes)

        # Add time embedding to node features
        h0 = h0 + self.embed_time(t)
        return h0, v0, edge_attr


class Processor(nn.Module):
    def __init__(self, num_convs: int, node_dim: int, num_heads: int, ff_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.num_convs = num_convs
        self.node_dim  = node_dim
        self.ff_dim    = ff_dim
        self.edge_dim  = edge_dim
        self.convs = nn.ModuleList(
            [copy.deepcopy(EquivariantTransformerLayer(node_dim, num_heads, ff_dim, edge_dim)) for _ in range(num_convs)]
        )

    def forward(self, h: Tensor, v: Tensor, edge_index: Tensor, edge_attr: Tensor, edge_len: Tensor) -> Tuple[Tensor, Tensor]:
        for conv in self.convs:
            h, v = conv(h, v, edge_index, edge_attr, edge_len)
        return h, v


class Decoder(nn.Module):
    def __init__(self, dim: int, num_scalar_out: int, num_vector_out: int, num_species = 5) -> None:
        super().__init__()
        self.dim = dim
        self.num_scalar_out = num_scalar_out
        self.num_vector_out = num_vector_out
        # Scalars
        self.Oh = nn.Parameter(torch.empty(dim, num_scalar_out))
        # Vectors: channel mix shared across xyz (equivariant)
        self.Ov = nn.Parameter(torch.empty(dim, num_vector_out))

        # Better init
        nn.init.kaiming_uniform_(self.Oh, a=5**0.5)
        nn.init.kaiming_uniform_(self.Ov, a=5**0.5)
        self.num_species = num_species

    def forward(self, h: torch.Tensor, v: torch.Tensor):
        """
        h: [N, D]
        v: [N, D, 3]  (last axis is xyz)
        returns:
          h_out: [N, num_scalar_out] (species part softmaxed if present)
          v_out: [N, num_vector_out, 3]
        """
        assert v.dim() == 3 and v.size(-1) == 3, f"expected v [N,D,3], got {list(v.shape)}"
        v = v.contiguous()

        # Scalars
        h_out = h @ self.Oh  # [N, S]

        # Vectors: mix channels only; identical weights for x/y/z
        # (V @ Ov) commutes with rotation on the last axis → stays equivariant
        v_out = torch.einsum('ndk,df->nfk', v, self.Ov)  # [N, F, 3]

        # Optional species softmax on scalars ONLY
        if h_out.shape[-1] >= self.num_species:
            head = h_out[..., :-self.num_species]
            species = torch.softmax(h_out[..., -self.num_species:], dim=-1)
            h_out = torch.hstack([head, species])

        # IMPORTANT: do NOT squeeze — preserve [N, F, 3] even when F==1 or N==1
        return h_out, v_out
        

    def extra_repr(self) -> str:
        return f'(Oh): tensor({list(self.Oh.shape)}, requires_grad={self.Oh.requires_grad}) \n' \
             + f'(Ov): tensor({list(self.Ov.shape)}, requires_grad={self.Ov.requires_grad})'



class EquivariantTransformer(nn.Module):
    def __init__(self, encoder, processor, decoder):
        super().__init__()
        self.encoder   = encoder
        self.processor = processor
        self.decoder   = decoder
    
    def forward(self, species: Tensor, edge_index: Tensor, edge_attr: Tensor, edge_vec: Tensor) -> Tuple[Tensor, Tensor]:
        h, v, edge_attr = self.encoder(species, edge_index, edge_attr, edge_vec)
        h, v = self.processor(h, v, edge_index, edge_attr, edge_len=torch.linalg.norm(edge_vec, dim=1, keepdim=True))
        return self.decoder(h, v)


from .utils.data_utils import (
    frac_to_cart_coords_with_lattice,
    get_pbc_distances,
    lattice_params_to_matrix_torch,
    radius_graph_pbc,
)


import numpy as np
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_subgraph_mask(edge_index: Tensor, n_frag_switch: Tensor) -> Tensor:
    r"""Filter out edges that have inter-fragment connections.
    Example:
    edge_index: [
        [0, 0, 1, 1, 2, 2],
        [1, 2, 0, 2, 0, 1],
        ]
    n_frag_switch: [0, 0, 1]
    -> [1, 0, 1, 0, 0, 0]

    Args:
        edge_index (Tensor): e_ij
        n_frag_switch (Tensor): fragment that a node belongs to

    Returns:
        Tensor: [n_edge], 1 for inner- and 0 for inter-fragment edge
    """
    in_same_frag = n_frag_switch[edge_index[0]] == n_frag_switch[edge_index[1]]
    return in_same_frag.to(torch.int64)

import torch
from torch import Tensor
import torch.nn as nn

class EquivariantTransformer_dpm(EquivariantTransformer):
    def __init__(self, encoder, processor, decoder, cutoff, latent_dim, embed_dim, num_radial=96, otf_graph = True, design=False, potential_model=False, tps_condition=False, abs_time_emb=False, num_frames=None, num_species=5, pbc=True, object_aware=False):
        super().__init__(encoder, processor, decoder)
        self.cutoff = cutoff
        self.otf_graph = otf_graph
        self.design = design
        self.potential_model = potential_model
        self.tps_condition = tps_condition
        self.pbc = pbc

        '''
        ## hp1:
        node_irreps = [(128, (0, +1))]                 # 32x0e
        msg_irreps  = [(64, (0, +1)), (32, (1, -1))]  # 64x0e + 16x1o
        self.edge_block = EdgeCGBlock(node_irreps, msg_irreps, lmax=2, num_rbf=num_radial, cutoff=self.cutoff)
        self.embed_atom = nn.Sequential(
            nn.Linear(num_species, 128, bias=False),  # learnable per-species weights
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        '''
        ## hp2:
        node_irreps = [(128, (0, +1))]                 # 32x0e
        msg_irreps  = [(64, (0, +1)), (48, (1, -1)), (8, (2, +1))]
        self.edge_block = EdgeCGBlock(node_irreps, msg_irreps, lmax=3, num_rbf=num_radial, cutoff=self.cutoff)
        self.embed_atom = nn.Sequential(
            nn.Linear(num_species, 128, bias=False),  # learnable per-species weights
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        '''
        ## hp2.1:
        node_irreps = [(128, (0, +1))]                 # 32x0e
        msg_irreps  = [(48, (0, +1)), (64, (1, -1)), (8, (2, +1))]
        self.edge_block = EdgeCGBlock(node_irreps, msg_irreps, lmax=3, num_rbf=num_radial, cutoff=self.cutoff)
        self.embed_atom = nn.Sequential(
            nn.Linear(num_species, 128, bias=False),  # learnable per-species weights
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        '''
        '''
        ## hp2.2:
        node_irreps = [(128, (0, +1))]                 # 32x0e
        msg_irreps  = [(32, (0, +1)), (80, (1, -1)), (8, (2, +1))]
        self.edge_block = EdgeCGBlock(node_irreps, msg_irreps, lmax=3, num_rbf=num_radial, cutoff=self.cutoff)
        self.embed_atom = nn.Sequential(
            nn.Linear(num_species, 128, bias=False),  # learnable per-species weights
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        '''
        '''
        ## hp3:
        node_irreps = [(32, (0, +1))]                 # 32x0e
        msg_irreps  = [(64, (0, +1)), (16, (1, -1))]  # 64x0e + 16x1o
        self.edge_block = EdgeCGBlock(node_irreps, msg_irreps, lmax=2, num_rbf=num_radial, cutoff=self.cutoff)
        self.embed_atom = nn.Sequential(
            nn.Linear(num_species, 32, bias=False),  # learnable per-species weights
            nn.SiLU(),
            nn.Linear(32, 32)
        )
        '''

        self.max_num_neighbors_threshold = 50
        self.max_cell_images_per_dim = 5

        cond_dim = latent_dim
        self.cond_dim = cond_dim
        if tps_condition:
            self.cond_to_emb_f = nn.Linear(cond_dim, embed_dim)
            self.cond_to_emb_r = nn.Linear(cond_dim, embed_dim)
            self.mask_to_emb_f = nn.Embedding(cond_dim, embed_dim)
            self.mask_to_emb_r = nn.Embedding(cond_dim, embed_dim)
            if self.pbc:
                self.v_cond_to_emb_f = nn.Linear(cond_dim, embed_dim)
                self.v_cond_to_emb_r = nn.Linear(cond_dim, embed_dim)
            else:
                self.v_cond_to_emb_f = nn.Linear(cond_dim, 3*embed_dim)
                self.v_cond_to_emb_r = nn.Linear(cond_dim, 3*embed_dim)
            self.v_mask_to_emb_f = nn.Embedding(cond_dim, 3*embed_dim)
            self.v_mask_to_emb_r = nn.Embedding(cond_dim, 3*embed_dim)
        else:
            self.cond_to_emb = nn.Linear(cond_dim, embed_dim)
            self.mask_to_emb = nn.Embedding(cond_dim, embed_dim)

        self.abs_time_emb = abs_time_emb
        if abs_time_emb:
            self.register_buffer('time_embed',
                                 nn.Parameter(torch.zeros(1, num_frames, embed_dim), requires_grad=False))
            time_embed = get_1d_sincos_pos_embed_from_grid(self.time_embed.shape[-1], np.arange(num_frames))
            self.time_embed.data.copy_(torch.from_numpy(time_embed).float().unsqueeze(0))
        
        self.num_species = num_species
        self.embed_dim = embed_dim
        self.object_aware = object_aware
    
        if self.pbc and self.tps_condition:
            print("WARNING:: tps_condition not implemented for when cell of the TS is different from the R or P")
            print("WARNING:: tps_condition not implemented for when species of the TS is different from the R or P")


    def _graph_forward(self, species: Tensor, edge_index: Tensor, edge_attr: Tensor, edge_vec: Tensor, t: Tensor, out_cond=None, sub_graph_mask=None) -> Tuple[Tensor, Tensor]:
        h, v, edge_attr = self.encoder(species, edge_index, edge_attr, edge_vec, t, sub_graph_mask)

        if self.abs_time_emb:
            h = h + self.time_embed[:, :, None]
        if self.tps_condition and out_cond is not None:
            if self.pbc:
                with torch.no_grad():
                    '''
                    edge_attr_cond_f = self.scalarize(out_cond['cond_f']['x'], 
                                                      out_cond['cond_f']["edge_index"], 
                                                      out_cond['cond_f']['distance_vec'], 
                                                      out_cond['cond_f']['cell'], 
                                                      out_cond['cond_f']['to_jimages'],
                                                      out_cond['cond_f']['num_bonds'])
                    '''
                    edge_attr_cond_f = self.edge_block(self.embed_atom(out_cond['species'].view(-1, self.num_species)), 
                                                        out_cond['cond_f']["edge_index"], 
                                                        out_cond['cond_f']['distance_vec'])
                    cond_f, v_cond_f, edge_attr_cond_f = self.encoder(
                        out_cond['species'].view(-1,self.num_species), 
                        out_cond['cond_f']['edge_index'], 
                        edge_attr_cond_f, 
                        out_cond['cond_f']['distance_vec'], 
                        torch.ones([*out_cond['species'].shape[:-1],1], device=out_cond['species'].device).reshape(-1,1), 
                        out_cond['cond_f']['sub_graph_mask']
                        )
                    '''
                    edge_attr_cond_r = self.scalarize(out_cond['cond_r']['x'], 
                                                      out_cond['cond_r']["edge_index"], 
                                                      out_cond['cond_r']['distance_vec'], 
                                                      out_cond['cond_r']['cell'], 
                                                      out_cond['cond_r']['to_jimages'],
                                                      out_cond['cond_r']['num_bonds'])
                    '''
                    edge_attr_cond_r = self.edge_block(self.embed_atom(out_cond['species'].view(-1, self.num_species)), 
                                                        out_cond['cond_r']["edge_index"], 
                                                        out_cond['cond_r']['distance_vec'])
                    cond_r, v_cond_r, edge_attr_cond_r = self.encoder(
                        out_cond['species'].view(-1,self.num_species), 
                        out_cond['cond_r']['edge_index'], 
                        edge_attr_cond_r, 
                        out_cond['cond_r']['distance_vec'], 
                        torch.ones([*out_cond['species'].shape[:-1],1], device=out_cond['species'].device).reshape(-1,1),
                        out_cond['cond_r']['sub_graph_mask']
                        )
                    cond_f_mask = out_cond["cond_f"]['mask']
                    cond_r_mask = out_cond["cond_r"]['mask']
                    
                    
                h = h + self.cond_to_emb_f(cond_f) + self.mask_to_emb_f(cond_f_mask)
                h = h + self.cond_to_emb_r(cond_r) + self.mask_to_emb_r(cond_r_mask)
                v_embc_f = self.v_cond_to_emb_f(v_cond_f.transpose(1,2)).permute(0,2,1)
                v = v + v_embc_f.reshape(-1,self.embed_dim,3) +  self.v_mask_to_emb_f(cond_f_mask).reshape(-1,self.embed_dim,3)
                v_embc_r = self.v_cond_to_emb_r(v_cond_r.transpose(1,2)).permute(0,2,1)
                v = v + v_embc_r.reshape(-1,self.embed_dim,3) +  self.v_mask_to_emb_r(cond_r_mask).reshape(-1,self.embed_dim,3)


        def _filter_edges(_edge_index, _edge_attr, _edge_vec, _mask: torch.Tensor):
                # mask: [E] in {0,1}
                keep = (_mask > 0)
                idx = keep.nonzero(as_tuple=True)[0]
                return _edge_index[:, idx], _edge_attr[idx], _edge_vec[idx]

        # ... inside your forward (right before calling the processor) ...
        # sub_graph_mask is what you already computed (1=intra, 0=inter)
        if self.object_aware:
            edge_index, edge_attr, edge_vec = _filter_edges(edge_index, edge_attr, edge_vec, sub_graph_mask)

        h, v = self.processor(h, v, edge_index, edge_attr, edge_len=torch.linalg.norm(edge_vec, dim=1, keepdim=True))
        # print('processor--> h=', )
        # for i in range(h.shape[0]):
        #     print(h[i,:10])
        # print('processor--> v=', )
        # for i in range(v.shape[0]):
        #     print(v[i,:10])
        return self.decoder(h, v)


    
    def inference(self, x: Tensor, t: Tensor, 
                cell=None, 
                num_atoms=None,
                conditions=None, 
                aatype=None, fragments_idx = None):
        B, T, N, _ = x.shape
        assert t.shape == (B,)

        if self.otf_graph:
            edge_index, to_jimages, num_bonds = radius_graph_pbc(
                cart_coords=x.view(-1, 3),
                lattice=cell.view(-1, 3, 3),
                num_atoms=num_atoms.view(-1),                                      # The num_atoms is used to separate batched structures before connecting graph
                radius=self.cutoff,
                max_num_neighbors_threshold=self.max_num_neighbors_threshold,
                max_cell_images_per_dim=self.max_cell_images_per_dim,
            )
            # remove inter-object edges here from edge_index
            if self.object_aware:
                assert fragments_idx is not None
                sub_graph_mask = get_subgraph_mask(edge_index, fragments_idx.reshape(-1))

            if conditions is not None and self.pbc:
                if self.tps_condition:
                    edge_index_cond_f, to_jimages_cond_f, num_bonds_cond_f = radius_graph_pbc(
                        cart_coords=conditions["cond_f"]['x'].view(-1, 3),
                        lattice=cell.view(-1, 3, 3),
                        num_atoms=num_atoms.view(-1),
                        radius=self.cutoff,
                        max_num_neighbors_threshold=self.max_num_neighbors_threshold,
                        max_cell_images_per_dim=self.max_cell_images_per_dim,
                    )
                    edge_index_cond_r, to_jimages_cond_r, num_bonds_cond_r = radius_graph_pbc(
                        cart_coords=conditions["cond_r"]['x'].view(-1, 3),
                        lattice=cell.view(-1, 3, 3),
                        num_atoms=num_atoms.view(-1),
                        radius=self.cutoff,
                        max_num_neighbors_threshold=self.max_num_neighbors_threshold,
                        max_cell_images_per_dim=self.max_cell_images_per_dim,
                    )

                    # remove inter-object edges here from edge_index
                    if self.object_aware:
                        assert conditions["cond_f"]['fragments_idx'] is not None
                        sub_graph_mask_f = get_subgraph_mask(edge_index_cond_f, conditions["cond_f"]['fragments_idx'].reshape(-1))

                        assert conditions["cond_r"]['fragments_idx'] is not None
                        sub_graph_mask_r = get_subgraph_mask(edge_index_cond_r, conditions["cond_r"]['fragments_idx'].reshape(-1))


            # self.otf_graph = False

        out = get_pbc_distances(
            x.view(-1, 3),
            edge_index,
            cell.view(-1, 3, 3),
            to_jimages,
            num_atoms.view(-1),
            num_bonds,
            coord_is_cart=True,
            return_offsets=True,
            return_distance_vec=True,
        )

        if conditions is not None:
            if self.pbc:
                if self.tps_condition:
                    out_cond = {}
                    out_cond['cond_f'] = get_pbc_distances(
                        conditions["cond_f"]["x"].view(-1, 3),
                        edge_index_cond_f,
                        cell.view(-1, 3, 3),
                        to_jimages_cond_f,
                        num_atoms.view(-1),
                        num_bonds_cond_f,
                        coord_is_cart=True,
                        return_offsets=True,
                        return_distance_vec=True,
                    )
                    if self.object_aware:
                        out_cond['cond_f']['sub_graph_mask'] = sub_graph_mask_f
                    else:
                        out_cond['cond_f']['sub_graph_mask'] = None
                    out_cond['cond_f']['x'] = conditions["cond_f"]["x"].view(-1, 3)
                    out_cond['cond_f']['cell'] = cell.view(-1,3,3)
                    out_cond['cond_f']['to_jimages'] = to_jimages_cond_f
                    out_cond['cond_f']['num_bonds'] = num_bonds_cond_f

                    out_cond['cond_r'] = get_pbc_distances(
                        conditions["cond_r"]["x"].view(-1, 3),
                        edge_index_cond_r,
                        cell.view(-1, 3, 3),
                        to_jimages_cond_r,
                        num_atoms.view(-1),
                        num_bonds_cond_r,
                        coord_is_cart=True,
                        return_offsets=True,
                        return_distance_vec=True,
                    )
                    if self.object_aware:
                        out_cond['cond_r']['sub_graph_mask'] = sub_graph_mask_r
                    else:
                        out_cond['cond_r']['sub_graph_mask'] = None
                    out_cond['cond_r']['x'] = conditions["cond_r"]["x"].view(-1, 3)
                    out_cond['cond_r']['cell'] = cell.view(-1,3,3)
                    out_cond['cond_r']['to_jimages'] = to_jimages_cond_r
                    out_cond['cond_r']['num_bonds'] = num_bonds_cond_r

                    out_cond["species"] = aatype
                    out_cond['cond_f']["mask"] = conditions['cond_f']["mask"]
                    out_cond['cond_r']["mask"] = conditions['cond_r']["mask"]
 
            else:
                out_cond = conditions
        else:
            out_cond=None
        edge_index = out["edge_index"]
        edge_len = out["distances"]
        edge_vec = out["distance_vec"]

        if aatype is not None:
            species = aatype
        else:
            aatype = torch.zeros([B,T,N], dtype=torch.long, device=x.device)
            species = torch.nn.functional.one_hot(aatype, num_classes=self.num_species).to(torch.float)

        # edge_attr = self.scalarize(x.view(-1,3), edge_index, edge_vec, cell.view(-1,3,3), to_jimages, num_bonds)
        edge_attr = self.edge_block(self.embed_atom(species.view(-1, self.num_species)), edge_index, edge_vec)
        edge_attr = msg_to_invariants_from_irreps(edge_attr, self.edge_block.msg_irreps)

        t = t.unsqueeze(-1).unsqueeze(1).expand(-1,T,-1).unsqueeze(2).expand(-1,-1,N,-1)
        if self.object_aware:
            scaler_out, vector_out = self._graph_forward(species.reshape(-1,self.num_species), edge_index, edge_attr, edge_vec, t.reshape(-1,1), out_cond, sub_graph_mask=sub_graph_mask)
        else:
            scaler_out, vector_out = self._graph_forward(species.reshape(-1,self.num_species), edge_index, edge_attr, edge_vec, t.reshape(-1,1), out_cond)
        if self.design:
            # return torch.hstack([vector_out, scaler_out]).view(B, T, N, -1)
            return scaler_out.view(B, T, N, -1)
        elif self.potential_model:
            return scaler_out.reshape(B, T, N, -1)
        else:
            return vector_out.reshape(B, T, N, -1)
        
    def forward(self, x: Tensor, t: Tensor, 
                cell=None, 
                num_atoms=None,
                conditions=None,
                aatype=None, x_latt=None, x1=None, v_mask=None, fragments_idx = None):
        if self.design:
            x_ = x_latt
            aatype_ = x
            if v_mask is not None: x_ = x_*v_mask+x1*(1-v_mask)
            scaler_out = self.inference(x_, t, cell, num_atoms, conditions, aatype_, fragments_idx=fragments_idx)
            return scaler_out*v_mask
        elif self.potential_model:
            if v_mask is not None:
                x = x*v_mask+x1*(1-v_mask)
            scaler_out = self.inference(x, t, cell, num_atoms, aatype=aatype, fragments_idx=fragments_idx)
            assert (torch.where(v_mask.ravel() == 0)[0]).size(0) + (torch.where((1-v_mask).ravel() == 0)[0]).size(0) == (v_mask.ravel()).size(0)
            return scaler_out
        else:
            if v_mask is not None: x = x*v_mask+x1*(1-v_mask)
            vector_out = self.inference(x, t, cell, num_atoms, conditions, aatype, fragments_idx=fragments_idx)
            return vector_out*v_mask

    def forward_inference(self, x: Tensor, t: Tensor, 
                cell=None, 
                num_atoms=None,
                conditions=None,
                aatype=None, x_latt=None, x1=None, v_mask=None, fragments_idx = None):
        if self.design:
            x_ = x_latt
            aatype_ = x
            if v_mask is not None:
                x_ = x_*v_mask+x1*(1-v_mask)
            scaler_out = self.inference(x_, t, cell, num_atoms, conditions, aatype_, fragments_idx=fragments_idx)
            return scaler_out*v_mask
        elif self.potential_model:
            if v_mask is not None:
                x = x*v_mask+x1*(1-v_mask)
            scaler_out = self.inference(x, t, cell, num_atoms, aatype=aatype)
            return scaler_out
        else:
            x = x*v_mask+x1*(1-v_mask)
            vector_out = self.inference(x, t, cell, num_atoms, conditions, aatype, fragments_idx=fragments_idx)
            return vector_out*v_mask
    
    
    def _get_processed_var(self, x: Tensor, t: Tensor, 
                cell=None, 
                num_atoms=None,
                conditions=None, 
                aatype=None):
        assert cell is not None
        B, T, N, _ = x.shape
        assert t.shape == (B,)
        if self.otf_graph:
            self.edge_index, self.to_jimages, self.num_bonds = radius_graph_pbc(
                cart_coords=x.view(-1, 3),
                lattice=cell.view(-1, 3, 3),
                num_atoms=num_atoms.view(-1),
                radius=self.cutoff,
                max_num_neighbors_threshold=self.max_num_neighbors_threshold,
                max_cell_images_per_dim=self.max_cell_images_per_dim,
            )
            # if conditions is not None:
            #     self.edge_index_cond, self.to_jimages_cond, self.num_bonds_cond = radius_graph_pbc(
            #         cart_coords=conditions["x"].view(-1, 3),
            #         lattice=conditions["cell"].view(-1, 3, 3),
            #         num_atoms=conditions["num_atoms"].view(-1),
            #         radius=self.cutoff,
            #         max_num_neighbors_threshold=self.max_num_neighbors_threshold,
            #         max_cell_images_per_dim=self.max_cell_images_per_dim,
            #     )
            # self.otf_graph = False

        out = get_pbc_distances(
            x.view(-1, 3),
            self.edge_index,
            cell.view(-1, 3, 3),
            self.to_jimages,
            num_atoms.view(-1),
            self.num_bonds,
            coord_is_cart=True,
            return_offsets=True,
            return_distance_vec=True,
        )
        # if conditions is not None:
        #     out_cond = get_pbc_distances(
        #         conditions["x"].view(-1, 3),
        #         self.edge_index_cond,
        #         conditions["cell"].view(-1, 3, 3),
        #         self.to_jimages_cond,
        #         conditions["num_atoms"].view(-1),
        #         self.num_bonds_cond,
        #         coord_is_cart=True,
        #         return_offsets=True,
        #         return_distance_vec=True,
        #     )
        #     out_cond["species"] = conditions["species"]
        #     out_cond["mask"] = conditions["mask"]
        edge_index = out["edge_index"]
        edge_len = out["distances"]
        edge_vec = out["distance_vec"]
        edge_attr = torch.hstack([edge_vec, edge_len.view(-1, 1)])

        t = t.unsqueeze(-1).unsqueeze(1).expand(-1,T,-1).unsqueeze(2).expand(-1,-1,N,-1)
        if aatype is not None:
            species = aatype
        else:
            aatype = torch.zeros([B,T,N], dtype=torch.long, device=x.device)
            species = torch.nn.functional.one_hot(aatype, num_classes=self.num_species, dtype=torch.float)
            
        h, v, edge_attr = self.encoder(species.reshape(-1,self.num_species), edge_index, edge_attr, edge_vec, t.reshape(-1,1))
        h, v = self.processor(h, v, edge_index, edge_attr, edge_len=torch.linalg.norm(edge_vec, dim=1, keepdim=True))
        return h, v
    
    def _get_encoded_var(self, x: Tensor, t: Tensor, 
                cell=None, 
                num_atoms=None,
                conditions=None, 
                aatype=None):
        assert cell is not None
        B, T, N, _ = x.shape
        assert t.shape == (B,)
        if self.otf_graph:
            self.edge_index, self.to_jimages, self.num_bonds = radius_graph_pbc(
                cart_coords=x.view(-1, 3),
                lattice=cell.view(-1, 3, 3),
                num_atoms=num_atoms.view(-1),
                radius=self.cutoff,
                max_num_neighbors_threshold=self.max_num_neighbors_threshold,
                max_cell_images_per_dim=self.max_cell_images_per_dim,
            )
            # if conditions is not None:
            #     self.edge_index_cond, self.to_jimages_cond, self.num_bonds_cond = radius_graph_pbc(
            #         cart_coords=conditions["x"].view(-1, 3),
            #         lattice=conditions["cell"].view(-1, 3, 3),
            #         num_atoms=conditions["num_atoms"].view(-1),
            #         radius=self.cutoff,
            #         max_num_neighbors_threshold=self.max_num_neighbors_threshold,
            #         max_cell_images_per_dim=self.max_cell_images_per_dim,
            #     )
            # self.otf_graph = False

        out = get_pbc_distances(
            x.view(-1, 3),
            self.edge_index,
            cell.view(-1, 3, 3),
            self.to_jimages,
            num_atoms.view(-1),
            self.num_bonds,
            coord_is_cart=True,
            return_offsets=True,
            return_distance_vec=True,
        )
        # if conditions is not None:
        #     out_cond = get_pbc_distances(
        #         conditions["x"].view(-1, 3),
        #         self.edge_index_cond,
        #         conditions["cell"].view(-1, 3, 3),
        #         self.to_jimages_cond,
        #         conditions["num_atoms"].view(-1),
        #         self.num_bonds_cond,
        #         coord_is_cart=True,
        #         return_offsets=True,
        #         return_distance_vec=True,
        #     )
        #     out_cond["species"] = conditions["species"]
        #     out_cond["mask"] = conditions["mask"]
        edge_index = out["edge_index"]
        edge_len = out["distances"]
        edge_vec = out["distance_vec"]
        edge_attr = torch.hstack([edge_vec, edge_len.view(-1, 1)])

        t = t.unsqueeze(-1).unsqueeze(1).expand(-1,T,-1).unsqueeze(2).expand(-1,-1,N,-1)
        if aatype is not None:
            species = aatype
        else:
            aatype = torch.zeros([B,T,N], dtype=torch.long, device=x.device)
            species = torch.nn.functional.one_hot(aatype, num_classes=self.num_species, dtype=torch.float)
            
        h, v, edge_attr = self.encoder(species.reshape(-1,self.num_species), edge_index, edge_attr, edge_vec, t.reshape(-1,1))
        
        return h, v, edge_attr

    def forward_encoder(self, x: Tensor, t: Tensor, 
                cell=None, 
                num_atoms=None,
                conditions=None,
                aatype=None, x_latt=None):
        if self.design:
            x_ = x_latt
            aatype_ = x
            scaler_out = self._get_encoded_var(x_, t, cell, num_atoms, conditions, aatype_)
            return scaler_out
        else:
            vector_out = self._get_encoded_var(x, t, cell, num_atoms, conditions, aatype)
            return vector_out
        

    def forward_processor(self, x: Tensor, t: Tensor, 
                cell=None, 
                num_atoms=None,
                conditions=None,
                aatype=None, x_latt=None):
        if self.design:
            x_ = x_latt
            aatype_ = x
            scaler_out = self._get_processed_var(x_, t, cell, num_atoms, conditions, aatype_)
            return scaler_out
        else:
            vector_out = self._get_processed_var(x, t, cell, num_atoms, conditions, aatype)
            return vector_out

    def rearrange_batch(self, idx, model_kwargs:dict):
        B = idx.shape[0]
        assert B == model_kwargs['x1'].shape[0]
        for k in model_kwargs.keys():
            if 'conditions' not in k:
                model_kwargs[k] = model_kwargs[k][idx]
            else:
                '''
                if self.tps_condition:
                    model_kwargs['conditions']['cond_f']['x'] = model_kwargs['conditions']['cond_f']['x'].reshape(B,-1,3)[idx].reshape(-1,3)
                    model_kwargs['conditions']['cond_f']['mask'] = model_kwargs['conditions']['cond_f']['mask'].reshape(B,-1)[idx].reshape(-1)
                    model_kwargs['conditions']['cond_r']['x'] = model_kwargs['conditions']['cond_r']['x'].reshape(B,-1,3)[idx].reshape(-1,3)
                    model_kwargs['conditions']['cond_r']['mask'] = model_kwargs['conditions']['cond_r']['mask'].reshape(B,-1)[idx].reshape(-1)
                else:
                '''
                if model_kwargs[k] is not None:
                    raise Exception("Shouldn't input condition")

class TransformerDecoder(nn.Module):
    def __init__(self, dim: int, num_scalar_out: int, num_vector_out: int, 
                 num_species: int=5,
                 nhead: int=4, 
                 dim_feedforward: int=1024,
                 activation: str='gelu',
                 dropout: float=0.0,
                 norm_first: bool = True,
                 bias: bool = True,
                 num_layers: int = 6,
                 ) -> None:
        super().__init__()
        self.num_species = num_species
        self.dim = dim
        self.num_scalar_out = num_scalar_out
        self.num_vector_out = num_vector_out

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                activation=activation,
                dropout=dropout,
                batch_first=True,
                norm_first=norm_first,
                bias=bias,
            ),
            norm=nn.LayerNorm(dim),
            num_layers=num_layers,
        )

        self.Oh = nn.Parameter(torch.randn(dim, num_scalar_out))
        self.Ov = nn.Parameter(torch.randn(dim, num_vector_out))
        self.Ov_frac = nn.Parameter(torch.randn(dim, num_vector_out))
        self.Ol = nn.Parameter(torch.randn(dim, num_vector_out*3+1, 6))

    def forward(self, h:Tensor, v: Tensor) -> Tensor:
        B,T,N,_ = h.shape
        h = h.reshape(B*T*N,-1)
        v = v.reshape(B*T*N,-1,3)
        h = h.unsqueeze(-1)
        x = torch.concatenate([h, v], dim=-1).transpose(1, 2)
        x = self.transformer.forward(x)
        x = x.transpose(1, 2)
        h = x[..., 0]
        v = x[..., 1:]
        h_ = h @ self.Oh
        v_out = torch.einsum('ndi, df -> nfi', v, self.Ov)
        v_frac_out = torch.einsum('ndi, df -> nfi', v, self.Ov_frac)
        x = x.reshape(B*T,N,-1,4)
        l_out = torch.einsum('ndi, dif -> nf', x.mean(1), self.Ol)
        assert h_.shape[-1] == self.num_species
        h_out = torch.nn.functional.softmax(h_[...,-self.num_species:], dim=-1)
        return {
            "aatype": h_out.reshape(B, T, N, -1), 
            "pos": v_out.squeeze().reshape(B, T, N, -1), 
            "frac_pos": v_frac_out.squeeze().reshape(B, T, N, -1), 
            "cell": l_out.squeeze().reshape(B, T, -1)
            }

    def extra_repr(self) -> str:
        return f'(Oh): tensor({list(self.Oh.shape)}, requires_grad={self.Oh.requires_grad}) \n' \
             + f'(Ov): tensor({list(self.Ov.shape)}, requires_grad={self.Ov.requires_grad})'



class TransformerProcessor(nn.Module):
    def __init__(self, dim: int, num_scalar_out: int, num_vector_out: int,
                 nhead: int=4, 
                 dim_feedforward: int=1024,
                 activation: str='gelu',
                 dropout: float=0.0,
                 norm_first: bool = True,
                 bias: bool = True,
                 num_layers: int = 6,
                 node_dim: int = 4,
                 edge_dim: int = 64,
                 input_dim: int = 1,
                 ) -> None:
        super().__init__()
        self.dim = dim
        self.num_scalar_out = num_scalar_out
        self.num_vector_out = num_vector_out

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                activation=activation,
                dropout=dropout,
                batch_first=True,
                norm_first=norm_first,
                bias=bias,
            ),
            norm=nn.LayerNorm(dim),
            num_layers=num_layers,
        )

        self.embed_time = nn.Sequential(
            GaussianRandomFourierFeatures(node_dim, input_dim=input_dim),
            MLP([node_dim, edge_dim, node_dim], act=nn.SiLU()),
        )

    def inference(self, x:Tensor) -> Tensor:
        x = x.transpose(1, 2)
        x = self.transformer.forward(x)
        x = x.transpose(1, 2)
        # h = x[..., 0]
        # v = x[..., 1:]
        return x
    
    def forward(self, x: Tensor, t: Tensor, x1=None, v_mask=None) -> Tensor:
        B,T,N,D,_ = x.shape
        # h = x[..., 0] 
        # v = x[..., 1:]
        x = x*v_mask+x1*(1-v_mask)
        x = x + self.embed_time(t)[None,None,None,:,None]
        x_out = self.inference(x.reshape(B*T*N,D,4))
        return x_out.reshape(B,T,N,D,4)

    
    def forward_inference(self, x: Tensor, t: Tensor, x1=None, v_mask=None, conditions=None) -> Tensor:
        B,T,N,D,_ = x.shape
        # h = x[..., 0] 
        # v = x[..., 1:]
        x = x*v_mask+x1*(1-v_mask)
        x = x + self.embed_time(t)[None,None,None,:,None]
        x_out = self.inference(x.reshape(B*T*N,D,4))
        return x_out.reshape(B,T,N,D,4)
