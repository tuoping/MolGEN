import torch, os
from mdgen.dataset import EquivariantTransformerDataset_Transition1x
torch.set_default_dtype(torch.float64)

trainset = EquivariantTransformerDataset_Transition1x("tests/test_data/Transition1x/", 12, sim_condition=False, tps_condition=True, stage="example-multi-fragments")
# trainset = torch.load(os.path.join("data/Transition1x", "tps_masked_train-fragmented_cutoffx1.5.pt"), weights_only=False)

batch_size = 1
train_loader = torch.utils.data.DataLoader(
    trainset,
    batch_size=batch_size,
    num_workers=0,
    shuffle=True,
)
sample_batch = next(iter(train_loader))


from mdgen.parsing import parse_train_args
args = parse_train_args()

args.num_frames=3

args.ckpt_freq = 10 
args.val_repeat = 0.2
args.epochs = 1
args.num_species = 5

args.edge_dim = 8
args.num_convs = 5
args.num_heads = 4
args.ff_dim = 16

args.cutoff= 12
args.localmask = False
args.path_type = "Linear"
args.batch_size = batch_size

args.tps_condition = True
args.sim_condition = False
args.ratio_conditonal = 1.
# args.prediction='score'
# args.sampling_method = "Euler"

args.design = False
args.potential_model = False
args.pbc = False 
args.object_aware = True

os.environ["MODEL_DIR"] = os.path.join("tests_cache", args.run_name)

from mdgen.equivariant_wrapper import EquivariantMDGenWrapper
model = EquivariantMDGenWrapper(args).double()
model.eval()

model.iter_step += 1
model.stage = "train"
prep = model.prep_batch(sample_batch)

_, x0, x1 = model.transport.sample(prep['latents'])

from mdgen.model.utils.data_utils import (
    get_pbc_distances,
    radius_graph_pbc,
)

from pymatgen.core.operations import SymmOp
import numpy as np
def rand_rot():
    axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)
    angle = np.random.uniform(0, 360)
    op = SymmOp.from_origin_axis_angle(
                    (0, 0, 0),
                    axis=tuple(axis),
                    angle=angle,
                    angle_in_radians=False
                )
    m = op.rotation_matrix
    return m

x = x1.double()
B,T,L,_ = x.shape
species = prep['species'].double()
cell = prep['model_kwargs']['cell'].double()
num_atoms = prep['model_kwargs']['num_atoms']
i_frag = torch.unique(prep['model_kwargs']['fragments_idx'][:,1,:], sorted=True, return_inverse=False, return_counts=False, dim=None)

from mdgen.model.nn.basis import EdgeCGBlock, GaussianRandomFourierFeatures

# node_irreps = [(128, (0, +1))]                 # 32x0e
# msg_irreps  = [(64, (0, +1)), (48, (1, -1)), (8, (2, +1))]
# edge_block = EdgeCGBlock(node_irreps, msg_irreps, lmax=3, num_rbf=96, cutoff=model.model.cutoff).double()
# import torch.nn as nn
# # 1) Map one-hot -> scalar node state (0e irreps)
# num_species = species.shape[-1]
# embed_atom = nn.Sequential(
#     nn.Linear(num_species, 128, bias=False),  # learnable per-species weights
#     nn.SiLU(),
#     nn.Linear(128, 128)
# ).double()

embed_atom = model.model.embed_atom
edge_block = model.model.edge_block

from mdgen.model.nn.mlp import MLP
t = torch.ones(B).double()

from mdgen.model.equivariant_latent_model import get_subgraph_mask

edge_index, to_jimages, num_bonds = radius_graph_pbc(
    cart_coords=x.view(-1, 3),
    lattice=cell.view(-1, 3, 3),
    num_atoms=num_atoms.view(-1),                                      # The num_atoms is used to separate batched structures before connecting graph
    radius=model.model.cutoff,
    max_num_neighbors_threshold=model.model.max_num_neighbors_threshold,
    max_cell_images_per_dim=model.model.max_cell_images_per_dim,
)
sub_graph_mask = get_subgraph_mask(edge_index, prep['model_kwargs']['fragments_idx'].reshape(-1))
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

edge_vec = out["distance_vec"]

f0 = embed_atom(species.view(-1, model.model.num_species))    
edge_attr = edge_block(f0, edge_index, edge_vec)  
from mdgen.model.equivariant_latent_model import msg_to_invariants_from_irreps
edge_attr = msg_to_invariants_from_irreps(edge_attr, edge_block.msg_irreps)
H, V, edge_attr = model.model.encoder(species.view(-1, model.model.num_species), edge_index, edge_attr, edge_vec, t, sub_graph_mask)


def _filter_edges(_edge_index, _edge_attr, _edge_vec, _mask: torch.Tensor):
            # mask: [E] in {0,1}
            keep = (_mask > 0)
            idx = keep.nonzero(as_tuple=True)[0]
            return _edge_index[:, idx], _edge_attr[idx], _edge_vec[idx]

with torch.autocast(device_type='cpu', dtype=torch.float64, enabled=True):
    edge_index, edge_attr, edge_vec = _filter_edges(edge_index, edge_attr, edge_vec, sub_graph_mask)
    H_processed, V_processed = model.model.processor(H, V, edge_index, edge_attr, edge_len=torch.linalg.norm(edge_vec, dim=1, keepdim=True))
    H_decoded, V_decoded = model.model.decoder(H_processed, V_processed)
    del edge_index
    del edge_attr
    del edge_vec
assert torch.all(out['distances'] > 0)

R = torch.from_numpy(rand_rot()).double()
# print(R)


x_r = x @ R.T * ((prep['model_kwargs']['fragments_idx'] == 0).unsqueeze(-1)) + x * ((prep['model_kwargs']['fragments_idx'] != 0).unsqueeze(-1))

edge_index_r, to_jimages_r, num_bonds_r = radius_graph_pbc(
    cart_coords=x_r.view(-1, 3),
    lattice=cell.view(-1, 3, 3),
    num_atoms=num_atoms.view(-1),                                      # The num_atoms is used to separate batched structures before connecting graph
    radius=model.model.cutoff,
    max_num_neighbors_threshold=model.model.max_num_neighbors_threshold,
    max_cell_images_per_dim=model.model.max_cell_images_per_dim,
)
sub_graph_mask_r = get_subgraph_mask(edge_index_r, prep['model_kwargs']['fragments_idx'].reshape(-1))
out_r = get_pbc_distances(
    x_r.view(-1, 3),
    edge_index_r,
    cell.view(-1, 3, 3),
    to_jimages_r,
    num_atoms.view(-1),
    num_bonds_r,
    coord_is_cart=True,
    return_offsets=True,
    return_distance_vec=True,
)
edge_vec_r = out_r['distance_vec']

edge_attr_r = edge_block(f0, edge_index_r, edge_vec_r)  
from mdgen.model.equivariant_latent_model import msg_to_invariants_from_irreps
edge_attr_r = msg_to_invariants_from_irreps(edge_attr_r, edge_block.msg_irreps)
H_r, V_r, edge_attr_r = model.model.encoder(species.view(-1, model.model.num_species), edge_index_r, edge_attr_r, edge_vec_r, t, sub_graph_mask_r)

# print("h err =", ((H-H_r).abs()/H.abs()).max())
assert torch.allclose(H, H_r, rtol=1e-3, atol=5e-5)

fragments_idx_V = prep['model_kwargs']['fragments_idx'].reshape(-1).unsqueeze(-1).unsqueeze(-1)
V_benchmark = V @ R.T * (fragments_idx_V == 0) + V * (fragments_idx_V != 0)

def vec_err(A, B, eps=1e-12):
    # A,B: [N,C,3]
    num = torch.linalg.norm(A - B, dim=-1)               # [N,C]
    den = torch.linalg.norm(A, dim=-1) + torch.linalg.norm(B, dim=-1) + eps
    return (num / den).max()   # scalar

# print("raw V err =", vec_err(V_r, V_benchmark).item())
assert torch.allclose(V_r, V_benchmark, rtol=1e-3, atol=5e-5)


with torch.autocast(device_type='cpu', dtype=torch.float64, enabled=True):
    edge_index_r, edge_attr_r, edge_vec_r = _filter_edges(edge_index_r, edge_attr_r, edge_vec_r, sub_graph_mask_r)
    H_processed_r, V_processed_r = model.model.processor(H_r, V_r, edge_index_r, edge_attr_r, edge_len=torch.linalg.norm(edge_vec_r, dim=1, keepdim=True))
    H_decoded_r, V_decoded_r = model.model.decoder(H_processed_r, V_processed_r)
    del edge_index_r
    del edge_attr_r
    del edge_vec_r


Vp_benchmark = V_processed @ R.T * (fragments_idx_V == 0) + V_processed * (fragments_idx_V != 0)
# print("Processed V err =", vec_err(V_processed_r, Vp_benchmark).item())
assert torch.allclose(V_processed_r, Vp_benchmark, rtol=1e-3, atol=5e-5)

from mdgen.model.equivariant_latent_model import msg_to_invariants_from_irreps
Vd_benchmark = V_decoded @ R.T * (fragments_idx_V == 0) + V_decoded * (fragments_idx_V != 0)
# print("Decoded V err =", vec_err(V_decoded_r, Vd_benchmark).item())
assert torch.allclose(V_decoded_r, Vd_benchmark, rtol=1e-3, atol=5e-5)


out_v = model.model.forward(x, t, **prep['model_kwargs']).view(-1,3)
assert torch.allclose(out_v, V_decoded.view(-1,3)*prep['model_kwargs']['v_mask'].view(-1,3))
out_v_r = model.model.forward(x_r, t, **prep['model_kwargs']).view(-1,3)
assert torch.allclose(out_v_r, V_decoded_r.view(-1,3)*prep['model_kwargs']['v_mask'].view(-1,3))
out_v_benchmark = out_v @ R.T * (fragments_idx_V.view(-1,1) == 0) + out_v * (fragments_idx_V.view(-1,1) != 0)
print("Out V err =", vec_err(out_v_r, out_v_benchmark).item())
assert torch.allclose(out_v_r, out_v_benchmark, rtol=1e-3, atol=5e-5)