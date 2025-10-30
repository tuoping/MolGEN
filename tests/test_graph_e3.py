import torch, os
from mdgen.dataset import EquivariantTransformerDataset_Transition1x

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
args.ratio_conditonal = 1.
args.sim_condition = False
# args.prediction='score'
# args.sampling_method = "Euler"

args.design = False
args.potential_model = False
args.pbc = True 

os.environ["MODEL_DIR"] = os.path.join("tests_cache", args.run_name)

from mdgen.equivariant_wrapper import EquivariantMDGenWrapper
model = EquivariantMDGenWrapper(args).double()
model.eval()

model.iter_step += 1
model.stage = "train"
prep = model.prep_batch(sample_batch)

_, x0, x1 = model.transport.sample(prep['latents'])
x = x1.double()
B,T,L,_ = x.shape
t = torch.ones(B).double()

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

species = prep['species'].double()
cell = prep['model_kwargs']['cell'].double()
num_atoms = prep['model_kwargs']['num_atoms']


def so3_random(device):
    # Draw a random rotation with det=+1
    Q, _ = torch.linalg.qr(torch.randn(3,3, device=device).double())
    if torch.linalg.det(Q) < 0: Q[:,0] = -Q[:,0]
    return Q

def vec_err(A, B, eps=1e-12):
    # A,B: [N,C,3]
    num = torch.linalg.norm(A - B, dim=-1)               # [N,C]
    den = torch.linalg.norm(A, dim=-1) + torch.linalg.norm(B, dim=-1) + eps
    return (num / den).max()   # scalar

R = so3_random(x.device)
print(R)

x_r = x @ R.T  + torch.randn(3)[None,None,None,:]


from mdgen.model.nn.basis import EdgeCGBlock, GaussianRandomFourierFeatures
embed_atom = model.model.embed_atom
edge_block = model.model.edge_block

from mdgen.model.nn.mlp import MLP


edge_index, to_jimages, num_bonds = radius_graph_pbc(
    cart_coords=x.view(-1, 3),
    lattice=cell.view(-1, 3, 3),
    num_atoms=num_atoms.view(-1),                                      # The num_atoms is used to separate batched structures before connecting graph
    radius=model.model.cutoff,
    max_num_neighbors_threshold=model.model.max_num_neighbors_threshold,
    max_cell_images_per_dim=model.model.max_cell_images_per_dim,
)

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
'''
H, V, edge_attr = model.model.encoder(species.view(-1, model.model.num_species), edge_index, edge_attr, edge_vec, t)
def _filter_edges(_edge_index, _edge_attr, _edge_vec, _mask: torch.Tensor):
            # mask: [E] in {0,1}
            keep = (_mask > 0)
            idx = keep.nonzero(as_tuple=True)[0]
            return _edge_index[:, idx], _edge_attr[idx], _edge_vec[idx]

with torch.autocast(device_type='cpu', dtype=torch.float64, enabled=True):
    H_processed, V_processed = model.model.processor(H, V, edge_index, edge_attr, edge_len=torch.linalg.norm(edge_vec, dim=1, keepdim=True))
    H_decoded, V_decoded = model.model.decoder(H_processed, V_processed)
    # del edge_index
    # del edge_attr
    # del edge_vec
assert torch.all(out['distances'] > 0)
'''

conditions = prep['model_kwargs']['conditions']
edge_index_cond_f, to_jimages_cond_f, num_bonds_cond_f = radius_graph_pbc(
    cart_coords=conditions["cond_f"]['x'].view(-1, 3),
    lattice=cell.view(-1, 3, 3),
    num_atoms=num_atoms.view(-1),
    radius=args.cutoff,
    max_num_neighbors_threshold=model.model.max_num_neighbors_threshold,
    max_cell_images_per_dim=model.model.max_cell_images_per_dim,
)
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
out_cond['cond_f']['sub_graph_mask'] = None
out_cond['cond_f']['x'] = conditions["cond_f"]["x"].view(-1, 3)
out_cond['cond_f']['cell'] = cell.view(-1,3,3)
out_cond['cond_f']['to_jimages'] = to_jimages_cond_f
out_cond['cond_f']['num_bonds'] = num_bonds_cond_f
out_cond["species"] = species

edge_attr_cond_f = model.model.edge_block(model.model.embed_atom(out_cond['species'].view(-1, model.model.num_species)), 
                                                        out_cond['cond_f']["edge_index"], 
                                                        out_cond['cond_f']['distance_vec'])
edge_attr_cond_f = msg_to_invariants_from_irreps(edge_attr_cond_f, model.model.edge_block.msg_irreps)

conditions = prep['model_kwargs']['conditions']
edge_index_cond_r, to_jimages_cond_r, num_bonds_cond_r = radius_graph_pbc(
    cart_coords=conditions["cond_r"]['x'].view(-1, 3),
    lattice=cell.view(-1, 3, 3),
    num_atoms=num_atoms.view(-1),
    radius=args.cutoff,
    max_num_neighbors_threshold=model.model.max_num_neighbors_threshold,
    max_cell_images_per_dim=model.model.max_cell_images_per_dim,
)
out_cond = {}
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
out_cond['cond_r']['sub_graph_mask'] = None
out_cond['cond_r']['x'] = conditions["cond_r"]["x"].view(-1, 3)
out_cond['cond_r']['cell'] = cell.view(-1,3,3)
out_cond['cond_r']['to_jimages'] = to_jimages_cond_r
out_cond['cond_r']['num_bonds'] = num_bonds_cond_r
out_cond["species"] = species

edge_attr_cond_r = model.model.edge_block(model.model.embed_atom(out_cond['species'].view(-1, model.model.num_species)), 
                                                        out_cond['cond_r']["edge_index"], 
                                                        out_cond['cond_r']['distance_vec'])
edge_attr_cond_r = msg_to_invariants_from_irreps(edge_attr_cond_r, model.model.edge_block.msg_irreps)

cell_r = cell @ R.T
edge_index_r, to_jimages_r, num_bonds_r = radius_graph_pbc(
    cart_coords=x_r.view(-1, 3),
    lattice=cell_r.view(-1, 3, 3),
    num_atoms=num_atoms.view(-1),                                      # The num_atoms is used to separate batched structures before connecting graph
    radius=model.model.cutoff,
    max_num_neighbors_threshold=model.model.max_num_neighbors_threshold,
    max_cell_images_per_dim=model.model.max_cell_images_per_dim,
)

out_r = get_pbc_distances(
    x_r.view(-1, 3),
    edge_index_r,
    cell_r.view(-1, 3, 3),
    to_jimages_r,
    num_atoms.view(-1),
    num_bonds_r,
    coord_is_cart=True,
    return_offsets=True,
    return_distance_vec=True,
)
edge_vec_r = out_r['distance_vec']
# assert torch.allclose(edge_vec_r, edge_vec @ R.T, atol=1e-7)

edge_attr_r = edge_block(f0, edge_index_r, edge_vec_r)  
from mdgen.model.equivariant_latent_model import msg_to_invariants_from_irreps
edge_attr_r = msg_to_invariants_from_irreps(edge_attr_r, edge_block.msg_irreps)

print('Edge_attr err = ', torch.abs(edge_attr-edge_attr_r).max())
assert torch.allclose(edge_attr, edge_attr_r, rtol=1e-4, atol=5e-5)

cond_f_mask = conditions = prep['model_kwargs']['conditions']['cond_f']['mask']
cond_r_mask = conditions = prep['model_kwargs']['conditions']['cond_r']['mask']
print(cond_f_mask)
h, v, edge_attr = model.model.encoder(species.view(-1, model.model.num_species), 
                                      edge_index, 
                                      torch.cat([edge_attr, edge_attr_cond_f, edge_attr_cond_r], dim=-1), 
                                      edge_vec, 
                                      t, None)

h = h + model.model.mask_to_emb_f(cond_f_mask) + model.model.mask_to_emb_r(cond_r_mask)

h_r, v_r, edge_attr_r = model.model.encoder(species.view(-1, model.model.num_species), 
                                            edge_index_r, 
                                            torch.cat([edge_attr_r, edge_attr_cond_f, edge_attr_cond_r], dim=-1), 
                                            edge_vec_r, 
                                            t, None)
h_r = h_r + model.model.mask_to_emb_f(cond_f_mask) + model.model.mask_to_emb_r(cond_r_mask)


print('H err = ', torch.abs(h-h_r).max())
assert torch.allclose(h, h_r, rtol=1e-4, atol=5e-5)
print('V err = ', vec_err(v @ R.T, v_r))
assert torch.allclose(v @ R.T , v_r, rtol=1e-4, atol=5e-5)
'''
H_r, V_r, edge_attr_r = model.model.encoder(species.view(-1, model.model.num_species), edge_index_r, edge_attr_r, edge_vec_r, t)


print("raw V err =", vec_err(V_r, V @ R.T).item())
print("h err =", ((H-H_r).abs()/H.abs()).max())
assert torch.allclose(H, H_r, rtol=1e-3)
assert torch.allclose(V_r, V @ R.T, rtol=1e-3, atol=5e-5   )

with torch.autocast(device_type='cpu', dtype=torch.float64, enabled=True):
    H_processed_r, V_processed_r = model.model.processor(H_r, V_r, edge_index_r, edge_attr_r, edge_len=torch.linalg.norm(edge_vec_r, dim=1, keepdim=True))
    H_decoded_r, V_decoded_r = model.model.decoder(H_processed_r, V_processed_r)
    del edge_index_r
    del edge_attr_r
    del edge_vec_r


A = V_processed_r
B = V_processed @ R.T
diff = (A - B).abs()
print("Processed V err =", vec_err(V_processed_r, V_processed @ R.T).item(), diff.max().item())
print("Processed h err =", ((H_processed-H_processed_r).abs()/H_processed.abs()).max())
assert torch.allclose(H_processed, H_processed_r, rtol=1e-3, atol=1e-6 )
assert torch.allclose(V_processed_r, V_processed @ R.T, rtol=1e-3, atol=1e-5    )

A = V_decoded_r
B = V_decoded @ R.T
diff = (A - B).abs()
print("Decoded V err =", vec_err(V_decoded_r, V_decoded @ R.T).item(), diff.max().item())
#print("Decoded h err =", ((H_decoded-H_decoded_r).abs()/H_decoded.abs()).max())
#assert torch.allclose(H_decoded, H_decoded_r, rtol=1e-2, atol=1e-4 )
assert torch.allclose(V_decoded_r, V_decoded @ R.T, rtol=1e-3, atol=1e-6   )

# ----- Permutation test -----
perm = torch.randperm(H.shape[0], device=H.device)
def permute_graph(H, V, edge_index, edge_attr):
    Hp = H[perm]
    Vp = V[perm]
    # remap edge_index node ids with inverse permutation
    inv = torch.empty_like(perm); inv[perm] = torch.arange(len(perm), device=perm.device)
    eip = inv[edge_index]
    return Hp, Vp, eip, edge_attr

Hp, Vp, eip, eap = permute_graph(H, V, edge_index, edge_attr)
H_proc_p, V_proc_p = model.model.processor(Hp, Vp, eip, eap, edge_len=torch.linalg.norm(edge_vec, dim=1, keepdim=True))

assert torch.allclose(H_processed[perm], H_proc_p, rtol=1e-3, atol=1e-6)
assert torch.allclose(V_processed[perm], V_proc_p, rtol=1e-3, atol=1e-6)
'''

#---------------------------Testing the source code-------------------------------------------------------
#
print('species', prep['model_kwargs']['aatype'].dtype)
from copy import deepcopy
prep_bck = deepcopy(prep)
out_v = model.model.forward(x, t, **prep['model_kwargs']).view(-1,3)
assert torch.allclose(prep_bck['model_kwargs']['conditions']['cond_f']['x'], prep['model_kwargs']['conditions']['cond_f']['x'])

out_v_r = model.model.forward(x_r, t, **prep['model_kwargs']).view(-1,3)
out_v_benchmark = out_v @ R.T 
print("Out V err =", vec_err(out_v_r, out_v_benchmark).item())
assert torch.allclose(out_v_r, out_v_benchmark, rtol=1e-4, atol=5e-5)
