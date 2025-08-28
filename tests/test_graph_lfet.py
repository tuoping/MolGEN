import torch, os
from mdgen.dataset import EquivariantTransformerDataset_Transition1x

trainset = EquivariantTransformerDataset_Transition1x("tests/test_data/Transition1x/", 12, sim_condition=False, tps_condition=True, stage="train")

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

args.tps_condition = False
args.sim_condition = False
# args.prediction='score'
# args.sampling_method = "Euler"

args.design = False
args.potential_model = False
args.pbc = False 

os.environ["MODEL_DIR"] = os.path.join("tests_cache", args.run_name)

from mdgen.equivariant_wrapper import EquivariantMDGenWrapper
model = EquivariantMDGenWrapper(args)

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


from torch_geometric.utils import scatter
def _encoder(species, edge_index, edge_attr, edge_vec):
    num_nodes = len(species)
    edge_attr = model.model.encoder.embed_bond(edge_attr)
    f = model.model.encoder.embed_atom(species)
    i = edge_index[0]
    j = edge_index[1]   

    e = torch.cat([f[i], f[j], edge_attr], dim=-1)

    h0 = model.model.encoder.phi_h(torch.cat([
            f, scatter(model.model.encoder.phi_s(e) * f[i], index=j, dim=0, dim_size=num_nodes)
    ], dim=-1))
    v0 = scatter(edge_vec[:, None, :] * model.model.encoder.phi_v(e)[:, :, None], index=j, dim=0, dim_size=num_nodes)
    return h0, v0

x = x1
B,T,L,_ = x.shape
species = prep['species']
cell = prep['model_kwargs']['cell']
num_atoms = prep['model_kwargs']['num_atoms']
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
edge_attr = model.model.scalarize(x.view(-1, 3), edge_index, edge_vec, cell.view(-1,3,3), to_jimages, num_bonds)
# H, V = _encoder(species.view(-1, model.model.num_species), edge_index, edge_attr, edge_vec)
t = torch.ones(B)
H, V, edge_attr = model.model.encoder(species.view(-1, model.model.num_species), edge_index, edge_attr, edge_vec, t)
fte_V = model.model.FTE_tensorize(V)
# fte_V = V

assert torch.all(out['distances'] > 0)

# R = torch.from_numpy(rand_rot()).to(torch.float32)
def so3_random(device):
    # Draw a random rotation with det=+1
    Q, _ = torch.linalg.qr(torch.randn(3,3, device=device))
    if torch.linalg.det(Q) < 0: Q[:,0] = -Q[:,0]
    return Q

R = so3_random(x.device)
print(R)

## Rotate Fi
Fi      = model.model.Fi.clone()              # [N,3,3]
Fi_rot  = Fi@R.T

x_r = x @ R.T   
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
assert torch.allclose(edge_vec_r, edge_vec @ R.T, atol=1e-7)
edge_attr_r = model.model.scalarize(x_r.view(-1, 3), edge_index_r, edge_vec_r, cell_r.view(-1,3,3), to_jimages_r, num_bonds_r)

def vec_err(A, B, eps=1e-12):
    # A,B: [N,C,3]
    num = torch.linalg.norm(A - B, dim=-1)               # [N,C]
    den = torch.linalg.norm(A, dim=-1) + torch.linalg.norm(B, dim=-1) + eps
    return (num / den).max()   # scalar

# Edge frame primary axis (should already hold if your earlier assert on edge_vec_r passes)
e1     = edge_vec / edge_vec.norm(dim=-1, keepdim=True)
e1_r   = edge_vec_r / edge_vec_r.norm(dim=-1, keepdim=True)
print('edge_vec_err =', (e1_r - e1 @ R.T).abs().max().item())
# print('com_err =', (com_i_r - com_i @ R.T).abs().max().item())

# H_r, V_r = _encoder(species.view(-1, model.model.num_species), edge_index_r, edge_attr_r, edge_vec_r)
H_r, V_r, edge_attr_r = model.model.encoder(species.view(-1, model.model.num_species), edge_index_r, edge_attr_r, edge_vec_r, t)
err_h = ((H - H_r).abs() / H.abs()).max()
fte_V_r = model.model.FTE_tensorize(V_r)
# fte_V_r = V_r
# Fi(xR) = RFi(x)
Fi_r = model.model.Fi
err_F = torch.max(torch.linalg.norm(Fi_r - Fi_rot, dim=(-2,-1)))
print('frame_err=', err_F.item())
# Orthonormality sanity (should be ~0)
I = torch.eye(3, device=Fi.device)
orth_err = (torch.linalg.norm(torch.einsum('nij,nkj->nik', Fi,   Fi)   - I) +
            torch.linalg.norm(torch.einsum('nij,nkj->nik', Fi_r, Fi_r) - I)).item()
print('orth_err =', orth_err)

print("raw V err =", vec_err(V_r, V @ R.T).item())
print("after FTE err =", vec_err(fte_V_r, fte_V @ R.T).item())

print("h err =", ((H-H_r).abs()/H.abs()).max())
assert torch.allclose(H, H_r, rtol=1e-3)
assert torch.allclose(V_r, V @ R.T, rtol=1e-3, atol=5e-5   )
assert torch.allclose(fte_V_r, fte_V @ R.T, rtol=1e-3, atol=5e-5   )
