import torch, os
from mdgen.dataset import EquivariantTransformerDataset_Transition1x
batch_size = 1

trainset = EquivariantTransformerDataset_Transition1x("tests/test_data/Transition1x/", 12, sim_condition=False, tps_condition=True, stage="example-multi-fragments")
train_loader = torch.utils.data.DataLoader(
    trainset,
    batch_size=batch_size,
    num_workers=0,
    shuffle=True,
)
sample_batch = next(iter(train_loader))
'''
condset = EquivariantTransformerDataset_Transition1x("tests/test_data/Transition1x/", 12, sim_condition=False, tps_condition=True, stage="train")
cond_loader = torch.utils.data.DataLoader(
    condset,
    batch_size=batch_size,
    num_workers=0,
    shuffle=True,
)
cond_2 = next(iter(cond_loader))
'''

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


species = prep['species'].double()
cell = prep['model_kwargs']['cell'].double()
num_atoms = prep['model_kwargs']['num_atoms']

print("#-------------------testing the effect of changing \'cond_f\'----------------------------")
from copy import deepcopy
prep_test_2 = deepcopy(prep)
prep_test_2['model_kwargs']['conditions']['cond_f']['x'] = prep_test_2['model_kwargs']['conditions']['cond_f']['x'] + torch.rand_like(prep_test_2['model_kwargs']['conditions']['cond_f']['x'])
assert not torch.allclose(prep_test_2['model_kwargs']['conditions']['cond_f']['x'], prep['model_kwargs']['conditions']['cond_f']['x'])

#----------------------------------------Testing components------------------------------------------------

R = so3_random(x.device)
print(R)

# x_r = x @ R.T  + torch.randn(3)[None,None,None,:]


from mdgen.model.utils.data_utils import (
    get_pbc_distances,
    radius_graph_pbc,
)
from mdgen.model.equivariant_latent_model import msg_to_invariants_from_irreps

print('>>> checking edge_attr_f <<<')
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

conditions_2 = prep_test_2['model_kwargs']['conditions']
edge_index_cond_f, to_jimages_cond_f, num_bonds_cond_f = radius_graph_pbc(
    cart_coords=conditions_2["cond_f"]['x'].view(-1, 3),
    lattice=cell.view(-1, 3, 3),
    num_atoms=num_atoms.view(-1),
    radius=args.cutoff,
    max_num_neighbors_threshold=model.model.max_num_neighbors_threshold,
    max_cell_images_per_dim=model.model.max_cell_images_per_dim,
)
out_cond_2 = {}
out_cond_2['cond_f'] = get_pbc_distances(
    conditions_2["cond_f"]["x"].view(-1, 3),
    edge_index_cond_f,
    cell.view(-1, 3, 3),
    to_jimages_cond_f,
    num_atoms.view(-1),
    num_bonds_cond_f,
    coord_is_cart=True,
    return_offsets=True,
    return_distance_vec=True,
)
out_cond_2['cond_f']['sub_graph_mask'] = None
out_cond_2['cond_f']['x'] = conditions_2["cond_f"]["x"].view(-1, 3)
out_cond_2['cond_f']['cell'] = cell.view(-1,3,3)
out_cond_2['cond_f']['to_jimages'] = to_jimages_cond_f
out_cond_2['cond_f']['num_bonds'] = num_bonds_cond_f
out_cond_2["species"] = species

edge_attr_cond_f_2 = model.model.edge_block(model.model.embed_atom(out_cond_2['species'].view(-1, model.model.num_species)), 
                                                        out_cond_2['cond_f']["edge_index"], 
                                                        out_cond_2['cond_f']['distance_vec'])
edge_attr_cond_f_2 = msg_to_invariants_from_irreps(edge_attr_cond_f_2, model.model.edge_block.msg_irreps)
print('Edge_attr_cond_f err = ', torch.abs(edge_attr_cond_f-edge_attr_cond_f_2).max())
assert not torch.allclose(edge_attr_cond_f, edge_attr_cond_f_2)
print('>>> DONE <<<')
print('>>> Testing rotation and translation invariance of  edge_attr_conf_f <<<')
from copy import deepcopy
conditions_r = deepcopy(prep['model_kwargs']['conditions'])
conditions_r['cond_f']['x'] = conditions_r['cond_f']['x'] @ R.T + torch.randn(3)[None,:]
edge_index_cond_f_r, to_jimages_cond_f_r, num_bonds_cond_f_r = radius_graph_pbc(
    cart_coords=conditions_r["cond_f"]['x'].view(-1, 3),
    lattice=cell.view(-1, 3, 3),
    num_atoms=num_atoms.view(-1),
    radius=args.cutoff,
    max_num_neighbors_threshold=model.model.max_num_neighbors_threshold,
    max_cell_images_per_dim=model.model.max_cell_images_per_dim,
)
out_cond_r = {}
out_cond_r['cond_f'] = get_pbc_distances(
    conditions_r["cond_f"]["x"].view(-1, 3),
    edge_index_cond_f_r,
    cell.view(-1, 3, 3),
    to_jimages_cond_f_r,
    num_atoms.view(-1),
    num_bonds_cond_f_r,
    coord_is_cart=True,
    return_offsets=True,
    return_distance_vec=True,
)
out_cond_r['cond_f']['sub_graph_mask'] = None
out_cond_r['cond_f']['x'] = conditions_r["cond_f"]["x"].view(-1, 3)
out_cond_r['cond_f']['cell'] = cell.view(-1,3,3)
out_cond_r['cond_f']['to_jimages'] = to_jimages_cond_f_r
out_cond_r['cond_f']['num_bonds'] = num_bonds_cond_f_r
out_cond_r["species"] = species

edge_attr_cond_f_r = model.model.edge_block(model.model.embed_atom(out_cond['species'].view(-1, model.model.num_species)), 
                                                        out_cond_r['cond_f']["edge_index"], 
                                                        out_cond_r['cond_f']['distance_vec'])
edge_attr_cond_f_r = msg_to_invariants_from_irreps(edge_attr_cond_f_r, model.model.edge_block.msg_irreps)
print('Edge_attr_cond_f err = ', torch.abs(edge_attr_cond_f-edge_attr_cond_f_r).max())
assert torch.allclose(edge_attr_cond_f, edge_attr_cond_f_r)

print(">>> DONE <<<")

print('#-------------------- Testing invariance of H, V, edge_attr with rotation and translation of \'cond_f\'------------------')
out_cond = {}
conditions = prep['model_kwargs']['conditions']
edge_index_cond_r, to_jimages_cond_r, num_bonds_cond_r = radius_graph_pbc(
    cart_coords=conditions["cond_r"]['x'].view(-1, 3),
    lattice=cell.view(-1, 3, 3),
    num_atoms=num_atoms.view(-1),
    radius=args.cutoff,
    max_num_neighbors_threshold=model.model.max_num_neighbors_threshold,
    max_cell_images_per_dim=model.model.max_cell_images_per_dim,
)
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
edge_vec = out['distance_vec']
edge_attr = model.model.edge_block(model.model.embed_atom(species.view(-1, model.model.num_species)), 
                                                        edge_index, 
                                                        out['distance_vec'])
edge_attr = msg_to_invariants_from_irreps(edge_attr, model.model.edge_block.msg_irreps)

cond_f_mask = conditions = prep['model_kwargs']['conditions']['cond_f']['mask']
cond_r_mask = conditions = prep['model_kwargs']['conditions']['cond_r']['mask']

h, v, _edge_attr = model.model.encoder(species.view(-1, model.model.num_species), 
                                      edge_index, 
                                      torch.cat([edge_attr, edge_attr_cond_f, edge_attr_cond_r], dim=-1), 
                                      edge_vec, 
                                      t, None)

h = h + model.model.mask_to_emb_f(cond_f_mask) + model.model.mask_to_emb_r(cond_r_mask)

h_r, v_r, _edge_attr_r = model.model.encoder(species.view(-1, model.model.num_species), 
                                            edge_index, 
                                            torch.cat([edge_attr, edge_attr_cond_f_r, edge_attr_cond_r], dim=-1), 
                                            edge_vec, 
                                            t, None)
h_r = h_r + model.model.mask_to_emb_f(cond_f_mask) + model.model.mask_to_emb_r(cond_r_mask)


print('H err = ', torch.abs(h-h_r).max())
assert torch.allclose(h, h_r, rtol=1e-4, atol=5e-5)
print('V err = ', vec_err(v, v_r))
assert torch.allclose(v, v_r, rtol=1e-4, atol=5e-5)
print('Edge attr err = ', torch.abs(_edge_attr - _edge_attr_r).max())
assert torch.allclose(_edge_attr, _edge_attr_r, rtol=1e-4, atol=5e-5)

h, v = model.model.processor(h, v, edge_index, _edge_attr, edge_len=torch.linalg.norm(edge_vec, dim=1, keepdim=True))
h_r, v_r = model.model.processor(h_r, v_r, edge_index, _edge_attr_r, edge_len=torch.linalg.norm(edge_vec, dim=1, keepdim=True))
print('Processor: H err = ', torch.abs(h-h_r).max())
assert torch.allclose(h, h_r, rtol=1e-4, atol=5e-5)
print('Processor: V err = ', vec_err(v, v_r))
assert torch.allclose(v, v_r, rtol=1e-4, atol=5e-5)

h, v = model.model.decoder(h, v)
h_r, v_r = model.model.decoder(h_r, v_r)

print('Decoder: V err = ', vec_err(v, v_r))
assert torch.allclose(v, v_r, rtol=1e-4, atol=5e-5)

print('#---------------------------Testing the source code-------------------------------------------------------#')

out_v = model.model.forward(x, t, **prep['model_kwargs']).view(-1,3)
out_v_2 = model.model.forward(x, t, **prep_test_2['model_kwargs']).view(-1,3)

print("Out V err =", vec_err(out_v_2, out_v).item())
assert not torch.allclose(out_v_2, out_v, rtol=1e-4, atol=5e-5)

print('#-------------------testing the effect of changing \'cond_r\'----------------------------')

prep_test_3 = deepcopy(prep)
prep_test_3['model_kwargs']['conditions']['cond_r']['x'] = prep_test_3['model_kwargs']['conditions']['cond_r']['x'] + torch.rand_like(prep_test_3['model_kwargs']['conditions']['cond_r']['x'])
assert not torch.allclose(prep_test_3['model_kwargs']['conditions']['cond_r']['x'], prep['model_kwargs']['conditions']['cond_r']['x'])
out_v_3 = model.model.forward(x, t, **prep_test_3['model_kwargs']).view(-1,3)

print("Out V err =", vec_err(out_v_3, out_v).item())
assert not torch.allclose(out_v_3, out_v, rtol=1e-4, atol=5e-5)

print('#---------------------testing rotation and translation invariance of \'cond_f\'-------------------------')

prep_test_4 = deepcopy(prep)
prep_test_4['model_kwargs']['conditions']['cond_f']['x'] = conditions_r['cond_f']['x']
out_v_4 = model.model.forward(x, t, **prep_test_4['model_kwargs']).view(-1,3)

tps_out_v_4 = out_v_4.view(B, T, L, 3)[:,1,...].view(-1,3)
tps_v_r = v_r.view(B, T, L, 3)[:,1,...].view(-1,3)

print('Check source code', vec_err(tps_out_v_4, tps_v_r))
assert torch.allclose(tps_out_v_4, tps_v_r, rtol=1e-4, atol=5e-5)
print("Out V err =", vec_err(out_v_4, out_v).item())
assert torch.allclose(out_v_4, out_v, rtol=1e-4, atol=1e-4)