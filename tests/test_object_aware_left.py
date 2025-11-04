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

t = torch.ones(B).double()



R = torch.from_numpy(rand_rot()).double()
# print(R)


x_r = x @ R.T * ((prep['model_kwargs']['fragments_idx'] == 0).unsqueeze(-1)) + x * ((prep['model_kwargs']['fragments_idx'] != 0).unsqueeze(-1))


fragments_idx_V = prep['model_kwargs']['fragments_idx'].reshape(-1).unsqueeze(-1).unsqueeze(-1)

def vec_err(A, B, eps=1e-12):
    # A,B: [N,C,3]
    num = torch.linalg.norm(A - B, dim=-1)               # [N,C]
    den = torch.linalg.norm(A, dim=-1) + torch.linalg.norm(B, dim=-1) + eps
    return (num / den).max()   # scalar


out_v = model.model.forward(x, t, **prep['model_kwargs']).view(-1,3)

out_v_r = model.model.forward(x_r, t, **prep['model_kwargs']).view(-1,3)

out_v_benchmark = out_v @ R.T * (fragments_idx_V.view(-1,1) == 0) + out_v * (fragments_idx_V.view(-1,1) != 0)
print("Out V err =", vec_err(out_v_r, out_v_benchmark).item())
assert torch.allclose(out_v_r, out_v_benchmark, rtol=1e-3, atol=5e-5)