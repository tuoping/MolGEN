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

args.data_dir="tests/test_data/Transition1x" 
args.num_frames=3

args.ckpt_freq = 10 
args.val_repeat = 0.2
args.epochs = 1
args.num_species = 5

args.edge_dim = 4
args.num_convs = 5
args.num_heads = 4
args.ff_dim = 16

args.cutoff= 12
args.localmask = True
args.path_type = "Linear"
args.batch_size = batch_size

args.tps_condition = True
args.sim_condition = False
# args.prediction='score'
# args.sampling_method = "Euler"
args.object_aware = False
args.design = False
args.potential_model = False
args.pbc = False 
args.ratio_conditonal = 1

os.environ["MODEL_DIR"] = os.path.join("tests_cache", args.run_name)

from mdgen.equivariant_wrapper import EquivariantMDGenWrapper
model = EquivariantMDGenWrapper(args)

model.iter_step += 1
model.stage = "train"
prep = model.prep_batch(sample_batch)

print(prep['latents'].shape)
print(prep['model_kwargs']['conditions']['cond_f']['mask'].reshape(1,3,-1))
assert prep['latents'].shape == sample_batch['x'].shape
assert prep['latents'].shape == (1,3,8,3)
assert prep["model_kwargs"]["conditions"]['cond_f']['mask'].shape == (24,)
assert prep["model_kwargs"]["conditions"]['cond_f']['x'].shape == (24,3)

assert torch.all(prep['loss_mask'][:,0,...] == 0)
assert torch.all(prep['loss_mask'][:,2,...] == 0)
assert torch.all(prep['loss_mask'][:,1,...] == 1)

assert torch.all(prep['model_kwargs']['v_mask'][:,0,...] == 0)
assert torch.all(prep['model_kwargs']['v_mask'][:,2,...] == 0)
assert torch.all(prep['model_kwargs']['v_mask'][:,1,...] == 1)

assert torch.all(prep['model_kwargs']['conditions']['cond_f']['mask'].reshape(1,3,-1)[:,0,...] == 1)
# assert torch.all(prep['model_kwargs']['conditions']['cond_f']['mask'].reshape(1,3,-1)[:,1,...] == 0)
# assert torch.all(prep['model_kwargs']['conditions']['cond_f']['mask'].reshape(1,3,-1)[:,2,...] == 0)

# assert torch.all(prep['model_kwargs']['conditions']['cond_r']['mask'].reshape(1,3,-1)[:,0,...] == 0)
# assert torch.all(prep['model_kwargs']['conditions']['cond_r']['mask'].reshape(1,3,-1)[:,1,...] == 0)
assert torch.all(prep['model_kwargs']['conditions']['cond_r']['mask'].reshape(1,3,-1)[:,2,...] == 1)
