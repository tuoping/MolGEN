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
args.object_aware = True
args.design = False
args.potential_model = False
args.pbc = False 
args.ratio_conditonal = 1

os.environ["MODEL_DIR"] = os.path.join("tests_cache", args.run_name)

from mdgen.equivariant_wrapper import EquivariantMDGenWrapper
model = EquivariantMDGenWrapper(args)
model.eval()
model.iter_step += 1
model.stage = "train"
prep = model.prep_batch(sample_batch)
print(prep['model_kwargs']['v_mask'])
out_dict = model.transport.training_losses(
    model=model.model,
    x1=prep['latents'],
    aatype1=prep['species'],
    mask=prep['loss_mask'],
    model_kwargs=prep['model_kwargs']
)

t, xt, ut = model.transport.path_sampler.plan(out_dict['t'], out_dict['x0'][0], prep['latents'])
model_output = model.model(xt, t, **prep['model_kwargs'])

assert torch.all(out_dict['pred'][:,0,...] == 0)
assert torch.all(out_dict['pred'][:,2,...] == 0)
assert torch.allclose(out_dict['pred'][:,:,...], model_output[:,:,...])
assert torch.all(out_dict['loss_continuous'][:,0,...]==0)
assert torch.all(out_dict['loss_continuous'][:,2,...]==0)
loss_continuous = ((0.5*(model_output)**2 - (ut)*model_output)*prep['loss_mask'])
assert torch.allclose(out_dict['loss_continuous'][:,1,...], loss_continuous[:,1,...])