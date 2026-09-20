# from mdgen.parsing import parse_train_args
# args = parse_train_args()

import glob
import os

# Dynamic neighbor graphs use differently sized CUDA allocations at each SDE
# step. Expandable segments reduce allocator fragmentation for this workload.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
### Stage 1
# run_tag=6
# ckpt_tag = 684
# stage_tag = 1
### Stage 2
run_tag=5
ckpt_tag = 99
stage_tag = 2

run_tag_stage_1 = 2
ckpt_tag_stage_1 = 93
inference_steps_stage_1 = 1000
bk_tag_stage_1 = 2
suffix_stage_1 = "_cubicts"

### ODE
# inference_steps = 50
# sampling_method = "dopri5"
### SDE
inference_steps = 1000
sampling_method = "euler"

### Stage 2
# sim_ckpt = glob.glob(f"workdir/bk.0.Stage2.priorD1/run{run_tag}.linear/epoch={ckpt_tag:03d}-step=*.ckpt")[0]
sim_ckpt = glob.glob(f"workdir/bk.0.Stage2.priorD1/run{run_tag}/last.ckpt")[0]

import torch, tqdm, time
import numpy as np
from mdgen.equivariant_wrapper import EquivariantMDGenWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('medium')

def _init_guidance(_x, t, **kwargs):
    _ckpt = torch.load("workdir/Springff/epoch=029-step=0000960-val_loss=0.0018.ckpt", weights_only=False, map_location=torch.device(device))
    _args = _ckpt['args']
    _hparams = _ckpt['args']
    _guidance_ener = EquivariantMDGenWrapper(_args).to(device)
    _guidance_ener.load_state_dict(_ckpt['state_dict'], strict=True)
    return -torch.einsum(
                    "btni,btij->btnj",
                    torch.autograd.grad(
                        _guidance_ener(_x, t, kwargs).sum(),
                    ),
                    torch.linalg.inv(kwargs['cell']).transpose(-1, -2),
                )


# Keep the checkpoint state dict off the GPU. Loading it directly onto CUDA
# otherwise leaves a second full copy of every parameter alive in `ckpt`.
ckpt = torch.load(sim_ckpt, map_location="cpu", weights_only=False)
hparams = ckpt["hyper_parameters"]
args = hparams['args']
args.sampling_method = sampling_method
args.inference_steps = inference_steps
args.data_dir = f"data/MOF/CoRE_MOF/CR/bk.{bk_tag_stage_1}.ASR.Stage1_r{run_tag_stage_1}e{ckpt_tag_stage_1}{suffix_stage_1}/"
# args.data_dir = "data/MOF/CoRE_MOF/CR/ASR/"
import sys
args.prior_std = float(sys.argv[1])
args.likelihood = None
args.K_hutchinson_probe = 1
args.K_hutchinson_probe_chunk = 1
species = [1, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21,
                                                22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 39, 40,
                                                41, 42, 44, 45, 46, 47, 48, 49, 51, 53, 55, 57, 58, 59, 60, 62, 63,
                                                64, 65, 66, 67, 68, 69, 70, 71, 72, 74, 77, 78, 79, 80, 82, 83, 90,
                                                92, 93, 94]
from mdgen.dataset import EquivariantTransformerDataset_MaterialProject
dataset = EquivariantTransformerDataset_MaterialProject(
                                        args, 
                                        species=species, 
                                        num_species=args.num_species, 
                                        sim_condition=False, 
                                        stage="test",)
model = EquivariantMDGenWrapper(**hparams)
print(model.model)
model.load_state_dict(ckpt["state_dict"], strict=True)
del ckpt, hparams
model.eval().to(device)
print(model.args)
print(model.args.path_type)
print(model.args.sampling_method)
print(model.args.inference_steps)
print(model.args.likelihood)
batch_size = 1
val_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=0,
    shuffle=True,
)
sample_batch = next(iter(val_loader))
from ase.data import chemical_symbols
map_to_chemical_symbol = {}
for i in range(len(species)):
    map_to_chemical_symbol[i] = chemical_symbols[species[i]]
idx_rollouts = np.arange(len(dataset))
from ase import Atoms
from ase.geometry.geometry import get_distances
import shutil, os
from ase.io import write

assert len(dataset) >= 1
for inference in [True, ]:

    if inference:
        out_dir = f"experiments/MOF/bk.0.Stage2.priorD1/test_stage1.bk{bk_tag_stage_1:02d}_r{run_tag_stage_1}e{ckpt_tag_stage_1}_step{inference_steps_stage_1}{suffix_stage_1}/sde_TSMloss_r{run_tag}e{ckpt_tag}_{sampling_method}_step{inference_steps}/"
    else:
        out_dir = f"experiments/MOF/bk.0.Stage2.priorD1/test_stage1.bk{bk_tag_stage_1:02d}_r{run_tag_stage_1}e{ckpt_tag_stage_1}_step{inference_steps_stage_1}{suffix_stage_1}/sde_TSMloss_r{run_tag}e{ckpt_tag}_{sampling_method}_step{inference_steps}/noiseprior_std{args.prior_std}_sample{sys.argv[2]}"
    # out_dir = f"experiments/MOF/bk.0.Stage2.priorD1/r{run_tag}e{ckpt_tag}_{sampling_method}_step{inference_steps}"
    print("Output folder: ", out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/README.md", "w") as fp:
        fp.write(sim_ckpt)

    all_rollout_atoms_ref_0 = []
    all_rollout_atoms = []
    all_rollout_atoms_ref = []
    start = time.time()
    all_logp = []
    for i_rollout in range(0, 1): # len(dataset)):
        # idx = idx_rollouts[i_rollout]
        idx = i_rollout
        print(i_rollout, idx)
        filename = os.path.join(out_dir, f"gentraj_{idx}.xyz")
        filename_ref = os.path.join(out_dir, f"reftraj_{idx}.xyz")
        for f in [filename, filename_ref, ]:
            if os.path.exists(f):
                os.remove(f)

        if args.likelihood is not None:
            filename_reverse = os.path.join(out_dir, f"reverse_gentraj_{idx}.xyz")
            filename_logp = os.path.join(out_dir, f"Logp_{idx}.txt")
            filename_reverse_logp = os.path.join(out_dir, f"reverse_Logp_{idx}.txt")
            filename_zs = os.path.join(out_dir, f"Uzs_{idx}.txt")
            for f in [filename_reverse, filename_logp, filename_reverse_logp, filename_zs]:
                if os.path.exists(f):
                    os.remove(f)

        for i_sample in range(1):
            item = dataset.__getitem__(idx, inference=inference)
            batch = next(iter(torch.utils.data.DataLoader([item])))

            for key in batch.keys():
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            x0std = None
            labels = torch.argmax(batch["species"], dim=3).squeeze(0)
            symbols = [[map_to_chemical_symbol[int(i_elem.to('cpu'))] for i_elem in labels[i_conf]] for i_conf in range(len(labels))]

            print("rollout", i_rollout, "idx = ", idx+i_sample)
            formula = "".join(symbols[0])
            if model.transport.latt_path:
                if args.likelihood is None:
                    all_pred_frac_pos, _, all_cell  = model.inference(batch)
                else:
                    raise Exception("Not verified")
                    logp, all_pred, _, zs = model.inference(batch)
                    all_pred_frac_pos = all_pred[0]
                    all_cell = all_pred[1]
                    pred_zs_cell = all_pred_reverse[1][-1]
                    np.savetxt(filename_logp, torch.tensor(logp).view(1,-1).detach().cpu().numpy() )
                    N = all_pred_frac_pos.shape[-2]
                    sigma = (torch.ones_like(zs) * x0std)
                    np.savetxt(filename_zs, [[( (zs@all_cell[0])**2/2/sigma**2).sum().detach().cpu().numpy(), ( (pred_zs@pred_zs_cell)**2/2/sigma**2).sum().detach().cpu().numpy()]])
            else:
                if args.likelihood is None:
                    all_pred_frac_pos, _  = model.inference(batch)
                else:
                    logp, all_pred_frac_pos, _, zs = model.inference(batch)
                    cell = batch['cell0'].cpu()
                    N = all_pred_frac_pos.shape[-2]

                    logp = torch.tensor(logp).detach().cpu()
                    np.savetxt(filename_logp, logp.view(1,-1).numpy() )

                    sigma = (torch.ones_like(zs) * x0std).detach().cpu()
                    np.savetxt(filename_zs, [[( (zs.detach().cpu()@cell)**2/2/sigma**2).sum().numpy(), ]])
            # if i_rollout == 0:
            dump_idx = range(len(all_pred_frac_pos))
            # else:
            #     dump_idx = [-1]
            for idx_traj in dump_idx:
            # for idx_traj in [-1]:
                pred_frac_pos = all_pred_frac_pos[idx_traj][0]
                if model.transport.latt_path:
                    cell_out = all_cell[idx_traj]
                    pred_pos = pred_frac_pos[0] @ cell_out[0][0]
                else:
                    cell_out = batch['cell']
                    pred_pos = pred_frac_pos[0] @ cell_out[0][0]

                atoms = Atoms(formula, positions=pred_pos.detach().cpu().numpy(), cell=cell_out[0][0].detach().cpu().numpy(), pbc=[1,1,1])
                write(filename, atoms, append=True)

            ref_pos = batch["x"][0][0] @ batch['cell'][0][0]
            atoms_ref = Atoms(formula, positions=ref_pos.cpu().numpy(), cell=batch['cell'][0][0].cpu().numpy(), pbc=[1,1,1])
            write(filename_ref, atoms_ref, append=True)


            # del pred_frac_pos
            # del pred_pos
            # del cell_out
            # del ref_pos
