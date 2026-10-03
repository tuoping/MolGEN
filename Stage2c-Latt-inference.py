import glob
import os
import re

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
### SDE
inference_steps = 100
sampling_method = "euler"

Sanity_check = True
guidance = False
if guidance:
    suffix = "_gtcubic"
else:
    suffix = ""

### Stage 2
run_tag=2
ckpt_tag = 365
stage_tag = 2
stage_subdirs = {
    2: "design_noTime/Stage2",
}
sim_ckpt = glob.glob(f"workdir/{stage_subdirs[stage_tag]}/run{run_tag}/epoch={ckpt_tag:03d}-step=*.ckpt")[0]


# run_tag_stage_1 = 13
# ckpt_tag_stage_1 = 195
# inference_steps_stage_1 = 100
# bk_tag_stage_1 = 0
# stage_tag_1 = 2.1
# suffix_stage_1 = "_gtcubic"

subfolder = "TRAJ/step6"


# print(f"experiments/MOF/{subfolder}/"
#             f"sde_TSMloss_r{run_tag_stage_1}e{ckpt_tag_stage_1}_"
#             f"euler_step{inference_steps_stage_1}{suffix_stage_1}/gentraj_*.xyz")
# data_dir = sorted(
#         glob.glob(
#             f"experiments/MOF/{subfolder}/"
#             f"sde_TSMloss_r{run_tag_stage_1}e{ckpt_tag_stage_1}_"
#             f"euler_step{inference_steps_stage_1}{suffix_stage_1}/gentraj_*.xyz"
#         ),
#         key=lambda x: int(re.search(r"gentraj_(\d+)\.xyz$", x).group(1))
#     )
data_dir = sorted(
        glob.glob(
            f"experiments/MOF/{subfolder}/gentraj_*.xyz"
        ),
        key=lambda x: int(re.search(r"gentraj_(\d+)\.xyz$", x).group(1))
    )


# inference_out_dir = f"experiments/MOF/{stage_subdirs[stage_tag]}/{subfolder}/test_stage1.bk{bk_tag_stage_1:.1f}_r{run_tag_stage_1}e{ckpt_tag_stage_1}_step{inference_steps_stage_1}{suffix_stage_1}/sde_TSMloss_r{run_tag}e{ckpt_tag}_{sampling_method}_step{inference_steps}{suffix}/"
inference_out_dir = f"experiments/MOF/TRAJ/step7/"



import torch, tqdm, time
import numpy as np
from mdgen.equivariant_wrapper import EquivariantMDGenWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('medium')


# Keep the checkpoint state dict off the GPU. Loading it directly onto CUDA
# otherwise leaves a second full copy of every parameter alive in `ckpt`.
ckpt = torch.load(sim_ckpt, map_location="cpu", weights_only=False)
hparams = ckpt["hyper_parameters"]
args = hparams['args']
args.sampling_method = sampling_method
args.inference_steps = inference_steps


stage_prefix = ""
if Sanity_check:
    args.data_dir = "data/MOF/CoRE_MOF/CR-most_frequent_composition_C12H8I2N4Zn/ASR/"
else:
    args.data_dir = data_dir


import sys
try:
    args.prior_std = float(sys.argv[1])
except:
    args.prior_std = 1
args.target_std = 0.0
args.likelihood = None
args.K_hutchinson_probe = 1
args.K_hutchinson_probe_chunk = 1
args.guidance = guidance


from mdgen.spring import ZBLRepulsiveWall
if args.guidance:
    wall = ZBLRepulsiveWall()


def _guidance(_x, t, **kwargs):
    # SDE sampling runs in inference mode. Clone its tensors with inference
    # mode disabled so autograd can compute the energy gradient with respect
    # to the current coordinates only.
    with torch.inference_mode(False):
        x = _x.detach().clone().requires_grad_(True)
        # `cell` was created by the sampler's outer inference_mode context.
        # Merely disabling inference mode does not convert an existing
        # inference tensor: clone it here so ZBL's backward pass may save it.
        cell = kwargs['cell'].detach().clone()
        # guidance_t = t.detach().clone()
        # guidance_kwargs = {
        #     key: value.detach().clone() if torch.is_tensor(value) else value
        #     for key, value in kwargs.items()
        # }
        # energy = guidance_model.potential_model(
        #     x,
        #     guidance_t,
        #     **guidance_kwargs,
        # )
        # grad_frac = torch.autograd.grad(energy.sum(), x)[0]
        # # force = -torch.einsum(
        # #     "btni,btij->btnj",
        # #     grad_frac,
        # #     torch.linalg.inv(guidance_kwargs["cell"]).transpose(-1, -2),
        # # )
        labels = torch.argmax(kwargs['aatype'][...,:71], dim=-1).squeeze(0).squeeze(0)  # T,L
        atomic_numbers = torch.tensor([species[label] for label in labels.flatten()]).to(x.device)  # T,L
        wall_force = wall.build_force(
            x.squeeze(1),
            cell.squeeze(0).squeeze(0),
            atomic_numbers,
        ) @ cell.transpose(-1, -2)
    # return -grad_frac.detach()
    return wall_force * 0.0001


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
# print(model.model)
model.load_state_dict(ckpt["state_dict"], strict=True)
del ckpt, hparams

if args.guidance:
    guidance = _guidance
    model.transport._init_guidance(guidance)


model.eval().to(device)
# print(model.args)
print(model.args.path_type)
print(model.args.sampling_method)
print(model.args.inference_steps)
print(model.args.likelihood)

print("Data folder: ", args.data_dir)

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
for _inference in [True]:

    if _inference:
        out_dir = inference_out_dir
    else:
        try:
            out_dir = f"{inference_out_dir}/noiseprior_std{args.prior_std}_sample{sys.argv[2]}"
        except:
            print(f"noiseprior_std{args.prior_std}_sample1")
            out_dir = f"{inference_out_dir}/noiseprior_std{args.prior_std}_sample1"
    if Sanity_check:
        inference = False
        out_dir = f"experiments/MOF/{stage_subdirs[stage_tag]}/sde_TSMloss_r{run_tag}e{ckpt_tag}_{sampling_method}_step{inference_steps}"
    else:
        inference = _inference


    print("Output folder: ", out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/README.md", "w") as fp:
        fp.write(sim_ckpt)

    all_rollout_atoms_ref_0 = []
    all_rollout_atoms = []
    all_rollout_atoms_ref = []
    start = time.time()
    all_logp = []
    for i_rollout in range(1): #, len(dataset)):
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
            print("rollout", i_rollout, "idx = ", idx+i_sample)
            if not args.design:
                labels = torch.argmax(batch["species"], dim=3).squeeze(0)
                symbols = [[map_to_chemical_symbol[int(i_elem.to('cpu'))] for i_elem in labels[i_conf]] for i_conf in range(len(labels))]
                formula = "".join(symbols[0])
                ref_labels = labels
                ref_symbols = symbols
                ref_formula = formula
            else:
                ref_labels = torch.argmax(batch["species"], dim=3).squeeze(0)
                ref_symbols = [[map_to_chemical_symbol[int(i_elem.to('cpu'))] for i_elem in ref_labels[i_conf]] for i_conf in range(len(ref_labels))]
                ref_formula = "".join(ref_symbols[0])
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
                    all_pred_frac_pos, aatype  = model.inference(batch)
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
                if args.design:
                    labels = torch.argmax(aatype[idx_traj][...,:71], dim=3).squeeze(0)
                    symbols = [[map_to_chemical_symbol[int(i_elem.to('cpu'))] for i_elem in labels[i_conf]] for i_conf in range(len(labels))]
                    formula = "".join(symbols[0])

                atoms = Atoms(formula, positions=pred_pos.detach().cpu().numpy(), cell=cell_out[0][0].detach().cpu().numpy(), pbc=[1,1,1])
                write(filename, atoms, append=True)

            ref_pos = batch["x"][0][0] @ batch['cell'][0][0]
            atoms_ref = Atoms(ref_formula, positions=ref_pos.cpu().numpy(), cell=batch['cell'][0][0].cpu().numpy(), pbc=[1,1,1])
            write(filename_ref, atoms_ref, append=True)


            # del pred_frac_pos
            # del pred_pos
            # del cell_out
            # del ref_pos
