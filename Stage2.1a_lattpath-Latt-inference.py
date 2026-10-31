import glob
import os
import re
import ase.io
import torch, tqdm, time
import numpy as np
from mdgen.equivariant_wrapper import EquivariantMDGenWrapper


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('medium')
# Keep the checkpoint state dict off the GPU. Loading it directly onto CUDA
# otherwise leaves a second full copy of every parameter alive in `ckpt`.

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
### SDE
inference_steps = 100
sampling_method = "euler"

Sanity_check = False
guidance = False
WALL_FORCE_SCALE = 10
if guidance:
    suffix = f"_gtsqr{WALL_FORCE_SCALE}"
else:
    suffix = ""
### Stage 2
'''
## noTime
run_tag=13
ckpt_tag = 195
stage_tag = 2.1
stage_subdirs = {
    2.1: "noTime/Stage2.1",
}
sim_ckpt = glob.glob(f"workdir/{stage_subdirs[stage_tag]}/run{run_tag}.graph/best_val_loss_path/best-val_loss_graph-epoch={ckpt_tag:03d}-step=*.ckpt")[0]

'''
## lattpath_noTime
run_tag=4
ckpt_tag = 359
stage_tag = 2.1
stage_subdirs = {
    2.1: "lattpath_noTime/Stage2.1",
}
# sim_ckpt = glob.glob(f"workdir/{stage_subdirs[stage_tag]}/run{run_tag}/epoch={ckpt_tag:03d}-step=*.ckpt")[0]
sim_ckpt = glob.glob(f"workdir/{stage_subdirs[stage_tag]}/run{run_tag}.forcenoise/best_val_loss_path/best-val_*-epoch={ckpt_tag:03d}-step=*.ckpt")[0]



# subfolder = "lattpath_noTime/Stage1.1/lattpath_noTime/bk.1.Stage1/test_stage1.bk01_r4e459_step100/"
subfolder = f"lattpath_noTime/Stage1.1/sde_TSMloss_r4e319_euler_step100/testlattpath_stage1.1/step1-1.1"

# run_tag_stage_1 = 3
# ckpt_tag_stage_1 = 483
# inference_steps_stage_1 = 100
# bk_tag_stage_1 = 0
# stage_tag_1 = 2.1
# suffix_stage_1 = "_gtcubic"
# subfolder = "lattpath_noTime/Stage2.1/lattpath_noTime/Stage1.1/lattpath_noTime/bk.1.Stage1/test_stage1.bk01_r4e459_step100/test_stage1.1.bk00_r1e365_step100/"


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


# inference_out_dir = f"experiments/MOF/{stage_subdirs[stage_tag]}/{subfolder}/test_stage{stage_tag_1}.bk{bk_tag_stage_1:02d}_r{run_tag_stage_1}e{ckpt_tag_stage_1}_step{inference_steps_stage_1}{suffix_stage_1}/sde_TSMloss_r{run_tag}e{ckpt_tag}_{sampling_method}_step{inference_steps}{suffix}/"
inference_out_dir = f"experiments/MOF/lattpath_noTime/Stage1.1/sde_TSMloss_r4e319_euler_step100/testlattpath_stage1.1/step2-2.1//{suffix}"

ckpt = torch.load(sim_ckpt, map_location="cpu", weights_only=False)
hparams = ckpt["hyper_parameters"]
args = hparams['args']
args.sampling_method = sampling_method
args.inference_steps = inference_steps

args.traj = []
for d in data_dir:
    args.traj.extend(ase.io.read(d, index=":", format="extxyz")[100::101])

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
# args.target_std = 0.0
args.likelihood = None
args.K_hutchinson_probe = 1
args.K_hutchinson_probe_chunk = 1

args.guidance = guidance

from ase import Atoms
from mdgen.spring import ZBLRepulsiveWall

MACE_MODEL_PATH = "/home/tuoping/odefed_mdgen/workdir_odefed_mdgen/data/MOF/CoRE_MOF/mace-mh-1.model"

if args.guidance:
    wall = ZBLRepulsiveWall()
    # from mace.calculators import mace_mp
    # mace_calculator = mace_mp(
    #     model=MACE_MODEL_PATH,
    #     head="omat_pbe",
    #     device=str(device),
    #     default_dtype="float32",
    #     dispersion=True,
    #     dispersion_xc="pbe",
    # )


def _guidance(_x, t, **kwargs):
    # Return the MACE + ZBL Cartesian force in the fractional-coordinate
    # convention used by the transport score.
    if torch.any(t > 0.75):
        with torch.inference_mode(False):
            x = _x.detach().clone()
            cell = kwargs['cell'].detach().clone()
            B, T, N, _ = x.shape

            labels = torch.argmax(kwargs['aatype'][..., :len(species)], dim=-1)
            species_tensor = torch.as_tensor(species, dtype=torch.long, device=x.device)
            atomic_numbers = species_tensor[labels]

            flat_x = x.reshape(B * T, N, 3)
            flat_cell = cell.reshape(B * T, 3, 3)
            flat_atomic_numbers = atomic_numbers.reshape(B * T, N)

            wall_force_cart = wall.build_force(
                flat_x,
                flat_cell,
                flat_atomic_numbers,
            ).reshape(B, T, N, 3)

            # mace_force_cart = torch.empty_like(x)
            # for i_batch in range(B):
            #     for i_frame in range(T):
            #         atoms = Atoms(
            #             numbers=atomic_numbers[i_batch, i_frame].detach().cpu().numpy(),
            #             scaled_positions=x[i_batch, i_frame].detach().cpu().numpy(),
            #             cell=cell[i_batch, i_frame].detach().cpu().numpy(),
            #             pbc=True,
            #         )
            #         atoms.wrap()
            #         atoms.calc = mace_calculator
            #         mace_force_cart[i_batch, i_frame] = torch.as_tensor(
            #             atoms.get_forces(),
            #             dtype=x.dtype,
            #             device=x.device,
            #         )

            # total_force_frac = mace_force_cart @ torch.linalg.inv(cell) + wall_force_cart @ torch.linalg.inv(cell) * WALL_FORCE_SCALE
            scale = t*4-3
            total_force_frac = wall_force_cart @ torch.linalg.inv(cell) * WALL_FORCE_SCALE * scale[..., None, None, None]**2
        return total_force_frac.detach()
    else:
        return torch.zeros_like(_x)


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
    if Sanity_check:
        num_rollout = 1
        num_samples = 10
    else:
        num_rollout = len(dataset)
        num_samples = 1
    for i_rollout in range(num_rollout):
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

        for i_sample in range(num_samples):
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
