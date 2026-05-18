# from mdgen.parsing import parse_train_args
# args = parse_train_args()

import glob

ckpt_tag = "889"
inference_steps = 100

sampling_method = "euler"
sim_ckpt = glob.glob(f"workdir/default/epoch={ckpt_tag}-step=0*.ckpt")[0]

device = "cuda"

import os, torch, tqdm, time
import numpy as np
from mdgen.fed_wrapper import EquivariantFEDWrapper

out_dir = f"experiments/SiO2_coesite_npt_nowrap/e{ckpt_tag}_{sampling_method}_step{inference_steps}/"

os.makedirs(out_dir, exist_ok=True)
with open(f"{out_dir}/README.md", "w") as fp:
    fp.write(sim_ckpt)


torch.set_float32_matmul_precision('medium')

ckpt = torch.load(sim_ckpt, weights_only=False)
hparams = ckpt["hyper_parameters"]
args = hparams['args']
args.sampling_method = sampling_method
args.inference_steps = inference_steps
args.data_dir = "data/SiO2/npt_1600K_1GPa/npt_coesite_dense/nvt/"
args.likelihood = "EJE"


from mdgen.dataset import EquivariantTransformerDataset_phasediagram
dataset = EquivariantTransformerDataset_phasediagram(args, species=[14, 8], sim_condition=False, stage="test")



model = EquivariantFEDWrapper(**hparams)
print(model.model)
model.load_state_dict(ckpt["state_dict"], strict=True)
model.eval().to(device)

print(model.args)
print(model.args.path_type)
print(model.args.sampling_method)
print(model.args.inference_steps)
print(model.args.likelihood)
print(model.args.x0std)

batch_size = 1
val_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=0,
    shuffle=True,
)
sample_batch = next(iter(val_loader))

@torch.no_grad()
def rollout(model, batch):
    logp, positions, _ = model.inference(batch)
    new_batch = {**batch}
    new_batch['x'] = positions
    return logp, positions, new_batch


map_to_chemical_symbol = {
    0: "O",
    1: "Si"
}

idx_rollouts = np.arange(len(dataset))

from ase import Atoms
from ase.geometry.geometry import get_distances
import shutil, os
from ase.io import write

all_rollout_atoms_ref_0 = []
all_rollout_atoms = []
all_rollout_atoms_ref = []
start = time.time()
all_logp = []
for i_rollout in range(429, len(idx_rollouts)):
    if i_rollout % 8 == 0:
        continue
    idx = idx_rollouts[i_rollout]
    print(i_rollout, idx)
    filename = os.path.join(out_dir, f"gentraj_{idx}.xyz")

    if args.likelihood is not None:
        filename_reverse = os.path.join(out_dir, f"reverse_gentraj_{idx}.xyz")
        filename_logp = os.path.join(out_dir, f"Logp_{idx}.txt")
        fout_logp = open(filename_logp, "a")
        filename_reverse_logp = os.path.join(out_dir, f"reverse_Logp_{idx}.txt")
        fout_reverse_logp = open(filename_reverse_logp, "a")
        filename_zs = os.path.join(out_dir, f"Uzs_{idx}.txt")
        fout_zs = open(filename_zs, "a")

    filename_ref = os.path.join(out_dir, f"reftraj_{idx}.xyz")

    for f in [filename, filename_ref, filename_reverse, filename_logp, filename_reverse_logp, filename_zs]:
        if os.path.exists(f):
            os.remove(f)

    for i_sample in range(1):
        item = dataset.__getitem__(idx)
        batch = next(iter(torch.utils.data.DataLoader([item])))

        for key in batch.keys():
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        labels = torch.argmax(batch["species"], dim=3).squeeze(0)
        symbols = [[map_to_chemical_symbol[int(i_elem.to('cpu'))] for i_elem in labels[i_conf]] for i_conf in range(len(labels))]

        print("rollout", i_rollout, "idx = ", idx+i_sample)
        formula = "".join(symbols[0])
        if model.transport.latt_path:
            if args.likelihood is None:
                all_pred_frac_pos, _, all_cell  = model.inference(batch)
            else:
                logp, all_pred, _, reverse_logp, zs, all_pred_reverse = model.inference(batch)
                all_pred_frac_pos = all_pred[0]
                all_cell = all_pred[1]
                all_pred_zs = all_pred_reverse[0]
                pred_zs = all_pred_zs[-1]
                pred_zs_cell = all_pred_reverse[1][-1]
                np.savetxt(fout_logp, logp.detach().cpu().numpy() )
                fout_logp.flush()
                np.savetxt(fout_reverse_logp, reverse_logp.detach().cpu().numpy() )
                fout_reverse_logp.flush()
                N = all_pred_frac_pos.shape[-2]
                sigma = (torch.ones_like(zs) * args.x0std/(N**(1./3.)))@pred_zs_cell
                np.savetxt(fout_zs, [[( (zs@all_cell[0])**2/2/sigma**2).sum().detach().cpu().numpy(), ( (pred_zs@pred_zs_cell)**2/2/sigma**2).sum().detach().cpu().numpy()]])
                fout_zs.flush()
        else:
            if args.likelihood is None:
                all_pred_frac_pos, _  = model.inference(batch)
            else:
                logp, all_pred_frac_pos, _, reverse_logp, zs, all_pred_zs = model.inference(batch)
                pred_zs = all_pred_zs[-1]
                cell = batch['cell']
                np.savetxt(fout_logp, logp.detach().cpu().numpy() )
                fout_logp.flush()
                np.savetxt(fout_reverse_logp, reverse_logp.detach().cpu().numpy() )
                fout_reverse_logp.flush()
                N = all_pred_frac_pos.shape[-2]
                sigma = (torch.ones_like(zs) * args.x0std/(N**(1./3.)))@cell
                np.savetxt(fout_zs, [[( (zs@cell)**2/2/sigma**2).sum().detach().cpu().numpy(), ( (pred_zs@cell)**2/2/sigma**2).sum().detach().cpu().numpy()]])
                fout_zs.flush()
        if i_rollout == 0:
            dump_idx = range(len(all_pred_frac_pos))
        else:
            dump_idx = [-1]
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

        if args.likelihood is not None:
            for idx_traj in dump_idx:
            # for idx_traj in [-1]:
                pred_frac_pos = all_pred_zs[idx_traj][0]
                if model.transport.latt_path:
                    cell_out = all_pred_reverse[1][idx_traj]
                    pred_pos = pred_frac_pos[0] @ cell_out[0][0]
                else:
                    cell_out = batch['cell']
                    pred_pos = pred_frac_pos[0] @ cell_out[0][0]

                atoms = Atoms(formula, positions=pred_pos.detach().cpu().numpy(), cell=cell_out[0][0].detach().cpu().numpy(), pbc=[1,1,1])
                write(filename_reverse, atoms, append=True)

        
        # del pred_frac_pos
        # del pred_pos
        # del cell_out
        # del ref_pos
