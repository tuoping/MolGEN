# from mdgen.parsing import parse_train_args
# args = parse_train_args()

import glob
ckpt_tag = "7159"
inference_steps = 50
sampling_method = "euler"
sim_ckpt = glob.glob(f"workdir/latinhypecubeprior/epoch={ckpt_tag}-step=0*.ckpt")[0]
device = "cuda"

import os, torch, tqdm, time
import numpy as np
from mdgen.equivariant_wrapper import EquivariantMDGenWrapper

out_dir = f"experiments/latinhypecubeprior_nnoise0.02/MP_C_N32_fracpos/e{ckpt_tag}_{sampling_method}_step{inference_steps}/"
os.makedirs(out_dir, exist_ok=True)
with open(f"{out_dir}/README.md", "w") as fp:
    fp.write(sim_ckpt)


torch.set_float32_matmul_precision('medium')

ckpt = torch.load(sim_ckpt, weights_only=False)
hparams = ckpt["hyper_parameters"]
args = hparams['args']
args.sampling_method = sampling_method
args.inference_steps = inference_steps
args.data_dir = "data/MP_C_data/"
# args.likelihood = "FND"


from mdgen.dataset import EquivariantTransformerDataset_MaterialProject
dataset = EquivariantTransformerDataset_MaterialProject(args, species=[6], sim_condition=False, stage="train_withforces")


model = EquivariantMDGenWrapper(**hparams)
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
    0: "C",
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
for i_rollout in range(0, len(idx_rollouts), 8):
    idx = idx_rollouts[i_rollout]
    print(i_rollout, idx)
    filename = os.path.join(out_dir, f"gentraj_{idx}.xyz")
    # _filename = os.path.join(out_dir, f"_gentraj_{idx}.xyz")
    # _frac_filename = os.path.join(out_dir, f"_frac_gentraj_{idx}.xyz")
    filename_ref = os.path.join(out_dir, f"reftraj_{idx}.xyz")
    for f in [filename, filename_ref]:
        if os.path.exists(f):
            os.remove(f)

    for i_sample in range(1):
        item = dataset.__getitem__(idx)
        batch = next(iter(torch.utils.data.DataLoader([item])))

        for key in ['species', 'x', 'cell', 'num_atoms', 'mask', 'v_mask']:
            try:
                batch[key] = batch[key].to(device)
            except:
                print(f"{key} not found")
        labels = torch.argmax(batch["species"], dim=3).squeeze(0)
        symbols = [[map_to_chemical_symbol[int(i_elem.to('cpu'))] for i_elem in labels[i_conf]] for i_conf in range(len(labels))]
        t = 0
        print("rollout", i_rollout, "idx = ", idx+i_sample, "t", t)
        formula = "".join(symbols[t])
        with torch.no_grad():
            if model.transport.latt_path:
                all_pred_frac_pos, _, all_cell_out  = model.inference(batch)
            else:
                all_pred_frac_pos, _  = model.inference(batch)
        for idx_traj in range(len(all_pred_frac_pos)):
        # for idx_traj in [-1]:
            pred_frac_pos = all_pred_frac_pos[idx_traj][0]
            if model.transport.latt_path:
                cell_out = all_cell_out[idx_traj]
                pred_pos = pred_frac_pos[0] @ cell_out[0][0]
                # atoms = Atoms(formula, scaled_positions=pred_frac_pos[0].detach().cpu().numpy(), cell=cell_out[0][0].detach().cpu().numpy(), pbc=[1,1,1])
                # write(_filename, atoms, append=True)
                # atoms = Atoms(formula, positions=pred_frac_pos[0].detach().cpu().numpy(), cell=cell_out[0][0].detach().cpu().numpy(), pbc=[1,1,1])
                # write(_frac_filename, atoms, append=True)
            else:
                cell_out = batch['cell']
                pred_pos = pred_frac_pos[0] @ cell_out[0][0]
                print(pred_frac_pos.shape, cell_out.shape)
                # atoms = Atoms(formula, scaled_positions=pred_frac_pos[0].detach().cpu().numpy(), cell=cell_out[0][0].detach().cpu().numpy(), pbc=[1,1,1])
                # write(_filename, atoms, append=True)
                # atoms = Atoms(formula, positions=pred_frac_pos[0].detach().cpu().numpy(), cell=cell_out[0][0].detach().cpu().numpy(), pbc=[1,1,1])
                # write(_frac_filename, atoms, append=True)


            atoms = Atoms(formula, positions=pred_pos.detach().cpu().numpy(), cell=cell_out[0][0].detach().cpu().numpy(), pbc=[1,1,1])
            write(filename, atoms, append=True)


        ref_pos = batch["x"][0][0] @ batch['cell'][0][0]
        atoms_ref = Atoms(formula, positions=ref_pos.cpu().numpy(), cell=batch['cell'][0][0].cpu().numpy(), pbc=[1,1,1])
        write(filename_ref, atoms_ref, append=True)
        
        # del pred_frac_pos
        # del pred_pos
        # del cell_out
        # del ref_pos
