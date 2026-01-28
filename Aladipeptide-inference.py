import glob
sim_ckpt = glob.glob("workdir/s1/epoch=1529-step*.ckpt")[0]
device = "cuda"

import os, torch, tqdm, time
import numpy as np
from mdgen.equivariant_wrapper import EquivariantMDGenWrapper

out_dir = "experiments/ala-dipeptide/s2/"

os.makedirs(out_dir, exist_ok=True)
with open(f"{out_dir}/README.md", "w") as fp:
    fp.write(sim_ckpt)


ckpt = torch.load(sim_ckpt, weights_only=False)
hparams = ckpt["hyper_parameters"]
args = hparams['args']
args.sampling_method = "euler"
args.inference_steps = 50
# if "forward" in out_dir:
args.likelihood = True


from mdgen.dataset import EquivariantTransformerDataset_Alanine_Dipeptide
dataset = EquivariantTransformerDataset_Alanine_Dipeptide(data_dirname=args.data_dir, sim_condition=args.sim_condition, tps_condition=args.tps_condition, num_species=4, stage="traj_4000steps_test")
print(len(dataset))


model = EquivariantMDGenWrapper(**hparams)
print(model.model)
model.load_state_dict(ckpt["state_dict"], strict=False)
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

@torch.no_grad()
def rollout(model, batch):
    logp, positions, _ = model.inference(batch)
    new_batch = {**batch}
    new_batch['x'] = positions
    return logp, positions, new_batch


map_to_chemical_symbol = {
    0: "H",
    1: 'C',
    2: "N",
    3: "O"

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
for i_rollout in range(len(idx_rollouts)):
    idx = idx_rollouts[i_rollout]
    item = dataset.__getitem__(idx+0%10)
    batch = next(iter(torch.utils.data.DataLoader([item])))
    for key in ['species', 'x', 'cell', 'num_atoms', 'mask', 'v_mask', "cv"]:
        try:
            batch[key] = batch[key].to(device)
        except:
            print(f"{key} not found")
    

    filename = os.path.join(out_dir, f"gentraj_{idx}.xyz")
    filename_ref = os.path.join(out_dir, f"reftraj_{idx}.xyz")
    # if os.path.exists(filename):
    #     os.remove(filename)
    #     os.remove(filename_ref)
    fout_cv = open(os.path.join(out_dir, f"CV_{idx}.txt"), "a")
    fout_logp = open(os.path.join(out_dir, f"Logp_{idx}.txt"), "a")
    fout_reverse_logp = open(os.path.join(out_dir, f"reverse_Logp_{idx}.txt"), "a")
    for i_sample in range(10):
        item = dataset.__getitem__(idx+i_sample%10)
        batch = next(iter(torch.utils.data.DataLoader([item])))

        for key in ['species', 'x', 'cell', 'num_atoms', 'mask', 'v_mask', "cv"]:
            try:
                batch[key] = batch[key].to(device)
            except:
                print(f"{key} not found")
        np.savetxt(fout_cv, batch['cv'].squeeze(0).cpu().numpy())
        fout_cv.flush()

        logp, pred_pos, _, reverse_logp, pred_zs = model.inference(batch)
        np.savetxt(fout_logp, logp.detach().cpu().numpy() )
        fout_logp.flush()
        np.savetxt(fout_reverse_logp, reverse_logp.detach().cpu().numpy() )
        fout_logp.flush()

        labels = torch.argmax(batch["species"], dim=3).squeeze(0)
        symbols = [[map_to_chemical_symbol[int(i_elem.to('cpu'))] for i_elem in labels[i_conf]] for i_conf in range(len(labels))]
        all_atoms = []
        all_atoms_ref = []
        for t in range(len(pred_pos[0])):
            print("rollout", i_rollout, "idx = ", idx+i_sample, "t", t)
            formula = "".join(symbols[t])
            atoms = Atoms(formula, positions=pred_pos[0][t].detach().cpu().numpy(), cell=batch['cell'][0][0].cpu().numpy(), pbc=[1,1,1])
            # atoms.set_chemical_symbols(symbols[t])
            all_atoms.append(atoms)
            atoms_ref = Atoms(formula, positions=batch["x"][0][t].cpu().numpy(), cell=batch['cell'][0][0].cpu().numpy(), pbc=[1,1,1])
            all_atoms_ref.append(atoms_ref)

        for atoms in all_atoms:
            write(filename, atoms, append=True)
        for ref_atoms in all_atoms_ref:
            write(filename_ref, ref_atoms, append=True)
