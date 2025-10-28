from argparse import Namespace
import sys
sim_ckpt = (sys.argv[1])
out_dir = (sys.argv[2])
args = Namespace(
    # sim_ckpt="workdir/sc-oa-TPS-x0std1.0-OT/l1/epoch=224-step=0035775-val_meanRMSD_Kabsch=0.1493.ckpt",
    # sim_ckpt="workdir/sc-oa-TPS-x0std1.0-OT/alpha/pref0.1/epoch=709-step=0112890-val_meanRMSD_Kabsch=0.1353.ckpt",
    # sim_ckpt="workdir/sc-oa-TPS-x0std1.0-OT/reversekl+l1/pref_reversekl0.2/epoch=589-step=0093810-val_meanRMSD_Kabsch=0.1290.ckpt",
    # sim_ckpt="workdir/sc-oa-TPS-x0std1.0-OT-alpha/epoch=1159-step=0184440-val_meanRMSD_Kabsch=0.1518.ckpt",
    # sim_ckpt="workdir/pretrain-diffusion/epoch=809-step=0128790-val_meanRMSD_Kabsch=0.1554.ckpt",
    # sim_ckpt="workdir/Dataset2-sc-oa-TPS-x0std1.0-OT-new/epoch=534-step=1555780-val_meanRMSD_Kabsch=0.1286.ckpt",
    # sim_ckpt="workdir/Dataset2-sc-oa-TPS-x0std1.0-OT-new/tune-symmkl/epoch=254-step=0741540-val_meanRMSD_Kabsch=0.1357.ckpt",
    # sim_ckpt='workdir/Dataset2-sc-oa-TPS-x0std1.0-OT-symmonly/epoch=159-step=0465280-val_meanRMSD_Kabsch=0.1609.ckpt',
    # sim_ckpt='workdir/Dataset2-sc-oa-TPS-x0std1.0-OT-symm/epoch=649-step=1890200-val_meanRMSD_Kabsch=0.1420.ckpt',
    sim_ckpt=sim_ckpt,
    # data_dir="./",
    data_dir="data/Transition1x/",
    suffix="",
    out_dir=f"experiments/{out_dir}",
    num_frames=3,
    localmask=False,
    tps_condition=True,
    sim_condition=False
    )
import glob
args.sim_ckpt = glob.glob(args.sim_ckpt)[0]
device = "cuda"

import os, torch, tqdm, time
import numpy as np
from mdgen.equivariant_wrapper import EquivariantMDGenWrapper

os.makedirs(args.out_dir, exist_ok=True)
with open(f"{args.out_dir}/README.md", "w") as fp:
    fp.write(args.sim_ckpt)

from mdgen.dataset import EquivariantTransformerDataset_Transition1x
# dataset = EquivariantTransformerDataset_Transition1x(data_dirname=args.data_dir, sim_condition=args.sim_condition, tps_condition=args.tps_condition, num_species=5, stage=out_dir)
dataset = EquivariantTransformerDataset_Transition1x(data_dirname=args.data_dir, sim_condition=args.sim_condition, tps_condition=args.tps_condition, num_species=5, stage="test-fragmented_cutoffx1.5")

ckpt = torch.load(args.sim_ckpt, weights_only=False)
hparams = ckpt["hyper_parameters"]
hparams['args'].guided = False
# hparams['args'].sampling_method = 'euler'
# hparams['args'].guidance_pref = 2
hparams['args'].inference_steps = 50
model = EquivariantMDGenWrapper(**hparams)
print(model.model)
model.load_state_dict(ckpt["state_dict"], strict=False)
model.eval().to(device)

print(ckpt["hyper_parameters"])
print(len(dataset))

print(ckpt["hyper_parameters"]['args'].path_type)
print(ckpt["hyper_parameters"]['args'].x0std)
print(ckpt["hyper_parameters"]['args'].sampling_method)

batch_size = 1
val_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=0,
    shuffle=True,
)
sample_batch = next(iter(val_loader))

print(sample_batch.keys())
print(dataset[499]["x"].shape)

for key in ['species', 'x', 'cell', 'num_atoms', 'mask', 'v_mask', "TKS_mask", "TKS_v_mask", "fragments_idx"]:
    try:
        sample_batch[key] = sample_batch[key].to(device)
    except:
        print(f"{key} not found")


pred_pos = model.inference(sample_batch)

prep = model.prep_batch(sample_batch)
print(prep['model_kwargs']['v_mask'])

@torch.no_grad()
def rollout(model, batch):
    expanded_batch = batch
    
    positions, _ = model.inference(expanded_batch)

    new_batch = {**batch}
    new_batch['x'] = positions
    return positions, new_batch


map_to_chemical_symbol = {
    0: "H",
    1: 'C',
    2: "N",
    3: "O"

}

if 'Ensemble' in args.out_dir:
    idx_rollouts = np.ones(100).astype(int)*i_ens
else:
    idx_rollouts = np.arange(len(dataset))
from ase import Atoms
from ase.geometry.geometry import get_distances
import shutil, os
from ase.io import write

all_rollout_atoms_ref_0 = []
all_rollout_atoms = []
all_rollout_atoms_ref = []
import time

for i_rollout in range(0, len(idx_rollouts)):
    idx = idx_rollouts[i_rollout]
    print("idx = ", idx, "rollout", i_rollout, out_dir)
    for i_trial in range(30):
    # for i_trial in [1]:
        start = time.time()
        item = dataset.__getitem__(idx)
        batch = next(iter(torch.utils.data.DataLoader([item])))

        for key in ['species', 'x', 'cell', 'num_atoms', 'mask', 'v_mask', "TKS_mask", "TKS_v_mask", "fragments_idx"]:
            try:
                batch[key] = batch[key].to(device)
            except:
                print(f"{key} not found")

        labels = torch.argmax(batch["species"], dim=3).squeeze(0)
        symbols = [[map_to_chemical_symbol[int(i_elem.to('cpu'))] for i_elem in labels[i_conf]] for i_conf in range(len(labels))]

        try:
            pred_pos, _ = rollout(model, batch)
        except:
            print(f"WARNING:: {i_rollout} of data {idx} failed")
            continue
        print("Time::", time.time()-start)        
        all_atoms = []
        all_atoms_ref = []
        for t in range(len(pred_pos[0])):
            print("rollout", i_rollout, "idx = ", idx, "t", t)
            formula = "".join(symbols[t])

            atoms = Atoms(formula, positions=pred_pos[0][t].cpu().numpy(), cell=batch['cell'][0][0].cpu().numpy(), pbc=[1,1,1])
            # atoms.set_chemical_symbols(symbols[t])
            all_atoms.append(atoms)
            atoms_ref = Atoms(formula, positions=batch["x"][0][t].cpu().numpy(), cell=batch['cell'][0][0].cpu().numpy(), pbc=[1,1,1])
            all_atoms_ref.append(atoms_ref)
        # all_rollout_atoms.append(all_atoms)
        # all_rollout_atoms_ref.append(all_atoms_ref)
        out_dir = args.out_dir
        dirname = os.path.join(out_dir, f"rollout_{i_rollout}")
        # if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

        with open(os.path.join(dirname, "README.md"), "w") as fp:
            fp.write("Data index from Transition1x: %d"%idx)
        np.savetxt(os.path.join(dirname, "Fragment_idx.dat"), batch['fragments_idx'].squeeze(0).cpu().numpy())
        filename = os.path.join(dirname, f"gentraj_{i_trial}.xyz")
        filename_ref = os.path.join(dirname, "reftraj_1.xyz")
        if os.path.exists(filename):
        #     shutil.move(filename_0, os.path.join(dirname, "bck.0.gentraj_0.xyz"))
            os.remove(filename)
        #     shutil.move(filename_ref_0, os.path.join(dirname, "bck.0.reftraj_0.xyz"))
            os.remove(filename_ref)
        for atoms in all_atoms:
            atoms.set_cell(np.eye(3,3)*25)
            write(filename, atoms, append=True)
        if not os.path.exists(filename_ref):
            for ref_atoms in all_atoms_ref:
                ref_atoms.set_cell(np.eye(3,3)*25)
                write(filename_ref, ref_atoms, append=True)
        
        if model.args.tps_condition:
            assert np.allclose(all_atoms[2].positions, all_atoms_ref[2].positions)
            assert np.allclose(all_atoms[0].positions, all_atoms_ref[0].positions)
            assert not np.allclose(all_atoms[1].positions, all_atoms_ref[1].positions)
        elif model.args.sim_condition:
            assert not np.allclose(all_atoms[2].positions, all_atoms_ref[2].positions)
            assert np.allclose(all_atoms[0].positions, all_atoms_ref[0].positions)
            assert np.allclose(all_atoms[1].positions, all_atoms_ref[1].positions)

