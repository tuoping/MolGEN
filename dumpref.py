# from mdgen.parsing import parse_train_args
# args = parse_train_args()

import glob
import os

# Dynamic neighbor graphs use differently sized CUDA allocations at each SDE
# step. Expandable segments reduce allocator fragmentation for this workload.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

run_tag=4
ckpt_tag = 499
### ODE
# inference_steps = 50
# sampling_method = "rk4"
### SDE
inference_steps = 1000
sampling_method = "euler"

# print("workdir/default/epoch=%03d-step=*-val_loss*.ckpt"%ckpt_tag)
# sim_ckpt = glob.glob(f"workdir/default/run{run_tag}/epoch=%03d-step=*-val_loss*.ckpt"%ckpt_tag)[0]
sim_ckpt = glob.glob(f"workdir/default/bk.3.run{run_tag}/last.ckpt")[0]

import torch, tqdm, time
import numpy as np
from mdgen.equivariant_wrapper import EquivariantMDGenWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

out_dir = f"experiments/MOF/Stage1/allref_train"
print("Output folder: ", out_dir)
os.makedirs(out_dir, exist_ok=True)
with open(f"{out_dir}/README.md", "w") as fp:
    fp.write(sim_ckpt)


torch.set_float32_matmul_precision('medium')

# Keep the checkpoint state dict off the GPU. Loading it directly onto CUDA
# otherwise leaves a second full copy of every parameter alive in `ckpt`.
ckpt = torch.load(sim_ckpt, map_location="cpu", weights_only=False)
hparams = ckpt["hyper_parameters"]
args = hparams['args']
args.sampling_method = sampling_method
args.inference_steps = inference_steps
args.data_dir = "data/MOF/CoRE_MOF/CR/ASR/"
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
                                        stage="train",)



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

@torch.no_grad()
def rollout(model, batch):
    logp, positions, _ = model.inference(batch)
    new_batch = {**batch}
    new_batch['x'] = positions
    return logp, positions, new_batch

from ase.data import chemical_symbols
map_to_chemical_symbol = {}
for i in range(len(species)):
    map_to_chemical_symbol[i] = chemical_symbols[species[i]]

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
for i_rollout in range(0, len(dataset)):
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
        item = dataset.__getitem__(idx)
        batch = next(iter(torch.utils.data.DataLoader([item])))

        for key in batch.keys():
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        x0std = None
        labels = torch.argmax(batch["species"], dim=3).squeeze(0)
        symbols = [[map_to_chemical_symbol[int(i_elem.to('cpu'))] for i_elem in labels[i_conf]] for i_conf in range(len(labels))]

        print("rollout", i_rollout, "idx = ", idx+i_sample)
        formula = "".join(symbols[0])


        ref_pos = batch["x"][0][0] @ batch['cell'][0][0]
        atoms_ref = Atoms(formula, positions=ref_pos.cpu().numpy(), cell=batch['cell'][0][0].cpu().numpy(), pbc=[1,1,1])
        write(filename_ref, atoms_ref, append=True)

        
        # del pred_frac_pos
        # del pred_pos
        # del cell_out
        # del ref_pos
