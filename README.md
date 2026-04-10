# MOLGEN

Implementation of [Flow matching for reaction pathway generation](https://arxiv.org/abs/2507.10530) by Ping Tuo*, Jiale Chen, Ju Li*.

## Installation

```
pip install numpy==1.26.0 pandas==1.5.3 scikit-learn==1.6.1
pip install torch==2.6.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch_lightning==2.0.4 mdtraj==1.9.9 biopython==1.79
pip install wandb dm-tree einops torchdiffeq fair-esm pyEMMA
pip install matplotlib==3.7.2
pip install omegaconf==2.3.0
pip install ase==3.22 pymatgen
# before installing torch_scatter, make sure the libstdc++.so.6 include GLIBCXX_3.4.32 by `strings .../libstdc++.so.6.0.33 | grep GLIB`
TORCH=2.6.0 
CUDA=cu124
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric  -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

## Datasets

I have written a dataset class `mdgen/dataset.py`: `EquivariantTransformerDataset_MaterialProject` for periodic systems.
One can read a xyz file and write to torch dataset by 

```python
import torch, os
from mdgen.dataset import EquivariantTransformerDataset_MaterialProject

torch.set_float32_matmul_precision('medium')

idx_test = [32, 34, 2]
idx_train = [x for x in range(64) if x not in idx_test]
trainset = EquivariantTransformerDataset_MaterialProject("data/MP_C_sims", 6, species=[6], localmask=False, sim_condition=False, stage="save", save_dir="data/MP_C_data", save_filename="train", sel_idx=idx_train)
testset = EquivariantTransformerDataset_MaterialProject("data/MP_C_sims", 6, species=[6], localmask=False, sim_condition=False, stage="save", save_dir="data/MP_C_data", save_filename="test", sel_idx=idx_test)
```


## Training

**Change the latt_path parameter in `equivariant_wrapper.py` to turn on/off variable cell**

Training command for a periodic system:
```

python train.py --data_dir data/MP_C_data/ --num_species 1 --epochs 600 --cutoff 12 --path-type Linear --embed_dim 128 --val_epoch_freq 100 --batch_size 1024 --overfit --x0std 0.2

```

## Inference

```
python Latt-inference.py 
```

## Overview of code updates relative to the main branch

- The lattice flow is enabled by
    - The training workflow in `mdgen/transport/transport.py`: `def training_losses`, where we enabled loss function: 
    ```python
    terms['loss_lattflow'] = mean_flat((lowertrigflow_output - lowertrigulatt).abs(), torch.ones_like(lowertrigflow_output, device=lowertrigflow_output.device))
    ```
    - The lattice path is written in `class ICPlan`: `def plan_latt`. (For path-type other than `Linear`, the lattice path is not enabled.)

- Lattice flow prediction is enabled in `mdgen/model/equivariant_latent_model.py`.

- Key insights:
  - For stable prediction, need to add noise to the target structure;
  - The current version is based on a fixed variance through the Gaussian flow matching path; The version with a chaning variance is underway;
  - The variance of the fractional coordinates is divided by $N^{1/3}$, where N is the number of atoms, to ensure that the variance of the cartisian coordinates is $\approx$ args.x0std.

Note:

- In `dataset.py:EquivariantTransformerDataset_MaterialProject`, the current version only reads one structure and adds random noise to it to make 1024 different data samples.
- The current default version of lattice flow is Riemann flow, but euclidean flow is also implemented, and can be switched by changing from calling `self.path_sampler.path_latt_riemann` to `self.path_sampler.path_latt` in `transport/transport.py: def training_losses`.

## License

MIT. Additional licenses may apply for third-party source code noted in file headers.

## Citation
```
@misc{tuo2025accurate,
  title        = {Flow matching for reaction pathway generation},
  author       = {Tuo, Ping and Che n, Jiale and Li, Ju},
  year         = {2025},
  eprint       = {2507.10530},
  archivePrefix= {arXiv},
  primaryClass = {physics.chem-ph},
  doi          = {10.48550/arXiv.2507.10530},
  url          = {https://arxiv.org/abs/2507.10530}
}
```

## Acknowledgements

Code developed based on

[Generative Modeling of Molecular Dynamics Trajectories](https://github.com/bjing2016/mdgen)
