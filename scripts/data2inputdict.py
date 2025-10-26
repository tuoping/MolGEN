import torch
from mdgen.dataset import EquivariantTransformerDataset_Transition1x
data_dir = "data/Transition1x"
stage="test"
dataset = EquivariantTransformerDataset_Transition1x(data_dirname=data_dir, sim_condition=False, tps_condition=True, num_species=5, stage=stage)

tps_masked_dataset = []
for i in range(len(dataset)):
    tps_masked_dataset.append(dataset[i])

torch.save(tps_masked_dataset, f"{data_dir}/tps_masked_{stage}.pt")

