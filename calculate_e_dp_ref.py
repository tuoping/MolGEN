
from deepmd.calculator import DP
import numpy as np


import ase, ase.io

MODEL_PATH = "./DP_R2SCAN.pb"


def make_dp_calculator(model_path):
    calculator = DP(model=model_path)
    model_type_map = calculator.dp.get_type_map()
    calculator.type_dict = {
        element: idx for idx, element in enumerate(model_type_map)
    }
    print(f"Using DP type map: {calculator.type_dict}")
    return calculator


def validate_elements(atoms, type_dict, traj_path):
    missing = sorted(set(atoms.get_chemical_symbols()) - set(type_dict))
    if missing:
        raise ValueError(
            f"{traj_path} contains elements not present in {MODEL_PATH}: {missing}. "
            f"Model type map is {type_dict}."
        )


calculator = make_dp_calculator(MODEL_PATH)

import matplotlib.pyplot as plt
import numpy as np

import sys
num_trials = int(sys.argv[1])

import time
import os

dirname = f'./'
for i_trial in range(0, num_trials):
    s_time = time.time()
    if os.path.exists(f"{dirname}/ref_energy_atoms_{i_trial}.dat") and isinstance(np.loadtxt(f"{dirname}/ref_energy_atoms_{i_trial}.dat"), int):
        print(f"WARNING:: skipping {i_trial}", np.loadtxt(f"{dirname}/ref_energy_atoms_{i_trial}.dat"))
        continue
    
    traj = ase.io.read(f'{dirname}/reftraj_{i_trial}.xyz', format='extxyz', index=":")
    ofile_e = open(f"{dirname}/ref_energy_atoms_{i_trial}.dat", "w")
    atoms = traj[-1]
    atoms.wrap()
    if os.path.exists(f"wrapped_ref_{i_trial}.xyz"): os.remove(f"wrapped_ref_{i_trial}.xyz")
    ase.io.write(f"wrapped_ref_{i_trial}.xyz", atoms)
    validate_elements(atoms, calculator.type_dict, f"{dirname}/reftraj_{i_trial}.xyz")
    atoms.calc = calculator
    energy_atoms = atoms.get_potential_energy()
           
    ofile_e.write(f"{energy_atoms}\n")
    ofile_e.flush()
    ofile_e.close()
    e_time = time.time()
    print(f"Rollout {i_trial} done, time: {e_time - s_time:.2f} s")
    
