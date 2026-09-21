K_to_eV = 8.617333262E-5
temperature_K = 1600
T_in_eV = K_to_eV*temperature_K

import numpy as np
import matplotlib.pyplot as plt
import ase.io

all_idx_sample = np.arange(10000)
num_atoms = 864
crystal_type = "coesite"

import os
import sys
all_logp_forward = []
all_logp_backward = []
all_ex = []
all_Uzs = []
all_volumes = []
for i in range(0, len(all_idx_sample)):
    idx_sample = all_idx_sample[i]
    if not os.path.exists(f"./data/gentraj_{idx_sample}.xyz"):
        continue
    logp_forward = np.loadtxt(f"./data/Logp_{idx_sample}.txt")
    # logp_backward = np.loadtxt(f"./data/reverse_Logp_{idx_sample}.txt")
    ex = np.loadtxt(f"./data/all_energy_atoms_{idx_sample}.dat")
    Uzs = np.asarray(np.loadtxt(f"./data/Uzs_{idx_sample}.txt")).reshape(-1)[0]
    all_logp_forward.append(logp_forward)
    # all_logp_backward.append(logp_backward)
    all_ex.append(ex)
    all_Uzs.append(Uzs)

    # atoms = ase.io.read(f"./data/gentraj_{i}.xyz", ":")[-1]
    # all_volumes.append(atoms.get_volume())
    if len(all_ex) == int(sys.argv[1]):
        break

all_logp_forward = np.array(all_logp_forward)
all_logp_backward = np.array(all_logp_backward)
all_ex = np.array(all_ex)
all_Uzs = np.array(all_Uzs)
all_volumes = np.array(all_volumes)


_Ediff_forward = []
work_forward = []

_Ediff_backward = []
work_backward = []
for i in range(len(all_ex)):
    U_b = all_ex[i]/T_in_eV
    # U_a = all_Uzs[i]
    logq0 = all_logp_forward[i,2]
    _Ediff_forward.append(U_b + logq0)
    work_forward.append(U_b + logq0 - (all_logp_forward[i,0] - all_logp_forward[i,1]/2))

    # _Ediff_backward.append(all_Uzs[i][1]-U_b)  
    # work_backward.append(all_Uzs[i][1]-U_b + (all_logp_backward[i,0] + all_logp_backward[i,1]/2))

_Ediff_backward = np.array(_Ediff_backward)
_Ediff_forward = np.array(_Ediff_forward)
work_forward = np.array(work_forward)
work_backward = np.array(work_backward)


from scipy.special import logsumexp
# ELBO = (logsumexp(work_backward) - np.log(len(work_backward)))
EUBO = -(logsumexp(-work_forward) - np.log(len(work_forward)))


print((EUBO)*T_in_eV/num_atoms)
