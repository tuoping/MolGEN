K_to_eV = 8.617333262E-5
temperature_K = 1600
T_in_eV = K_to_eV*temperature_K

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

batch_size = 3
nsamples = 490 // batch_size
num_atoms = 1125

def ode_fed(all_idx_sample):
    all_logp_forward = []
    all_logp_backward = []
    all_ex = []
    all_Uzs = []
    for i in range(len(all_idx_sample)):
        idx_sample = all_idx_sample[i]
        logp_forward = np.loadtxt(f"./data/Logp_{idx_sample}.txt")
        logp_backward = np.loadtxt(f"./data/reverse_Logp_{idx_sample}.txt")
        ex = np.loadtxt(f"./data/all_energy_atoms_{idx_sample}.dat")
        Uzs = np.loadtxt(f"./data/Uzs_{idx_sample}.txt")
        all_logp_forward.append(logp_forward)
        all_logp_backward.append(logp_backward)
        all_ex.append(ex)
        all_Uzs.append(Uzs)

    all_logp_forward = np.array(all_logp_forward)
    all_logp_backward = np.array(all_logp_backward)
    all_ex = np.array(all_ex)
    all_Uzs = np.array(all_Uzs)

    
    _Ediff_forward = []
    work_forward = []

    _Ediff_backward = []
    work_backward = []
    for i in range(len(all_ex)):
        U_b = all_ex[i]/T_in_eV
        U_a = all_Uzs[i][0]
        _Ediff_forward.append(U_b - U_a  + (-U_a+all_Uzs[i,1]))
        work_forward.append(U_b - U_a  + (-U_a+all_Uzs[i,1]) - all_logp_forward[i])

        _Ediff_backward.append(all_Uzs[i][1]-U_b  + (all_Uzs[i,1]-all_Uzs[i,0]))  
        work_backward.append(all_Uzs[i][1]-U_b  + (all_Uzs[i,1]-all_Uzs[i,0]) - all_logp_backward[i])

    _Ediff_backward = np.array(_Ediff_backward)
    _Ediff_forward = np.array(_Ediff_forward)
    work_forward = np.array(work_forward)
    work_backward = np.array(work_backward)

    ELBO = (logsumexp(work_backward) - np.log(len(work_backward)))
    EUBO = -(logsumexp(-work_forward) - np.log(len(work_forward)))
    C = (-ELBO + EUBO)/2

    F0_forward_ens = -logsumexp(-all_Uzs[:,0])
    return C, F0_forward_ens


F_all_FED = []
Fz_all = []
for i in range(batch_size):
    all_idx_sample = np.arange(i*nsamples, (i+1)*nsamples)
    dF_convergence, Fz = ode_fed(all_idx_sample)
    F_all_FED.append(dF_convergence+Fz)
    Fz_all.append(Fz)



with open("F_FED.out", "w") as fout:
    fout.write("F  var_dF    F(eV)  var_dF(eV)\n")
    fout.write("%f  %f    %f  %f\n"%(np.mean(F_all_FED)/num_atoms, 1.644853626951*np.std(F_all_FED)/np.sqrt(batch_size)/num_atoms,  
                                             (np.mean(F_all_FED)/num_atoms)*T_in_eV, 1.644853626951*np.std(F_all_FED)/np.sqrt(batch_size)/num_atoms*T_in_eV,  ))
