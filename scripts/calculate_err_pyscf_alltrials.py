import sys
import glob
num_samples = len(glob.glob("rollout_*"))
task_name = sys.argv[1]

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import math
import ase.io
from pymatgen.core import Molecule
from pymatgen.analysis.molecule_matcher import BruteForceOrderMatcher, GeneticOrderMatcher, HungarianOrderMatcher, KabschMatcher
import numpy as np
from pymatgen.io.xyz import XYZ

def xh2pmg(species, xh):
    mol = Molecule(
        species=species,
        coords=xh[:, :3],
    )
    return mol


def xyz2pmg(xyzfile):
    xyz_converter = XYZ(mol=None)
    mol = xyz_converter.from_file(xyzfile).molecule
    return mol


def rmsd_core(mol1, mol2, threshold=0.5, same_order=False):
    _, count = np.unique(mol1.atomic_numbers, return_counts=True)
    if same_order:
        bfm = KabschMatcher(mol1)
        _, rmsd = bfm.fit(mol2)

        # Raw-centered RMSD (translation removed, no rotation)
        A = np.asarray(mol1.cart_coords, dtype=np.float64)
        B = np.asarray(mol2.cart_coords, dtype=np.float64)
        A0 = A - A.mean(0, keepdims=True)
        B0 = B - B.mean(0, keepdims=True)
        rmsd_raw_centered = float(np.sqrt(((A0 - B0) ** 2).sum(axis=1).mean()))
        if rmsd_raw_centered < rmsd:
            print(mol1.species, mol2.species)
            print(mol1.cart_coords, mol2.cart_coords)
            raise RuntimeError

        return rmsd
    total_permutations = 1
    for c in count:
        total_permutations *= math.factorial(c)  # type: ignore
    if total_permutations < 1e4:
        bfm = BruteForceOrderMatcher(mol1)
        _, rmsd = bfm.fit(mol2)
    else:
        bfm = GeneticOrderMatcher(mol1, threshold=threshold)
        pairs = bfm.fit(mol2)
        rmsd = threshold
        for pair in pairs:
            rmsd = min(rmsd, pair[-1])
        if not len(pairs):
            bfm = HungarianOrderMatcher(mol1)
            _, rmsd = bfm.fit(mol2)
    return rmsd


def pymatgen_rmsd(
    species, 
    mol1,
    mol2,
    ignore_chirality: bool = True,
    threshold: float = 0.5,
    same_order: bool = False,
):
    if isinstance(mol1, str):
        mol1 = xyz2pmg(species, mol1)
    if isinstance(mol2, str):
        mol2 = xyz2pmg(species, mol2)
    rmsd = rmsd_core(mol1, mol2, threshold, same_order=same_order)
    if ignore_chirality:
        coords = mol2.cart_coords
        coords[:, -1] = -coords[:, -1]
        mol2_reflect = Molecule(
            species=mol2.species,
            coords=coords,
        )
        rmsd_reflect = rmsd_core(
            mol1, mol2_reflect, threshold, same_order=same_order)
        rmsd = min(rmsd, rmsd_reflect)
    return rmsd


def batch_rmsd_sb(
    species,
    fragments_node,
    pred_xh,
    target_xh,
    threshold: float = 0.5,
    same_order: bool = False,
):

    rmsds = []
    end_ind = np.cumsum(fragments_node)
    start_ind = np.concatenate([np.int64(np.zeros(1)), end_ind[:-1]])
    for start, end in zip(start_ind, end_ind):
        mol1 = xh2pmg(species[start:end], pred_xh[start : end])
        mol2 = xh2pmg(species[start:end], target_xh[start : end])
        rmsd = pymatgen_rmsd(
            species[start:end], 
            mol1,
            mol2,
            ignore_chirality=True,
            threshold=threshold,
            same_order=same_order,
            
        )
        rmsds.append(min(rmsd, 1.0))
    return rmsds


all_positions = None
all_positions_ref = None
all_energy_atoms = []
all_energy_atoms_ref = []
all_atomic_numbers = None
mean_distances = []
mean_distances_read = []
all_idx_min = []
all_atoms = []
for i_dir in range(0, num_samples):
    dirname = f'./rollout_{i_dir}'
    sample_positions = None
    sample_energy_atoms = []
    sample_energy_atoms_ref = []
    sample_atomic_numbers = None
    # sample_mean_distances = []
    sample_mean_distances_read = []
    sample_fragments_node = []
    sample_atoms = []

    atoms_ref = ase.io.read(f'{dirname}/reftraj_1.xyz', format='xyz', index=":")[1]
    atomic_numers = atoms_ref.get_atomic_numbers()
    atoms_ref.set_cell(np.eye(3,3)*25)
    # atoms_ref.calc = calculator
    # energy_atoms_ref = atoms_ref.get_potential_energy()/ len(atomic_numers)
    # energy_atoms_ref = np.loadtxt(f"{dirname}/all_energy_atoms_ref.dat")[1]*len(atoms_ref.get_atomic_numbers())
    for i_trial in range(30):
        dirname = f'./rollout_{i_dir}'
        
        # sample_mean_distances_read.append(np.loadtxt(f'{dirname}/mean_distances_{i_trial}.dat'))
        atoms = ase.io.read(f'{dirname}/gentraj_{i_trial}.xyz', format='xyz', index=":")[1]
        atoms.set_cell(np.eye(3,3)*25)
        assert np.all(atoms_ref.get_atomic_numbers() == atoms.get_atomic_numbers())
        sample_atoms.append(atoms)
        # atoms.calc = calculator
        # energy_atoms = atoms.get_potential_energy()/ len(atoms.get_atomic_numbers())
        # try:
        energy_atoms = np.loadtxt(f"{dirname}/all_energy_atoms_{i_trial}.dat")[1]*len(atoms.get_atomic_numbers())
        # except:
        #     print("Warning:: skipping", i_dir, i_trial)
        #     continue
        sample_energy_atoms.append(energy_atoms)

        fragments_idx = np.loadtxt(f"{dirname}/Fragment_idx.dat")[1]
        fragments_node = np.unique(fragments_idx, return_counts=True)[1]
        sample_fragments_node.append(fragments_node)
        # fragments_node = [len(atoms.get_atomic_numbers())]
        # rmsd = np.mean( batch_rmsd_sb(
        #     atoms.get_chemical_symbols(),
        #     fragments_node,
        #     atoms.get_positions(),
        #     atoms_ref.get_positions(),
        # ))
        # sample_mean_distances.append(rmsd)
        if sample_positions is None:
            sample_positions = list(atoms.get_positions())
        else:
            sample_positions += list(atoms.get_positions())

        del atoms
    idx_min = np.argmin(sample_energy_atoms)
    print(i_dir, idx_min, sample_energy_atoms[idx_min])
    all_idx_min.append([i_dir, idx_min, sample_energy_atoms[idx_min]])
    all_energy_atoms.append(sample_energy_atoms[idx_min])
    all_atoms.append(sample_atoms[idx_min])
    # all_energy_atoms_ref.append(energy_atoms_ref)
    rmsd = np.mean( batch_rmsd_sb(
            atoms_ref.get_chemical_symbols(),
            sample_fragments_node[idx_min],
            sample_atoms[idx_min].get_positions(),
            atoms_ref.get_positions(),
        ))
    mean_distances.append(rmsd)
    # mean_distances_read.append(sample_mean_distances_read[idx_min])
    if all_positions is None:
        all_positions = sample_positions[idx_min*len(atomic_numers):(idx_min+1)*len(atomic_numers)]
        all_atomic_numbers = list(atomic_numers)
    else:
        all_positions += (sample_positions[idx_min*len(atomic_numers):(idx_min+1)*len(atomic_numers)])
        all_atomic_numbers += list(atomic_numers)
    if all_positions_ref is None:
        all_positions_ref = list(atoms_ref.get_positions())
    else:
        all_positions_ref += list(atoms_ref.get_positions())
    del atoms_ref

all_positions = np.array(all_positions)
all_positions_ref = np.array(all_positions_ref)


eV_2_kcalmol = 23.0605

all_conf = np.loadtxt(f"/home/tuoping/odefed_mdgen/TS-GEN-transition_states/edge_e3/TS-GEN-embedv-P/experiments/{task_name}/rollout_0/Selected_prod.txt").astype(int)
print(all_conf)
e_optimized = []
import os
for c in all_conf:
    e = np.loadtxt(f"/home/tuoping/odefed_mdgen/TS-GEN-transition_states/edge_e3/TS-GEN-embedv-P/experiments/{task_name}/rollout_0/traj/optimized_energy_atoms_{c}.dat")
    e_optimized.append(e)

all_conf = np.array(all_conf)
e_optimized = np.array(e_optimized)

from adjacency import Table_generator
import numpy as np
import ase.io
def check_same(r, prod_1):
    # reactant = ase.io.read(f1, ":")[2]
    # prod_1 = ase.io.read(f2, ':')[2]
    adj_mat_r = Table_generator(r.get_chemical_symbols(), r.positions)
    adj_mat_p1 = Table_generator(prod_1.get_chemical_symbols(), prod_1.positions)
    return np.all(adj_mat_r == adj_mat_p1)

import os
def read_allfreq_files(fname):
    lines = open(fname, 'r').readlines()
    for line in lines:
        if "Number of imaginary frequencies:" in line:
            num_imag_freq = int(line.split()[-1])
        if "Frequencies (cm^-1):" in line:
            freqs = [float(x.strip(',')) for x in line.split()[2:]]

    if num_imag_freq == 1:
        return True, freqs
    # elif num_imag_freq == 2 and freqs[1] > -100:
    #     return True, freqs
    else:
        return False, freqs

all_freqs = []
all_localmin = []
all_freqs_optimized = []
all_localmin_optimized = []
idx_optimized = []
for i_dir in range(0, num_samples):
    dirname = f'./rollout_{i_dir}'

    is_localmin, freqs =  read_allfreq_files(f"{dirname}/ts_report_allfreq.txt")
    all_localmin.append(is_localmin)
    all_freqs.append(freqs)

    if os.path.exists(f"{dirname}/optimized_ts_report_allfreq.txt"):
        is_localmin, freqs = read_allfreq_files(f"{dirname}/optimized_ts_report_allfreq.txt")
        all_localmin_optimized.append(is_localmin)
        all_freqs_optimized.append(freqs)
        idx_optimized.append(i_dir)
    else:
        all_localmin_optimized.append(False)
        if os.path.exists(f"{dirname}/ts_optimized.xyz"):
            # raise Exception("Some thing went wrong with freq. calculation", i_dir)
            print("!!!!!!!! Some thing went wrong with freq. calculation", i_dir)

print("Number of successfully optimized =", len(idx_optimized))
print(idx_optimized)

all_idx_min = np.array(all_idx_min)
Eb_final = []
idx_final = []
idx_unopt = []
idx_opt = []
for i in range(len(all_localmin)):
    if all_localmin[i]:
        Eb_final.append((all_idx_min[i,2]+778.6669508446397*12)*eV_2_kcalmol)
        idx_final.append(i)
        idx_unopt.append(i)
    elif all_localmin_optimized[i]:
        e = np.loadtxt(f"rollout_{i}/optimized_energy_atoms.dat")
        if e.size == 0:
            raise Exception("Some thing went wrong with energy evaluation of the optimized structure", i)
        Eb_final.append((e*12+778.6669508446397*12)*eV_2_kcalmol)
        idx_final.append(i)
        idx_opt.append(i)

idx_final = np.array(idx_final)
Eb_final = np.array(Eb_final)
print("Number of TS =", len(idx_final))
print(idx_final)
print("Number of unoptimized TS = ", len(idx_unopt))
print(idx_unopt)
print("Number of optimized TS = ", len(idx_opt))
print(idx_opt)
np.savetxt('TS.txt', idx_final)
np.savetxt("unoptimized_TS.txt", idx_unopt)
np.savetxt("optimized_TS.txt", idx_opt)

with open("runjob-irc_r-optimized.sh", 'w') as fp:
    string_job_list = " ".join([str(i) for i in idx_opt])
    fp.write(f"for i in {string_job_list}; do cd rollout_$i; sbatch ../submit-psi4-irc_r.sh; cd ../; done")

with open("runjob-irc_r-unoptimized.sh", 'w') as fp:
    string_job_list = " ".join([str(i) for i in idx_unopt])
    fp.write(f"for i in {string_job_list}; do cd rollout_$i; sbatch ../submit-psi4-irc_r-gentraj.sh $i; cd ../; done")

with open("runjob-irc_p-optimized.sh", 'w') as fp:
    string_job_list = " ".join([str(i) for i in idx_opt])
    fp.write(f"for i in {string_job_list}; do cd rollout_$i; sbatch ../submit-psi4-irc_p.sh; cd ../; done")

with open("runjob-irc_p-unoptimized.sh", 'w') as fp:
    string_job_list = " ".join([str(i) for i in idx_unopt])
    fp.write(f"for i in {string_job_list}; do cd rollout_$i; sbatch ../submit-psi4-irc_p-gentraj.sh $i; cd ../; done")


import glob
from ase import Atoms
from adjacency import Table_generator
from pathlib import Path
def read_xyz_optimized(path: Path):
    txt = path.read_text().splitlines()
    if len(txt) < 2:
        raise ValueError("XYZ file too short")
    atoms = []
    for line in txt[1:]:
        parts = line.split()
        if len(parts) < 4:
            break
            # raise ValueError(f"Bad XYZ atom line: {line}")
        sym, x, y, z = parts[0], parts[1], parts[2], parts[3]
        atoms.append([float(x), float(y), float(z)])
    # print("Number of atoms = ", len(atoms))
    return atoms

def pick_outfile(outfile_list, key_str):
    for o in outfile_list:
        lines = open(o, 'r').readlines()
        if key_str in "".join(lines):
            return lines
    raise Exception("Didn't find the right file")

def read_irc_finalconf(lines):
    for idx_line in range(len(lines)):
        if 'Final optimized geometry and variables:' in lines[idx_line]:
            r_s = []
            r_c = []
            for i_atom in range(num_atoms):
                parts = lines[idx_line +6 + i_atom].split()
                symbol, x, y, z = parts[0], parts[1], parts[2], parts[3]
                r_s.append(symbol)
                r_c.append([float(x), float(y), float(z)])
            irc_r = Atoms("".join(r_s), positions=r_c)
            return irc_r
    return None

from copy import deepcopy
all_irc_traj = []
print("Reading IRC calculation results::")
for i in idx_final:
    outfile_list = sorted(glob.glob(f"rollout_{i}/slurm-*"))
    
    num_atoms = all_atoms[i].get_number_of_atoms()
    irc_traj = []
    lines = pick_outfile(outfile_list, "IRC forward")
    assert "\'OPT_TYPE\': \'IRC\'" in "".join(lines)
    irc_r = read_irc_finalconf(lines)
    if irc_r is not None:
        irc_traj.append(irc_r)
    lines = pick_outfile(outfile_list, "IRC backward")
    assert "\'OPT_TYPE\': \'IRC\'" in "".join(lines)
    irc_p = read_irc_finalconf(lines)
    if irc_p is not None:
        irc_traj.append(irc_p)

    if len(irc_traj) > 1:
        if i in idx_opt:
            TS = deepcopy(irc_traj[0])
            positions = read_xyz_optimized(Path(f"rollout_{i}/ts_optimized.xyz"))
            TS.set_positions(positions)
        elif i in idx_unopt:
            TS = ase.io.read(f"rollout_{i}/gentraj_{int(all_idx_min[i,1])}.xyz",':')[1]
        else:
            print("Some thing is wrong")
        traj_to_write = [irc_traj[0], TS,  irc_traj[1]]
        ase.io.write(f"rollout_{i}/irc_traj.xyz", traj_to_write)

    all_irc_traj.append(irc_traj)



import os
p_number_passed_IRC = {}
idx_passed_IRC = 1
print("IRC tests::")
firctest = open("IRCtest.txt", 'w')
for i, idx in enumerate(idx_final):
    reactant = ase.io.read(f"rollout_{idx}/reftraj_1.xyz", ':')[0]
    adj_mat_r = Table_generator(reactant.get_chemical_symbols(), reactant.positions)
    product = ase.io.read(f"rollout_{idx}/reftraj_1.xyz", ':')[2]
    adj_mat_p = Table_generator(product.get_chemical_symbols(), product.positions)

    if len(all_irc_traj[i]) == 2:
        prod_1 = all_irc_traj[i][1]
        adj_mat_p1 = Table_generator(prod_1.get_chemical_symbols(), prod_1.positions)
    else:
        adj_mat_p1 = None

    if len(all_irc_traj[i]) > 0:
        react_1 = all_irc_traj[i][0]
        adj_mat_r1 = Table_generator(react_1.get_chemical_symbols(), react_1.positions)
    else:
        adj_mat_r1 = None

    if adj_mat_p1 is not None:
        irc_r_pass = np.all(adj_mat_r == adj_mat_r1) | np.all(adj_mat_r == adj_mat_p1)
        if np.all(adj_mat_r == adj_mat_r1) and np.all(adj_mat_r == adj_mat_p1):
            n_r_pass = 2
        elif irc_r_pass:
            n_r_pass = 1
        else:
            n_r_pass = 0
        irc_p_pass = np.all(adj_mat_p == adj_mat_r1) | np.all(adj_mat_p == adj_mat_p1)
        if np.all(adj_mat_p == adj_mat_r1) and np.all(adj_mat_p == adj_mat_p1):
            n_p_pass = 2
        elif irc_p_pass:
            n_p_pass = 1
        else:
            n_p_pass = 0
    elif adj_mat_r1 is not None:
        irc_r_pass = np.all(adj_mat_r == adj_mat_r1)
        if irc_r_pass:
            n_r_pass = 1
        else:
            n_r_pass = 0
        irc_p_pass = np.all(adj_mat_p == adj_mat_r1)
        if irc_p_pass:
            n_p_pass = 1
        else:
            n_p_pass = 0
    else:
        irc_r_pass = False
        irc_p_pass = False
    print(idx, irc_r_pass, irc_p_pass, n_r_pass, n_p_pass)
    firctest.write("%d    %s  %s    %d %d\n"%(idx, irc_r_pass, irc_p_pass, n_r_pass, n_p_pass))
    if irc_r_pass and irc_p_pass:
        p_number_passed_IRC[idx] = idx_passed_IRC
        idx_passed_IRC += 1
        

import matplotlib.pyplot as plt
plt.figure(figsize=(4,3.5))

# plt.scatter((e_optimized[:]*12+778.6669508446397*12)*eV_2_kcalmol, (all_idx_min[:,2]+778.6669508446397*12)*eV_2_kcalmol, s=60)
plt.scatter((e_optimized[idx_final]*12+778.6669508446397*12)*eV_2_kcalmol, Eb_final, s=60)

for i, (x, y) in enumerate(zip((e_optimized[idx_final]*12+778.6669508446397*12)*eV_2_kcalmol, Eb_final)):
    # if x < 25 and y < 80:
    if idx_final[i] in p_number_passed_IRC.keys():
        text_number = p_number_passed_IRC[idx_final[i]]
        plt.text(x, y, str(text_number), fontsize=9, ha='center', va='center', color='orange')
plt.xlabel("$\Delta E$ (kcal/mol)")
plt.ylabel("$E_b$ (kcal/mol)")
plt.tight_layout()
plt.savefig("PassedIRC_samples")

fpout = open("PassedIRC.txt", 'w')
print("Passed IRC")
for idx in p_number_passed_IRC.keys():
    fpout.write(f"{p_number_passed_IRC[idx]}    {(e_optimized[idx]*12+778.6669508446397*12)*eV_2_kcalmol}  {Eb_final[list(idx_final).index(idx)]}    {idx}\n")
    print(p_number_passed_IRC[idx], (e_optimized[idx]*12+778.6669508446397*12)*eV_2_kcalmol, Eb_final[list(idx_final).index(idx)], idx)
