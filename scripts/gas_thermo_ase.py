import sys
freq_file = sys.argv[1]
energy_file = sys.argv[2]
atoms_file_idx = sys.argv[3]
import ase.io
from pathlib import Path
def read_xyz_optimized(path: Path, num_atoms: int):
    txt = path.read_text().splitlines()
    if len(txt) < 2:
        raise ValueError("XYZ file too short")
    atoms = []
    for line in txt[1:num_atoms+1]:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Bad XYZ atom line: {line}")
        sym, x, y, z = parts[0], parts[1], parts[2], parts[3]
        atoms.append([float(x), float(y), float(z)])
    # print("Number of atoms = ", len(atoms))
    return atoms

atoms = ase.io.read(f"irc_traj.xyz", ":")[2]
num_atoms = len(atoms)
positions = read_xyz_optimized(Path(atoms_file_idx), num_atoms)
atoms.set_positions(positions)

import numpy as np
pot_energy = np.loadtxt(energy_file)

lines = open(freq_file, 'r').readlines()
for line in lines:
    if "Frequencies (cm^-1):" in line:
        freqs = [float(x.strip(',')) for x in line.split()[2:]]
if freqs[1] < 0:
    freq_file2 = 'ts_report_allfreq.txt'
    lines = open(freq_file, 'r').readlines()
    for line in lines:
        if "Frequencies (cm^-1):" in line:
            freqs = [float(x.strip(',')) for x in line.split()[2:]]

if freqs[0] < 0:
    vib_energies = np.array(freqs)[1:]*1.23984193E-4
else:
    vib_energies = np.array(freqs)*1.23984193E-4
from ase.thermochemistry import IdealGasThermo
thermo = IdealGasThermo(vib_energies = vib_energies,
                        geometry = 'nonlinear',
                        potentialenergy = pot_energy,
                        atoms = atoms,
                        symmetrynumber = 1,
                        spin = 0)

G = thermo.get_gibbs_energy(temperature=298.15, pressure=101325.0)
