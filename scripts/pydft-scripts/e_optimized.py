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
        total_permutations *= np.math.factorial(c)  # type: ignore
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


from ase.calculators.calculator import Calculator, all_changes
from ase.units import Hartree, Bohr
import numpy as np

class PySCFCalculator(Calculator):
    implemented_properties = ['energy']  # add 'forces' if you compute them
    def __init__(self, xc='wb97x', basis='6-31g*', charge=0, spin=0, **kwargs):
        super().__init__(**kwargs)
        self.xc, self.basis, self.charge, self.spin = xc, basis, charge, spin

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        # Lazy import to avoid import issues if not installed
        from pyscf import gto, dft

        # Build PySCF molecule (Angstrom → Bohr conversion handled by PySCF when unit='Angstrom')
        coords = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()
        mol = gto.Mole()
        mol.unit = 'Angstrom'
        mol.atom = [(s, tuple(x)) for s, x in zip(symbols, coords)]
        mol.basis = self.basis
        mol.charge = self.charge
        mol.spin = self.spin  # 2S
        mol.build()

        # RKS with requested functional
        mf = dft.RKS(mol)
        mf.xc = self.xc
        eh = mf.kernel()  # Hartree
        self.results['energy'] = eh * Hartree  # ASE wants eV


def psi4geom_to_ase(path):
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]

    # Detect "charge multiplicity" in first line
    head = lines[0].split()
    if len(head) == 2 and all(t.lstrip("+-").isdigit() for t in head):
        coords = []
        for l in lines[1:]:
            ll = l.lower()
            if ll.startswith(("symmetry", "units", "#")):
                break
            parts = l.split()
            if len(parts) >= 4:
                sym, x, y, z = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
                coords.append((sym, x, y, z))
        symbols = [s for s, *_ in coords]
        positions = np.array([[x, y, z] for _, x, y, z in coords])
        return Atoms(symbols=symbols, positions=positions)
    else:
        # Fallback: treat as normal XYZ
        from ase.io import read
        return read(path)

import ase, ase.io
from ase import Atoms

calculator = PySCFCalculator(xc='wb97x', basis='6-31g*', charge=0, spin=0)

import matplotlib.pyplot as plt
import numpy as np

import sys
idx_inpath = (sys.argv[1])

all_positions = None
all_positions_ref = None
all_energy_atoms = []
all_energy_atoms_ref = []
all_atomic_numbers = None
mean_distances = []
import time
s_time = time.time()

# ofile_e = open(f"./rollout_{idx_inpath}/optimized_energy_atoms.dat", "w")
ofile_e = open(f"./{idx_inpath}/optimized_energy_atoms.dat", "w")

# atoms = psi4geom_to_ase(f'./rollout_{idx_inpath}/ts_optimized.xyz')
atoms = psi4geom_to_ase(f'./{idx_inpath}/ts_optimized.xyz')
atoms.set_cell(np.eye(3,3)*50)
atoms.calc = calculator
energy_atoms = atoms.get_potential_energy()/ len(atoms.get_atomic_numbers())
ofile_e.write(f"{energy_atoms}\n")
all_energy_atoms.append(energy_atoms)
ofile_e.flush()
ofile_e.close()
