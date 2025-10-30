import psi4
from pathlib import Path

def read_xyz(path: Path):
    txt = path.read_text().strip().splitlines()
    if len(txt) < 2:
        raise ValueError("XYZ file too short")
    try:
        n = int(txt[0].strip())
    except Exception:
        raise ValueError("First line of XYZ must be atom count")
    
    # skip comment line (txt[1])
    atoms = []
    for line in txt[2+n+2:2+n+2+n]:
    # for line in txt[2:2+n]:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Bad XYZ atom line: {line}")
        sym, x, y, z = parts[0], parts[1], parts[2], parts[3]
        atoms.append((sym, float(x), float(y), float(z)))
    return atoms

def read_xyz_optimized(path: Path):
    txt = path.read_text().strip().splitlines()
    if len(txt) < 2:
        raise ValueError("XYZ file too short")
    '''
    try:
        n = int(txt[0].strip())
    except Exception:
        raise ValueError("First line of XYZ must be atom count")
    '''
    # skip comment line (txt[1])
    atoms = []
    # for line in txt[2+n+2:2+n+2+n]:
    # for line in txt[2:2+n]:
    for line in txt[1:]:
        parts = line.split()
        if len(parts) < 4:
            break
            # raise ValueError(f"Bad XYZ atom line: {line}")
        sym, x, y, z = parts[0], parts[1], parts[2], parts[3]
        atoms.append((sym, float(x), float(y), float(z)))
    print("Number of atoms = ", len(atoms))
    return atoms

def atoms_to_psi4_geom(atoms, charge, multiplicity):
    lines = [f"{charge} {multiplicity}"]
    for sym, x, y, z in atoms:
        lines.append(f"{sym:2s}  {x: .10f}  {y: .10f}  {z: .10f}")
    lines += ["symmetry c1", "units angstrom"]
    return "\n".join(lines)

psi4.set_memory('4 GB')
psi4.set_num_threads(max(1, psi4.get_num_threads()))

import sys
import numpy as np
inpath = Path("ts_optimized.xyz")
atoms = read_xyz_optimized(inpath)
# idx_inpath = int(sys.argv[1])
# idxmin = int(np.loadtxt("../idxmin.dat")[idx_inpath][1])
# inpath = Path(f"gentraj_{idxmin}.xyz")
# atoms = read_xyz(inpath)

geom_str = atoms_to_psi4_geom(atoms, 0, 1)
mol = psi4.geometry(geom_str)
print(f"Hessian of ts_optimized.xyz")
'''
# TS optimization
psi4.set_options({'opt_type': 'ts', 'geom_maxiter': 100})
Ets, wfn_ts = psi4.optimize('wb97x-d3/def2-TZVP', molecule=mol, return_wfn=True)

# Freq at TS, write Hessian
psi4.set_options({'hessian_write': True})
Ef, wfn_freq = psi4.frequency('wb97x-d3/def2-TZVP', molecule=mol, return_wfn=True)
'''
# IRC settings
import glob
hessian_file = glob.glob("optimized.*.hess")
assert len(hessian_file) == 1
base_opts = {
    'opt_type': 'irc',
    'irc_points': 25,
    'irc_step_size': 0.15,
    'cart_hess_read': True,
    'geom_maxiter': 200,
    "e_convergence": 2e-3,
    'hessian_file': hessian_file[0],         # <-- match your actual file name
}
psi4.set_options(base_opts)
'''
# Forward
print(f"IRC forward of ts_optimized.xyz")
psi4.set_options({'irc_direction': 'forward'})
Efwd, wfn_fwd = psi4.optimize('wb97x-d3/def2-TZVP', molecule=mol.clone(), return_wfn=True)
'''
# Backward
print(f"IRC backward of ts_optimized.xyz")
psi4.set_options({
    'irc_direction': 'backward',
    'irc_save_path': True,         # <- save geometries along IRC
    'writer_file_label': 'irc_p-optimized'   # <- prefix for all saved files
    })
Ebwd, wfn_bwd = psi4.optimize('wb97x-d3/def2-TZVP', molecule=mol.clone(), return_wfn=True)

