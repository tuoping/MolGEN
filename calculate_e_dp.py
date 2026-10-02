from deepmd.calculator import DP
from pathlib import Path
import re
import time

import ase.io

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


def trajectory_sort_key(path):
    """Sort numeric trajectory suffixes numerically and other suffixes by name."""
    suffix = path.stem.removeprefix("gentraj_")
    return (0, int(suffix)) if suffix.isdigit() else (1, suffix)


def main():
    work_dir = Path.cwd()
    trajectory_paths = sorted(work_dir.glob("gentraj_*.xyz"), key=trajectory_sort_key)
    if not trajectory_paths:
        print(f"No gentraj_*.xyz files found in {work_dir}")
        return

    calculator = None
    for trajectory_path in trajectory_paths:
        suffix = trajectory_path.stem.removeprefix("gentraj_")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", suffix):
            print(f"WARNING:: skipping unexpected filename {trajectory_path.name}")
            continue

        energy_path = work_dir / f"all_energy_atoms_{suffix}.dat"
        # A non-empty output indicates that this trajectory was already handled.
        # Missing and zero-byte outputs are (re)calculated.
        if energy_path.is_file() and energy_path.stat().st_size > 0:
            print(
                f"WARNING:: skipping {trajectory_path.name}; "
                f"{energy_path.name} exists and is non-empty"
            )
            continue

        if calculator is None:
            calculator = make_dp_calculator(MODEL_PATH)

        start_time = time.time()
        traj = ase.io.read(trajectory_path, format="extxyz", index=":")
        if not traj:
            print(f"WARNING:: skipping empty trajectory {trajectory_path.name}")
            continue

        atoms = traj[-1]
        validate_elements(atoms, calculator.type_dict, trajectory_path)
        atoms.calc = calculator
        energy_atoms = atoms.get_potential_energy()
        energy_path.write_text(f"{energy_atoms}\n", encoding="utf-8")
        print(
            f"{trajectory_path.name} done, time: "
            f"{time.time() - start_time:.2f} s"
        )


if __name__ == "__main__":
    main()
