"""Install MolGEN and its dependencies.

The PyTorch Geometric binary wheels are tied to the Python, PyTorch, and CUDA
versions.  The project currently targets the same PyTorch 2.6/CUDA 12.4 stack
documented in README.md, so their URLs are generated for the running Python.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

from setuptools import find_namespace_packages, setup


ROOT = Path(__file__).parent


def pyg_wheel_requirements() -> list[str]:
    """Return PyG extension wheels for Torch 2.6 and CUDA 12.4."""
    if platform.python_implementation() != "CPython":
        raise RuntimeError("MolGEN's PyG wheels require CPython.")

    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if python_tag not in {"cp39", "cp310", "cp311", "cp312"}:
        raise RuntimeError(
            "The Torch 2.6/CUDA 12.4 PyG wheels support Python 3.9 through "
            "3.12. Please create a compatible environment before installing."
        )

    system = platform.system()
    machine = platform.machine().lower()
    if system == "Linux" and machine in {"x86_64", "amd64"}:
        platform_tag = "linux_x86_64"
    elif system == "Windows" and machine in {"x86_64", "amd64"}:
        platform_tag = "win_amd64"
    else:
        raise RuntimeError(
            "Prebuilt Torch 2.6/CUDA 12.4 PyG wheels are only available for "
            "64-bit Linux and Windows on x86_64."
        )

    base_url = "https://data.pyg.org/whl/torch-2.6.0%2Bcu124"
    wheels = {
        "pyg-lib": "pyg_lib-0.5.0+pt26cu124",
        "torch-cluster": "torch_cluster-1.6.3+pt26cu124",
        "torch-scatter": "torch_scatter-2.1.2+pt26cu124",
        "torch-sparse": "torch_sparse-0.6.18+pt26cu124",
        "torch-spline-conv": "torch_spline_conv-1.2.2+pt26cu124",
    }
    return [
        f"{project} @ {base_url}/{wheel.replace('+', '%2B')}-"
        f"{python_tag}-{python_tag}-{platform_tag}.whl"
        for project, wheel in wheels.items()
    ]


install_requires = [
    "numpy==1.26.0",
    "pandas==1.5.3; python_version < '3.12'",
    "pandas>=2.1,<3; python_version >= '3.12'",
    "scikit-learn==1.6.1",
    "torch==2.6.0",
    "pytorch-lightning==2.0.4",
    "mdtraj==1.9.9; python_version < '3.12'",
    "mdtraj>=1.10,<2; python_version >= '3.12'",
    "biopython==1.79; python_version < '3.12'",
    "biopython>=1.83,<2; python_version >= '3.12'",
    "setuptools<81",
    "wandb",
    "dm-tree",
    "einops",
    "torchdiffeq",
    "fair-esm",
    "torch-linear-assignment",
    "matplotlib==3.7.2; python_version < '3.12'",
    "matplotlib>=3.8,<4; python_version >= '3.12'",
    "omegaconf==2.3.0",
    "ase==3.22",
    "pymatgen",
    "spglib",
    "matscipy",
    "torch-geometric",
    "e3nn",
    *pyg_wheel_requirements(),
]


setup(
    name="molgen-local-env",
    version="1.0.0",
    description="Flow matching for reaction pathway generation",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license="MIT",
    python_requires=">=3.9,<3.13",
    install_requires=install_requires,
    packages=find_namespace_packages(include=["mdgen", "mdgen.*"]),
)
