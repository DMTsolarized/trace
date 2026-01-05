from __future__ import annotations

import glob
import os

import matplotlib.pyplot as plt

from ase.io import read

from utils.dtxb_calculator import DXTBCalculator


def main() -> None:
    pattern = os.path.join("parser_tests", "oxidative_addition", "result_*.xyz")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"No files found for pattern: {pattern}")

    energies = []
    labels = []
    for path in paths:
        atoms = read(path)
        atoms.calc = DXTBCalculator(method="GFN1")
        energy = atoms.get_potential_energy()
        energies.append(energy)
        labels.append(os.path.basename(path))

    plt.figure(figsize=(8, 4))
    plt.plot(range(len(energies)), energies, marker="o", linestyle="-")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.xlabel("Structure")
    plt.ylabel("Energy (eV)")
    plt.title("Precomplex Energies")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
