from ase.data import covalent_radii
from utils.reaction_parser import BondFormationSettings
import numpy as np
from typing import Generator
from ase import Atoms
from ase.constraints import Hookean


def get_covalent_bond_length(numbers: tuple[int, int]) -> float:
    a1, a2 = numbers
    return covalent_radii[a1] + covalent_radii[a2]


def generate_dynamic_distances_for_sample(
    settings: BondFormationSettings, numbers: tuple[int, int]
):
    bond_length: float = get_covalent_bond_length(numbers) * settings.distance_scale
    range_low, range_top = settings.distance_range
    return np.linspace(bond_length * range_low, bond_length * range_top, 5)


def recover_indices_from_merged_complex(
    len: int, idx_site_a: int, idx_site_b: int
) -> tuple[int, int]:
    return (idx_site_a, len + idx_site_b)


def walk(start: float, target: float, step: float) -> Generator[float, None, None]:
    d = start
    sign = -1 if start > target else 1
    step = abs(step) * sign

    while (d - target) * sign < 1e-6:
        yield round(d, 3)
        d += step


def set_pair_distance_split(
    atoms: Atoms, i: int, j: int, d_target: float, start_idx: int
) -> None:
    r_i = atoms.positions[i]
    r_j = atoms.positions[j]
    v = r_j - r_i
    dist = np.linalg.norm(v)
    if dist < 1e-12:
        return
    unit = v / dist
    shift = (d_target - dist) * unit
    atoms.positions[start_idx:] += shift
