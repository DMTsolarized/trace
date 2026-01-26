import numpy as np
from ase import Atoms, units

# MAYBE USE ASE UNITS
EV_TO_KCALMOL = 23.06054194532933


def filter_based_on_energies_and_forces(
    obj: Atoms, e_cutoff: float, fmax_cutoff: float, e_min: float
) -> bool:
    return (
        obj.get_total_energy() - e_min
    ) * EV_TO_KCALMOL < e_cutoff and np.linalg.norm(
        obj.get_forces(), axis=1
    ).max() < fmax_cutoff
