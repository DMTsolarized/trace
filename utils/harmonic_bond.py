import numpy as np
from ase.constraints import FixConstraint


class HarmonicBond(FixConstraint):
    def __init__(self, a1, a2, rt, k):
        self.a1 = a1
        self.a2 = a2
        self.rt = rt
        self.k = k
        if isinstance(a2, int):
            self._type = "atoms"
        else:
            self._type = "point"

    def adjust_forces(self, atoms, forces):
        if self._type == "atoms":
            r12 = atoms.positions[self.a2] - atoms.positions[self.a1]
        else:
            r12 = np.array(self.a2, dtype=float) - atoms.positions[self.a1]
        r = np.linalg.norm(r12)
        if r == 0:
            return
        e = r12 / r
        # harmonic: U = 1/2 k (r - rt)^2  ->  F = -k (r - rt) e
        fmag = -self.k * (r - self.rt)
        fvec = fmag * e
        if self._type == "atoms":
            forces[self.a2] += fvec
            forces[self.a1] -= fvec
        else:
            forces[self.a1] -= fvec
