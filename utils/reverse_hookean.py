import numpy as np
from ase.constraints import FixConstraint


class ReverseHookean(FixConstraint):
    """Applies a Hookean force only when distance is below rt (pushes apart)."""

    def __init__(self, a1, a2, rt, k):
        self.a1 = a1
        self.a2 = a2
        self.rt = rt
        self.k = k

    def adjust_forces(self, atoms, forces):
        r12 = atoms.positions[self.a2] - atoms.positions[self.a1]
        r = np.linalg.norm(r12)
        if r == 0 or r >= self.rt:
            return
        e = r12 / r
        # Reverse Hookean: only when r < rt, push apart toward rt
        fmag = -self.k * (r - self.rt)
        fvec = fmag * e
        forces[self.a2] += fvec
        forces[self.a1] -= fvec
