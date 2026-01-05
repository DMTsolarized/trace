import os
import numpy as np
from ase.io import iread, write
from ase.constraints import Hookean
from ase.optimize import LBFGS
from ase.data import covalent_radii

from utils.precomplex_builder import (
    ReactiveSite,
    PrecomplexBuilder,
    BondSigmaStarApproach,
    RandomApproach,
    infer_bonds_simple,
)
from utils.dtxb_calculator import DXTBCalculator


def build_neighbors(bonds):
    neighbors = {}
    for i, j in bonds:
        neighbors.setdefault(i, []).append(j)
        neighbors.setdefault(j, []).append(i)
    return neighbors


def find_beta_hydride_site(atoms, pd_idx):
    pos = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()
    symbols = atoms.get_chemical_symbols()
    bonds = infer_bonds_simple(pos, numbers, scale=1.2)
    neighbors = build_neighbors(bonds)

    alpha_carbons = [i for i in neighbors.get(pd_idx, []) if symbols[i] == "C"]
    if not alpha_carbons:
        carbons = [i for i, s in enumerate(symbols) if s == "C"]
        alpha_carbons = [
            carbons[
                int(np.argmin([np.linalg.norm(pos[pd_idx] - pos[i]) for i in carbons]))
            ]
        ]

    candidates = []
    for alpha in alpha_carbons:
        beta_carbons = [
            i for i in neighbors.get(alpha, []) if symbols[i] == "C" and i != pd_idx
        ]
        for beta_c in beta_carbons:
            beta_hs = [i for i in neighbors.get(beta_c, []) if symbols[i] == "H"]
            for beta_h in beta_hs:
                dist = float(np.linalg.norm(pos[pd_idx] - pos[beta_h]))
                candidates.append((dist, beta_c, beta_h))

    if not candidates:
        raise RuntimeError("Could not identify a beta C-H site for elimination check.")

    candidates.sort(key=lambda x: x[0])
    _, beta_c_idx, beta_h_idx = candidates[0]
    return beta_c_idx, beta_h_idx


mol = list(iread("dontdelete/syninsertion.xyz"))[-1]
pd_idx = mol.get_chemical_symbols().index("Pd")
beta_c_idx, beta_h_idx = find_beta_hydride_site(mol, pd_idx)

siteA = ReactiveSite(atom_idx=pd_idx, site_type="metal")
siteB = ReactiveSite(atom_idx=beta_h_idx, partner_idx=beta_c_idx, site_type="beta_H")

pd_num = mol[pd_idx].number
h_num = mol[beta_h_idx].number
sum_pd_h = covalent_radii[pd_num] + covalent_radii[h_num]
distances = np.linspace(sum_pd_h * 1.2, sum_pd_h * 3, 7)

builder = PrecomplexBuilder(
    mol,
    mol,
    siteA,
    siteB,
    distances=distances,
)

sigma = BondSigmaStarApproach()
random = RandomApproach(n=16, seed=42)
builder.add_approach_generator(sigma)
builder.add_approach_generator(random)

results = builder.build(max_keep=40, do_energy=False, max_energy_eval=0)
os.makedirs("beta_hydride", exist_ok=True)

for i, result in enumerate(filter(lambda x: x["clash_score"] <= 0, results[:1])):
    mol_copy = mol.copy()
    mol_copy.set_positions(result["at_pos"])
    mol_copy.calc = DXTBCalculator(method="GFN1")

    scale = 1.35
    c_num = mol[beta_c_idx].number
    d_PdH = (covalent_radii[pd_num] + covalent_radii[h_num]) * scale

    cons = [
        Hookean(a1=pd_idx, a2=beta_h_idx, rt=d_PdH, k=0.2 / d_PdH**2),
    ]
    mol_copy.set_constraint(cons)

    opt = LBFGS(mol_copy, trajectory=f"beta_hydride/precomplex_{i}.traj")
    opt.run(fmax=0.10)
    mol_copy.set_constraint()
    write(f"beta_hydride/precomplex_{i}.xyz", mol_copy)
