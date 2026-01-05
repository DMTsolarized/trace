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
    PiFaceApproach,
    MixedApproach,
    EnergyScorer,
)
from utils.dtxb_calculator import DXTBCalculator


def find_vinyl_double_bond(atoms):
    pos = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    carbons = [i for i, s in enumerate(symbols) if s == "C"]
    for i in carbons:
        for j in carbons:
            if i < j:
                d = np.linalg.norm(pos[i] - pos[j])
                if 1.30 < d < 1.40:
                    return i, j
    return None


fragA = list(iread("ase_ox.xyz"))[-1]
fragB = list(iread("styrene.xyz"))[-1]

pd_idx = fragA.get_chemical_symbols().index("Pd")
carbons = [i for i, s in enumerate(fragB.get_chemical_symbols()) if s == "C"]
vinyl = find_vinyl_double_bond(fragB)
if vinyl is None:
    raise RuntimeError("Could not locate vinyl C=C bond in styrene.")

vinyl_c, vinyl_partner = vinyl

siteA = ReactiveSite(atom_idx=pd_idx, site_type="metal")
siteB = ReactiveSite(
    atom_idx=vinyl_c,
    partner_idx=vinyl_partner,
    plane_indices=carbons,
    site_type="vinyl",
)

builder = PrecomplexBuilder(
    fragA,
    fragB,
    siteA,
    siteB,
    distances=np.linspace(2.91, 2.91 * 1.5, 7),
)

sigma = BondSigmaStarApproach()
pi_face = PiFaceApproach()
mixed = MixedApproach(sigma, pi_face, weights=np.linspace(0.0, 0.9, 10))

builder.add_approach_generator(sigma)
builder.add_approach_generator(pi_face)
builder.add_approach_generator(mixed)
builder.set_energy_scorer(EnergyScorer(method="GFN1", spin=1))

results = builder.build(max_keep=40, do_energy=True, max_energy_eval=40)
os.makedirs("styrene", exist_ok=True)

for i, result in enumerate(filter(lambda x: x["clash_score"] <= 0, results[:1])):
    fragA_copy = fragA.copy()
    fragA_copy.set_positions(result["at_pos"])
    merged = fragA_copy + fragB
    merged.calc = DXTBCalculator(method="GFN1")

    scale = 1.35
    pd_num = fragA[pd_idx].number
    c_num = fragB[vinyl_c].number
    c_partner_num = fragB[vinyl_partner].number
    d_PdC = (covalent_radii[pd_num] + covalent_radii[c_num]) * scale
    d_PdC_partner = (covalent_radii[pd_num] + covalent_radii[c_partner_num]) * scale
    nA = len(fragA)
    pd_idx_m = pd_idx
    c_idx_m = nA + vinyl_c
    c_partner_idx_m = nA + vinyl_partner

    cons = [
        Hookean(a1=pd_idx_m, a2=c_idx_m, rt=d_PdC, k=0.2 / d_PdC**2),
        Hookean(
            a1=pd_idx_m,
            a2=c_partner_idx_m,
            rt=d_PdC_partner,
            k=0.1 / d_PdC_partner**2,
        ),
    ]
    merged.set_constraint(cons)

    opt = LBFGS(merged, trajectory=f"styrene/precomplex_{i}.traj")
    opt.run(fmax=0.10)
    merged.set_constraint()
    write(f"styrene/precomplex_{i}.xyz", merged)
