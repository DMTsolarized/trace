import numpy as np
from ase.io import iread, write
from utils.precomplex_builder import (
    ReactiveSite,
    PrecomplexBuilder,
    SigmaStarApproach,
    PiFaceApproach,
    MixedApproach,
    EnergyScorer,
)
from utils.dtxb_calculator import DXTBCalculator
from ase.optimize import BFGS


def find_vinyl_from_coords(positions, symbols):
    C_idx = [i for i, s in enumerate(symbols) if s == "C"]
    vinyl = []
    for i in C_idx:
        for j in C_idx:
            if i < j:
                d = np.linalg.norm(positions[i] - positions[j])
                if 1.30 < d < 1.40:  # crude filter for C=C
                    vinyl = [i, j]
                    return vinyl
    return None


fragA = list(iread("ase_ox.xyz"))[-1]  # metal complex (fragment A)
fragB = list(iread("styrene.xyz"))[-1]  # substrate (fragment B)
# detect simple reactive indices (user must verify or implement template detection)
pd_idx = fragA.get_chemical_symbols().index("Pd")
# for substrate: find Br and its nearest carbon
ar_sym = fragB.get_chemical_symbols()
ar_pos = fragB.get_positions()
vinyl = find_vinyl_from_coords(ar_pos, ar_sym)
carbons = [i for i, s in enumerate(ar_sym) if s == "C"]
siteA = ReactiveSite(atom_idx=pd_idx, site_type="metal")
siteB = ReactiveSite(atom_idx=0, site_type="C=C")
builder = PrecomplexBuilder(fragA, fragB, siteA, siteB)
# approach generators: sigma*, ring-normal, and mixtures
ring = PiFaceApproach(ring_indices=vinyl if vinyl else carbons)
builder.add_approach_generator(ring)
# optionally random
# builder.add_approach_generator(RandomApproach(n=12, seed=42))
# energy scorer
builder.set_energy_scorer(EnergyScorer(method="GFN1", spin=1))
# distances to sample
builder.set_distances(np.linspace(3.2, 4.5, 14))
results = builder.build(max_keep=40, do_energy=True, max_energy_eval=40)


for i, result in enumerate(filter(lambda x: x["clash_score"] <= 0, results[:2])):
    copy = fragA.copy()
    copy.set_positions(result["at_pos"])
    test_complex_out = copy + fragB
    test_complex_out.calc = DXTBCalculator(method="GFN1")
    opt_out = BFGS(test_complex_out, trajectory=f"styrene/test_out_{i}.traj")
    opt_out.run(fmax=0.04)
    write(f"styrene/test_out_{i}.xyz", test_complex_out)
