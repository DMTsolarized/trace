import numpy as np
from ase.io import iread, write
from utils.precomplex_builder import (
    ReactiveSite,
    PrecomplexBuilder,
    BondSigmaStarApproach,
    PiFaceApproach,
    MixedApproach,
    EnergyScorer,
)
from utils.dtxb_calculator import DXTBCalculator
from ase.optimize import LBFGS
from ase.constraints import Hookean
from ase.data import covalent_radii, atomic_numbers


fragA = list(iread("dontdelete/ni_complex.xyz"))[-1]  # metal complex (fragment A)
fragB = list(iread("ar_br.xyz"))[-1]  # substrate (fragment B)
# detect simple reactive indices (user must verify or implement template detection)
pd_idx = fragA.get_chemical_symbols().index("Ni")
# for substrate: find Br and its nearest carbon
ar_sym = fragB.get_chemical_symbols()
br_idx = ar_sym.index("Br")
ar_pos = fragB.get_positions()
carbons = [i for i, s in enumerate(ar_sym) if s == "C"]
c_idx = carbons[
    int(np.argmin([np.linalg.norm(ar_pos[br_idx] - ar_pos[i]) for i in carbons]))
]
siteA = ReactiveSite(atom_idx=pd_idx, site_type="metal")
siteB = ReactiveSite(
    atom_idx=c_idx,
    partner_idx=br_idx,
    plane_indices=carbons,  # crude ring
    site_type="aryl_CBr",
)
builder = PrecomplexBuilder(
    fragA,
    fragB,
    siteA,
    siteB,
    distances=np.linspace(3.294, 3.294 * 1.35, 7),
)

# approach generators: sigma*, ring-normal, and mixtures
sigma = BondSigmaStarApproach()
ring = PiFaceApproach()
mixed = MixedApproach(sigma, ring, weights=np.linspace(0.0, 0.9, 14))
builder.add_approach_generator(sigma)
builder.add_approach_generator(ring)
builder.add_approach_generator(mixed)
# optionally random
# builder.add_approach_generator(RandomApproach(n=12, seed=42))
# energy scorer
builder.set_energy_scorer(EnergyScorer(method="GFN1", spin=1))
# distances to sample
results = builder.build(max_keep=40, do_energy=True, max_energy_eval=40)


for i, result in enumerate(filter(lambda x: x["clash_score"] <= 0, results[:5])):
    fragA_copy = fragA.copy()
    fragA_copy.set_positions(result["at_pos"])
    merged = fragA_copy + fragB

    merged.calc = DXTBCalculator(method="GFN1")

    # --- distance restraints from covalent radii ---
    scale = 1.35
    pd_num = fragA[pd_idx].number
    c_num = fragB[c_idx].number
    br_num = fragB[br_idx].number
    d_PdC = (covalent_radii[pd_num] + covalent_radii[c_num]) * scale
    d_PdBr = (covalent_radii[pd_num] + covalent_radii[br_num]) * scale
    nA = len(fragA)  # number of atoms in the metal fragment

    pd_idx_m = pd_idx  # Pd lives in fragA
    c_idx_m = nA + c_idx  # ipso C in merged
    br_idx_m = nA + br_idx  # Br in merged

    cons = [
        Hookean(a1=pd_idx_m, a2=c_idx_m, rt=d_PdC, k=0.2 / d_PdC**2),
        Hookean(a1=pd_idx_m, a2=br_idx_m, rt=d_PdBr, k=0.2 / d_PdBr**2),
    ]
    merged.set_constraint(cons)

    opt = LBFGS(merged, trajectory=f"builder_test/test_out_{i}.traj")
    # opt.run(fmax=0.50, steps=20)  # only local cleanup
    merged.set_constraint()
    write(f"builder_test_ni/test_out_{i}.xyz", merged)
