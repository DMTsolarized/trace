from __future__ import annotations

import numpy as np

from ase import Atoms

from runner.candidate_pool import Candidate, CandidatePool
from runner.stage_runners.base_stage_runner import StageRunner
from utils.generate_sample_distances import generate_dynamic_distances_for_sample
from utils.precomplex_builder import (
    BondMidpointApproach,
    BondSigmaStarApproach,
    MixedApproach,
    PiFaceApproach,
    PrecomplexBuilder,
)


class BiMolSamplingStageRunner(StageRunner):
    def run(self) -> None:
        sample_size = self.stage.kwargs.get("sample", 10)
        if self.context.bond_formation is None:
            raise ValueError("bond_formation settings are required for sampling.")
        reactant_map = self.context.reactants
        candidates: list[Candidate] = []
        for pair in self.context.bond_formation.atom_pairs:
            reactant_a = reactant_map.get(pair[0])
            reactant_b = reactant_map.get(pair[1])
            if reactant_a is None or reactant_b is None:
                raise ValueError(f"Reactant pair not found: {pair}")
            if reactant_a.atoms is None or reactant_b.atoms is None:
                raise ValueError(f"Missing Atoms for reactant pair: {pair}")
            if not reactant_a.reactive_centers or not reactant_b.reactive_centers:
                raise ValueError(f"Reactive centers missing for reactant pair: {pair}")

            site_a = reactant_a.reactive_centers[0]
            site_b = reactant_b.reactive_centers[0]
            site_a_atomic_number = reactant_a.atoms[site_a.atom_idx].number
            site_b_atomic_number = reactant_b.atoms[site_b.atom_idx].number
            builder = PrecomplexBuilder(
                reactant_a.atoms,
                reactant_b.atoms,
                site_a,
                site_b,
                distances=generate_dynamic_distances_for_sample(
                    settings=self.context.bond_formation,
                    numbers=(site_a_atomic_number, site_b_atomic_number),
                ),
            )
            sigma = BondSigmaStarApproach()
            ring = PiFaceApproach()
            midpoint = BondMidpointApproach()
            mixed = MixedApproach(sigma, ring, weights=np.linspace(0.0, 0.9, 18))
            builder.add_approach_generator(sigma)
            builder.add_approach_generator(ring)
            builder.add_approach_generator(mixed)
            builder.add_approach_generator(midpoint)
            results = builder.build(max_keep=sample_size, do_energy=False)

            for result in results:
                frag_a_copy = reactant_a.atoms.copy()
                frag_a_copy.set_positions(result["at_pos"])
                merged: Atoms = frag_a_copy + reactant_b.atoms
                candidates.append(Candidate(atoms=merged))

        self.context.pool = CandidatePool(candidates)
