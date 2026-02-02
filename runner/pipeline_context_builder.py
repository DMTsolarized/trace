from __future__ import annotations

from ase import Atoms

from runner.candidate_pool import CandidatePool
from runner.pipeline_context import PipelineContext
from runner.reaction_parser import ReactionDefinition, ValidationError


class PipelineContextBuilder:
    def __init__(self, definition: ReactionDefinition) -> None:
        self.definition = definition

    def build(self) -> PipelineContext:
        reactant_atoms: dict[str, Atoms] = {}
        reactants = {}
        for reactant in self.definition.reactants:
            if reactant.atoms is None:
                raise ValidationError(f"Reactant '{reactant.id}' is missing atoms.")
            reactant_atoms[reactant.id] = reactant.atoms
            reactants[reactant.id] = reactant
        calculator_factories = {
            stage.id: stage.calculator_factory for stage in self.definition.pipeline
        }
        optimizer_factory = self.definition.settings.optimization.optimizer
        pool = CandidatePool([])
        context = PipelineContext(
            reactant_atoms=reactant_atoms,
            reactants=reactants,
            bond_formation=self.definition.settings.bond_formation,
            calculator_factories=calculator_factories,
            optimizer_factory=optimizer_factory,
            _pool=pool,
            workdir=self.definition.workdir,
            filter_registry={},
        )
        context.mount()
        return context
