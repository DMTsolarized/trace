from __future__ import annotations

from runner.stage_runners.base_stage_runner import StageRunner
from ase.io import write


class RelaxReactantsStageRunner(StageRunner):
    def run(self) -> None:
        calculator_factory = self.context.calculator_factories.get(self.stage.id)
        if calculator_factory is None:
            raise ValueError("Default relax stage must have a calculator attached")
        for reactant_id, reactant in self.context.reactants.items():
            if reactant.atoms is None:
                continue
            calculator = calculator_factory(None)
            if calculator is None:
                raise ValueError("Default relax stage must have a calculator attached")
            reactant.atoms.calc = calculator
            opt = self.context.optimizer_factory(reactant.atoms)
            opt.run(fmax=self.stage.fmax, steps=self.stage.steps)
            self.context.reactant_atoms[reactant_id] = reactant.atoms

    def write_stage(self) -> None:
        for reactant_id, atoms in self.context.reactant_atoms.items():
            atoms.set_constraint([])
            write(f"{self.dir}/{reactant_id}.xyz", atoms)
