from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, TYPE_CHECKING
from ase.io import write

from runner.pipeline_context import PipelineContext
from runner.reaction_parser import PipeStage


if TYPE_CHECKING:
    from runner.candidate_pool import CandidatePool


class StageRunner(ABC):
    def __init__(
        self,
        stage: PipeStage,
        context: PipelineContext,
    ) -> None:
        self.stage = stage
        self.context = context
        self.dir: str = context.stage_dir(stage.id)

    @abstractmethod
    def run(self) -> None:
        """
        Defaults as a stage runner for actions whicb have not been introduced and is not one of the defaults:
        defaults list here
        """
        raise NotImplementedError

    def calculate(self) -> None:
        calculator_factory = self.context.calculator_factories.get(self.stage.id)
        if calculator_factory is None:
            return
        for candidate in self.context.pool:
            calculator = calculator_factory(None)
            if calculator is None:
                return
            candidate.atoms.calc = calculator
            opt = self.context.optimizer_factory(candidate.atoms)
            opt.run(fmax=self.stage.fmax, steps=self.stage.steps)

    def write_stage(self) -> None:
        """
        writes to checkpoint folder, strips atoms of every constraint to avoid errors
        """
        for i, atoms in enumerate(self.context.pool.atoms):
            # need to remove constraint
            atoms.set_constraint([])
            write(f"{self.dir}/figure_{i}.xyz", atoms)

    def apply_filters(self) -> None:
        for spec in self.stage.filters:
            if spec.name not in self.context.filter_registry:
                raise ValueError(
                    f"Unknown filter '{spec.name}' for stage '{self.stage.id}'."
                )
            filter_fn = self.context.filter_registry[spec.name]
            args = dict(spec.args)
            if "e_min" in args and (args["e_min"] is None or args["e_min"] == "auto"):
                if not self.context.pool.candidates:
                    return
                args["e_min"] = min(
                    candidate.atoms.get_total_energy()
                    for candidate in self.context.pool
                )
            self.context.pool = self.context.pool.filter(
                lambda candidate: filter_fn(candidate.atoms, **args)
            )

    def end(self) -> None:
        self.apply_filters()
        self.write_stage()
