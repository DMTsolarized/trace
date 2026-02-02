from __future__ import annotations

from runner.stage_runners.base_stage_runner import StageRunner


class RelaxStageRunner(StageRunner):

    def run(self) -> None:
        calculator_factory = self.context.calculator_factories.get(self.stage.id)
        if calculator_factory is None or calculator_factory(None) is None:
            raise ValueError("Default relax stage must have a calculator attached")
        self.calculate()
