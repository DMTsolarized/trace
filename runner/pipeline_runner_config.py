from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Type, TYPE_CHECKING
from runner.stage_runners.relax_stage_runner import RelaxStageRunner
from runner.stage_runners.relax_reactants_stage_runner import RelaxReactantsStageRunner
from runner.stage_runners.bi_molecular_sample_stage_runner import (
    BiMolSamplingStageRunner,
)
from runner.stage_runners.move_stage_runner import MoveStageRunner
from utils.post_calculation_filters import filter_based_on_energies_and_forces

if TYPE_CHECKING:
    from runner.pipeline_runner import StageRunner

DEFAULT_ACTION_REGISTRY: dict[str, Type["StageRunner"]] = {
    "relax": RelaxStageRunner,
    "relax-reactants": RelaxReactantsStageRunner,
    "bi-sample": BiMolSamplingStageRunner,
    "walk": MoveStageRunner,
}

DEFAULT_FILTER_REGISTRY: dict[str, Callable[..., bool]] = {
    "energy_force": filter_based_on_energies_and_forces,
}


@dataclass(frozen=True)
class PipelineRunnerConfig:
    action_registry: dict[str, Type["StageRunner"]] = field(default_factory=dict)
    filter_registry: dict[str, Callable[..., bool]] = field(default_factory=dict)

    def merged_action_registry(
        self,
        defaults: Mapping[str, Type["StageRunner"]],
    ) -> dict[str, Type["StageRunner"]]:
        merged = dict(defaults)
        merged.update(self.action_registry)
        return merged

    def merged_filter_registry(
        self,
        defaults: Mapping[str, Callable[..., bool]],
    ) -> dict[str, Callable[..., bool]]:
        merged = dict(defaults)
        merged.update(self.filter_registry)
        return merged
