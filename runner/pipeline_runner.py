from __future__ import annotations

from typing import Mapping, Type

from runner.pipeline_context_builder import PipelineContextBuilder
from runner.reaction_parser import ReactionDefinition
from runner.pipeline_runner_config import (
    DEFAULT_ACTION_REGISTRY,
    DEFAULT_FILTER_REGISTRY,
    PipelineRunnerConfig,
)
from runner.stage_runners.base_stage_runner import StageRunner


class PipelineRunner:
    def __init__(
        self,
        definition: ReactionDefinition,
        config: PipelineRunnerConfig | None = None,
        action_registry: Mapping[str, Type["StageRunner"]] | None = None,
    ) -> None:
        self.definition = definition
        self.config = config or PipelineRunnerConfig()
        defaults = action_registry or DEFAULT_ACTION_REGISTRY
        self.action_registry = self.config.merged_action_registry(defaults)
        self.filter_registry = self.config.merged_filter_registry(
            DEFAULT_FILTER_REGISTRY
        )
        self.context = PipelineContextBuilder(definition).build()
        self.context.filter_registry = self.filter_registry

    def run(self) -> None:
        for stage in self.definition.pipeline:
            runner_cls = self.action_registry.get(stage.action, StageRunner)
            runner = runner_cls(stage, self.context)
            runner.run()
            runner.end()
