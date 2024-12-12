import logging
from enum import Enum

from tapeagents.agent import Agent
from tapeagents.environment import CodeExecutionResult
from tapeagents.llms import LLM
from tapeagents.nodes import MonoNode
from tapeagents.steps import ActionExecutionFailure, ReasoningThought, VideoObservation
from tapeagents.tools.code_executor import PythonCodeAction
from tapeagents.tools.container_executor import extract_code_blocks
from tapeagents.tools.simple_browser import PageObservation

from .prompts import PromptRegistry
from .steps import (
    AGENT_STEPS,
    STEPS_WITHOUT_CODE,
    GaiaQuestion,
    ListOfFactsThought,
    PlanThought,
    ReadingResultThought,
)
from .tape import GaiaTape

logger = logging.getLogger(__name__)


class PlanningMode(str, Enum):
    simple = "simple"
    facts_and_sources = "facts_and_sources"
    multiplan = "multiplan"
    replan_after_sources = "replan_after_sources"
    reflect = "reflect"


class GaiaNode(MonoNode):
    system_prompt: str = PromptRegistry.system_prompt
    steps_prompt: str = PromptRegistry.allowed_steps

    def prepare_tape(self, tape: GaiaTape, max_chars: int = 200) -> GaiaTape:
        """
        Trim long observations except for the last 3 steps
        """
        tape = super().prepare_tape(tape)  # type: ignore
        steps = []
        steps_border = -3
        for step in tape.steps[:steps_border]:
            if isinstance(step, PageObservation) and len(step.text) > max_chars:
                trimmed_step = step.model_copy(update=dict(text=f"{step.text[:max_chars]}\n..."))
            elif isinstance(step, ActionExecutionFailure) and len(step.error) > max_chars:
                trimmed_step = step.model_copy(update=dict(error=f"{step.error[:max_chars]}\n..."))
            elif isinstance(step, VideoObservation):
                trimmed_step = step.model_copy(update=dict(video_contact_sheet_paths=None, subtitle_text=None))
            else:
                trimmed_step = step
            steps.append(trimmed_step)
        return tape.model_copy(update=dict(steps=steps + tape.steps[steps_border:]))


class GaiaAgent(Agent):
    plain_code: bool

    @classmethod
    def create(cls, llm: LLM, plain_code: bool = False, **kwargs):
        nodes = [
            GaiaNode(name="plan", guidance=PromptRegistry.plan, agent_steps=PlanThought),
            GaiaNode(name="facts_survey", guidance=PromptRegistry.facts_survey, agent_steps=ListOfFactsThought),
            GaiaNode(
                name="start_execution",
                guidance=PromptRegistry.start_execution,
                steps_prompt=PromptRegistry.allowed_steps_code if plain_code else PromptRegistry.allowed_steps,
                agent_steps=STEPS_WITHOUT_CODE if plain_code else AGENT_STEPS,
            ),
            GaiaNode(
                name="act",
                steps_prompt=PromptRegistry.allowed_steps_code if plain_code else PromptRegistry.allowed_steps,
                agent_steps=STEPS_WITHOUT_CODE if plain_code else AGENT_STEPS,
                next_node="act",
            ),
        ]
        return super().create(llm, nodes=nodes, max_iterations=2, plain_code=plain_code, **kwargs)
