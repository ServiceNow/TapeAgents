import logging
from enum import Enum
from typing import Any

from tapeagents.agent import Agent
from tapeagents.environment import CodeExecutionResult, ExecuteCode
from tapeagents.llms import LLM
from tapeagents.nodes import MonoNode
from tapeagents.steps import ActionExecutionFailure, VideoObservation
from tapeagents.tools.calculator import CalculationResultObservation
from tapeagents.tools.container_executor import extract_code_blocks
from tapeagents.tools.simple_browser import PageObservation

from .prompts import PromptRegistry
from .steps import (
    AGENT_STEPS,
    STEPS_WITHOUT_CODE,
    GaiaQuestion,
    ListOfFactsThought,
    PlanThought,
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
    allowed_steps: str

    def get_steps_description(self, tape: GaiaTape, agent: Any) -> str:
        """
        Allow different subset of steps based on the agent's configuration
        """
        return self.steps_prompt.format(allowed_steps=self.allowed_steps)

    def prepare_tape(self, tape: GaiaTape, max_chars: int = 200) -> GaiaTape:
        """
        Trim long observations except for the last 3 steps
        """
        tape = super().prepare_tape(tape)  # type: ignore
        steps = []
        for step in tape.steps[:-3]:
            if isinstance(step, PageObservation):
                short_text = f"{step.text[:max_chars]}\n..." if len(step.text) > max_chars else step.text
                new_step = step.model_copy(update=dict(text=short_text))
            elif isinstance(step, ActionExecutionFailure):
                short_error = f"{step.error[:max_chars]}\n..." if len(step.error) > max_chars else step.error
                new_step = step.model_copy(update=dict(error=short_error))
            elif isinstance(step, VideoObservation):
                new_step = step.model_copy(update=dict(video_contact_sheet_paths=None, subtitle_text=None))
            else:
                new_step = step
            steps.append(new_step)
        trimmed_tape = tape.model_copy(update=dict(steps=steps + tape.steps[-3:]))
        return trimmed_tape

    def trim_tape(self, tape: GaiaTape) -> GaiaTape:
        """
        Make tape shorter to fit llm context size limits
        """
        summarization_border = int(len(tape) * 0.66)
        short_tape = tape.model_copy(update=dict(steps=[]))
        pre_tape: GaiaTape = tape[:summarization_border]  # type: ignore
        for step in pre_tape.steps:
            if isinstance(
                step,
                (
                    GaiaQuestion,
                    PlanThought,
                    ListOfFactsThought,
                    CalculationResultObservation,
                    CodeExecutionResult,
                ),
            ):
                short_tape.steps.append(step)
        for step in tape.steps[summarization_border:]:
            short_tape.steps.append(step)
        logger.info(f"Tape reduced from {len(tape)} to {len(short_tape)} steps")
        return short_tape

    def parse_completion(self, llm_output: str, prompt_id: str):
        if llm_output.strip().startswith("```"):
            code_blocks = extract_code_blocks(llm_output)
            yield ExecuteCode(code=code_blocks)
        else:
            for step in super().parse_completion(llm_output, prompt_id):
                yield step


class GaiaAgent(Agent):
    plain_code: bool

    @classmethod
    def create(cls, llm: LLM, plain_code: bool = False, **kwargs):
        nodes = [
            GaiaNode(name="plan", guidance=PromptRegistry.plan, allowed_steps=plan_steps),
            GaiaNode(name="facts_survey", guidance=PromptRegistry.facts_survey, allowed_steps=plan_steps),
            GaiaNode(
                name="start_execution",
                guidance=PromptRegistry.start_execution,
                steps_prompt=PromptRegistry.allowed_steps_code if plain_code else PromptRegistry.allowed_steps,
                allowed_steps=STEPS_WITHOUT_CODE if plain_code else AGENT_STEPS,
            ),
            GaiaNode(
                name="act",
                steps_prompt=PromptRegistry.allowed_steps_code if plain_code else PromptRegistry.allowed_steps,
                allowed_steps=STEPS_WITHOUT_CODE if plain_code else AGENT_STEPS,
                next_node="act",
            ),
        ]
        return super().create(llm, nodes=nodes, max_iterations=2, plain_code=plain_code, **kwargs)
