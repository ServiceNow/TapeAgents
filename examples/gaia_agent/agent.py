from tapeagents.agent import Agent
from tapeagents.llms import LLM
from tapeagents.nodes import MonoNode
from tapeagents.steps import ActionExecutionFailure, VideoObservation
from tapeagents.tools.simple_browser import PageObservation

from .prompts import PromptRegistry
from .steps import AGENT_STEPS, STEPS_WITHOUT_CODE, FactsSurvey, Plan
from .tape import GaiaTape


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
    @classmethod
    def create(cls, llm: LLM, plain_code: bool = False, **kwargs):
        steps_prompt = PromptRegistry.allowed_steps_code if plain_code else PromptRegistry.allowed_steps
        steps = STEPS_WITHOUT_CODE if plain_code else AGENT_STEPS
        nodes = [
            GaiaNode(name="plan", guidance=PromptRegistry.plan, agent_steps=Plan),
            GaiaNode(name="facts_survey", guidance=PromptRegistry.facts_survey, agent_steps=FactsSurvey),
            GaiaNode(name="start", guidance=PromptRegistry.start, steps_prompt=steps_prompt, agent_steps=steps),
            GaiaNode(name="act", steps_prompt=steps_prompt, agent_steps=steps, next_node="act"),
        ]
        return super().create(llm, nodes=nodes, max_iterations=2, **kwargs)
