from pydantic import Field

from tapeagents.agent import Agent
from tapeagents.core import Step
from tapeagents.llms import LLM
from tapeagents.nodes import MonoNode
from tapeagents.tools.simple_browser import PageObservation

from .prompts import PromptRegistry
from .steps import (
    WebAgentStep,
    WebTape,
)


class WebNode(MonoNode):
    system_prompt: str = PromptRegistry.system_prompt
    steps_prompt: str = PromptRegistry.allowed_steps
    agent_steps: type[Step] | tuple[type[Step], ...] = Field(exclude=True, default=WebAgentStep)

    def prepare_tape(self, tape: WebTape, max_chars: int = 100):
        """
        Trim all page observations except the last two.
        """
        tape = super().prepare_tape(tape)  # type: ignore
        page_positions = [i for i, step in enumerate(tape.steps) if isinstance(step, PageObservation)]
        if len(page_positions) < 2:
            return tape
        prev_page_position = page_positions[-2]
        steps = []
        for step in tape.steps[:prev_page_position]:
            if isinstance(step, PageObservation):
                short_text = f"{step.text[:max_chars]}\n..." if len(step.text) > max_chars else step.text
                new_step = step.model_copy(update=dict(text=short_text))
            else:
                new_step = step
            steps.append(new_step)
        trimmed_tape = tape.model_copy(update=dict(steps=steps + tape.steps[prev_page_position:]))
        return trimmed_tape


class WebAgent(Agent):
    @classmethod
    def create(cls, llm: LLM, max_iterations: int = 4):
        return super().create(
            llm,
            nodes=[
                MonoNode(
                    name="set_goal",
                    guidance=PromptRegistry.start,
                    system_prompt=PromptRegistry.system_prompt,
                    steps_prompt=PromptRegistry.allowed_steps,
                    agent_steps=WebAgentStep,
                ),
                MonoNode(
                    name="reflect",
                    guidance=PromptRegistry.reflect,
                    system_prompt=PromptRegistry.system_prompt,
                    steps_prompt=PromptRegistry.allowed_steps,
                    agent_steps=WebAgentStep,
                ),
                MonoNode(
                    name="act",
                    guidance=PromptRegistry.act,
                    system_prompt=PromptRegistry.system_prompt,
                    steps_prompt=PromptRegistry.allowed_steps,
                    agent_steps=WebAgentStep,
                    next_node="reflect",
                ),
            ],
            max_iterations=max_iterations,
        )
