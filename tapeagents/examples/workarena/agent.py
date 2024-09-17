import platform
from typing import Any

from tapeagents.core import Prompt
from tapeagents.dialog import SystemStep, UserStep
from tapeagents.guided_agent import GuidedAgent
from tapeagents.utils import get_step_schemas_from_union_type

from .prompts import TEMPLATES, PromptRegistry
from .steps import (
    PageObservation,
    WorkArenaAction,
    WorkArenaAgentStep,
    WorkArenaBaselineStep,
    WorkArenaTape,
    WorkArenaTask,
)


class WorkArenaBaseline(GuidedAgent):
    """
    Agent that is close to the original workarena one.
    Implemented features (best feature set for gpt4o from workarena paper):
    - thinking
    - action_history
    - last_error
    - extract_visible_tag
    - extract_clickable_tag
    - individual_examples
    - long_description
    """

    _start_step_cls: Any = PageObservation
    _agent_step_cls: Any = WorkArenaAgentStep

    def make_prompt(self, tape: WorkArenaTape) -> Prompt:
        assert isinstance(tape.steps[1], WorkArenaTask)
        goal = PromptRegistry.goal_instructions.format(goal=tape.steps[1].task)
        obs = [s for s in tape if isinstance(s, PageObservation)][-1].text
        history = self.history_prompt(tape)
        allowed_steps = PromptRegistry.allowed_steps.format(
            allowed_steps=get_step_schemas_from_union_type(WorkArenaBaselineStep)
        )
        mac_hint = PromptRegistry.mac_hint if platform.system() == "Darwin" else ""
        main_prompt = f"""\
{goal}
{obs}
{history}
{PromptRegistry.baseline_steps_prompt}{allowed_steps}{mac_hint}
{PromptRegistry.hints}
{PromptRegistry.be_cautious}
{PromptRegistry.abstract_example}
{PromptRegistry.concrete_example}
        """.strip()
        messages = [
            SystemStep(content=PromptRegistry.baseline_system_prompt).model_dump(),
            UserStep(content=main_prompt).model_dump(),
        ]
        return Prompt(messages=messages)

    def history_prompt(self, tape: WorkArenaTape) -> str:
        prompts = []
        i = 0
        for step in tape:
            if isinstance(step, WorkArenaAction):
                prompts.append(f"## step {i}")
                prompts.append(step.llm_view(indent=None))
                i += 1
            elif isinstance(step, PageObservation) and step.last_action_error:
                prompts.append(f"Error from previous action: {step.last_action_error}")
        if len(prompts):
            prompt = "# History of interaction with the task:\n" + "\n".join(prompts) + "\n"
        else:
            prompt = ""
        return prompt


class WorkArenaAgent(GuidedAgent):
    _start_step_cls: Any = PageObservation
    _agent_step_cls: Any = WorkArenaAgentStep
    templates: dict[str, str] = TEMPLATES

    def get_steps_description(self, tape) -> str:
        return self.templates["allowed_steps"].format(
            allowed_steps=get_step_schemas_from_union_type(WorkArenaAgentStep)
        )

    def prepare_tape(self, tape, max_chars: int = 100):
        """
        Trim all page observations except the last two.
        """
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
