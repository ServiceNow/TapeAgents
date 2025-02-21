import platform
from typing import Any

from pydantic import Field

from tapeagents.agent import Agent
from tapeagents.core import Prompt, Step
from tapeagents.llms import LLM
from tapeagents.nodes import StandardNode
from tapeagents.tools.browser import PageObservation

from .prompts import (
    ABSTRACT_EXAMPLE,
    ACT,
    ALLOWED_STEPS,
    BASELINE_STEPS_PROMPT,
    BASELINE_SYSTEM_PROMPT,
    BE_CAUTIOUS,
    CONCRETE_EXAMPLE,
    GOAL_INSTRUCTIONS,
    HINTS,
    MAC_HINT,
    REFLECT,
    START,
    SYSTEM_PROMPT,
)
from .steps import (
    WorkArenaAction,
    WorkArenaAgentStep,
    WorkArenaBaselineStep,
    WorkArenaTape,
    WorkArenaTask,
)


class WorkArenaBaselineNode(StandardNode):
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

    guidance: str = ""
    agent_steps: type[Step] | tuple[type[Step], ...] = Field(exclude=True, default=WorkArenaAgentStep)

    def make_prompt(self, agent: Any, tape: WorkArenaTape) -> Prompt:
        assert isinstance(tape.steps[1], WorkArenaTask)
        goal = GOAL_INSTRUCTIONS.format(goal=tape.steps[1].task)
        obs = [s for s in tape if isinstance(s, PageObservation)][-1].text
        history = self.history_prompt(tape)
        allowed_steps = ALLOWED_STEPS.format(allowed_steps=agent.llm.get_step_schema(WorkArenaBaselineStep))
        mac_hint = MAC_HINT if platform.system() == "Darwin" else ""
        main_prompt = f"""{goal}\n{obs}\n{history}
{BASELINE_STEPS_PROMPT}{allowed_steps}{mac_hint}
{HINTS}
{BE_CAUTIOUS}
{ABSTRACT_EXAMPLE}
{CONCRETE_EXAMPLE}
        """.strip()
        return Prompt(
            messages=[
                {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
                {"role": "user", "content": main_prompt},
            ]
        )

    def history_prompt(self, tape: WorkArenaTape) -> str:
        prompts = []
        i = 0
        for step in tape:
            if isinstance(step, WorkArenaAction):
                prompts.append(f"## step {i}")
                prompts.append(step.llm_view(indent=None))
                i += 1
            elif isinstance(step, PageObservation) and step.error:
                prompts.append(f"Error from previous action: {step.error}")
        if len(prompts):
            prompt = "# History of interaction with the task:\n" + "\n".join(prompts) + "\n"
        else:
            prompt = ""
        return prompt


class WorkArenaAgent(Agent):
    @classmethod
    def create(cls, llm: LLM, max_iterations: int = 4):
        return super().create(
            llm,
            nodes=[
                StandardNode(
                    name="set_goal",
                    system_prompt=SYSTEM_PROMPT,
                    guidance=START,
                    steps_prompt=ALLOWED_STEPS,
                    steps=WorkArenaAgentStep,
                ),
                StandardNode(
                    name="reflect",
                    system_prompt=SYSTEM_PROMPT,
                    guidance=REFLECT,
                    steps_prompt=ALLOWED_STEPS,
                    steps=WorkArenaAgentStep,
                ),
                StandardNode(
                    name="act",
                    system_prompt=SYSTEM_PROMPT,
                    guidance=ACT,
                    next_node="reflect",
                    steps_prompt=ALLOWED_STEPS,
                    steps=WorkArenaAgentStep,
                ),
            ],
            max_iterations=max_iterations,
        )
