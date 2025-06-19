import platform
from pprint import pprint
from typing import Any

from pydantic import Field

from examples.workarena.prompts import (
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
from examples.workarena.steps import (
    WorkArenaAgentStep,
    WorkArenaTape,
    WorkArenaTask,
)
from tapeagents.agent import Agent
from tapeagents.core import Action, Prompt, Step
from tapeagents.llms import LLM
from tapeagents.nodes import StandardNode
from tapeagents.tool_calling import as_openai_tool
from tapeagents.tools.browser import PageObservation


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
        mac_hint = MAC_HINT if platform.system() == "Darwin" else ""
        main_prompt = f"""{goal}\n\n# Web Page Content\n{obs}\n\n{history}
{BASELINE_STEPS_PROMPT}{mac_hint}
{HINTS}
{BE_CAUTIOUS}
{ABSTRACT_EXAMPLE}
{CONCRETE_EXAMPLE}
        """.strip()
        self._steps = self.prepare_step_types(agent)
        tools = [as_openai_tool(s).model_dump() for s in self._steps]
        pprint(tools, width=140)
        return Prompt(
            messages=[
                {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
                {"role": "user", "content": main_prompt},
            ],
            tools=tools,
        )

    def history_prompt(self, tape: WorkArenaTape) -> str:
        prompts = []
        i = 0
        for step in tape:
            if isinstance(step, Action):
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
