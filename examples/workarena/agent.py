import platform
from typing import Any

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
    FinalAnswerAction,
    ReflectionThought,
    WorkArenaTape,
)
from tapeagents.agent import Agent
from tapeagents.core import Action, Prompt
from tapeagents.llms import LLM
from tapeagents.nodes import StandardNode
from tapeagents.steps import ReasoningThought
from tapeagents.tools.browser import PageObservation

mac_hint = MAC_HINT if platform.system() == "Darwin" else ""


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

    guidance: str = f"{mac_hint}\n{HINTS}\n{BE_CAUTIOUS}\n{ABSTRACT_EXAMPLE}\n{CONCRETE_EXAMPLE}"
    system_prompt: str = BASELINE_SYSTEM_PROMPT
    steps_prompt: str = BASELINE_STEPS_PROMPT

    def make_prompt(self, agent: Any, tape: WorkArenaTape) -> Prompt:
        goal = GOAL_INSTRUCTIONS.format(goal=tape.steps[1].content)
        obs = [s for s in tape if isinstance(s, PageObservation)][-1].text
        history = self.history_prompt(tape)

        self._steps = self.prepare_step_types(agent)
        steps_description = self.get_steps_description(agent)

        main_prompt = f"{goal}\n\n# Web Page Content\n{obs}\n\n{history}\n{steps_description}\n{self.guidance}"
        # tools = [as_openai_tool(s).model_dump() for s in self._steps]
        return Prompt(
            messages=[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": main_prompt}],
            # tools=tools,
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
                    use_known_actions=True,
                    steps=[ReasoningThought, ReflectionThought, FinalAnswerAction],
                ),
                StandardNode(
                    name="reflect",
                    system_prompt=SYSTEM_PROMPT,
                    guidance=REFLECT,
                    steps_prompt=ALLOWED_STEPS,
                    use_known_actions=True,
                    steps=[ReasoningThought, ReflectionThought, FinalAnswerAction],
                ),
                StandardNode(
                    name="act",
                    system_prompt=SYSTEM_PROMPT,
                    guidance=ACT,
                    next_node="reflect",
                    steps_prompt=ALLOWED_STEPS,
                    use_known_actions=True,
                    steps=[ReasoningThought, ReflectionThought, FinalAnswerAction],
                ),
            ],
            max_iterations=max_iterations,
        )
