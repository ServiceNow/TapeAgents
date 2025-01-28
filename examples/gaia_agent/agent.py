from tapeagents.agent import Agent
from tapeagents.core import Step
from tapeagents.llms import LLM
from tapeagents.nodes import StandardNode
from tapeagents.steps import ReasoningThought

from .prompts import (
    ALLOWED_STEPS,
    ALLOWED_STEPS_CODE,
    FACTS_SURVEY,
    FORMAT,
    PLAN,
    REFLECT_OBSERVATION,
    SYSTEM_PROMPT,
)
from .steps import THOUGHTS, FactsSurvey, Plan, ReadingResultThought


class GaiaAgent(Agent):
    name: str = "gaia_agent_v4"

    @classmethod
    def create(cls, llm: LLM, actions: tuple[Step, ...], plain_code: bool = False, **kwargs):
        steps_prompt = ALLOWED_STEPS_CODE if plain_code else ALLOWED_STEPS
        steps = actions + THOUGHTS
        nodes = [
            StandardNode(
                name="plan",
                system_prompt=SYSTEM_PROMPT,
                guidance=PLAN,
                steps_prompt=ALLOWED_STEPS,
                steps=Plan,
            ),
            StandardNode(
                name="facts_survey",
                system_prompt=SYSTEM_PROMPT,
                guidance=FACTS_SURVEY,
                steps_prompt=ALLOWED_STEPS,
                steps=FactsSurvey,
            ),
            StandardNode(
                name="reflect",
                system_prompt=SYSTEM_PROMPT,
                guidance=REFLECT_OBSERVATION,
                steps_prompt=steps_prompt,
                steps=(ReadingResultThought, ReasoningThought),
            ),
            StandardNode(
                name="act",
                system_prompt=SYSTEM_PROMPT,
                guidance=FORMAT,
                steps_prompt=steps_prompt,
                steps=steps,
                next_node="reflect",
            ),
        ]
        return super().create(llm, nodes=nodes, max_iterations=2, **kwargs)
