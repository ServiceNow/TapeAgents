from tapeagents.agent import Agent
from tapeagents.core import Step
from tapeagents.llms import LLM
from tapeagents.nodes import StandardNode

from .prompts import (
    ACT_STEPS,
    ALLOWED_STEPS,
    ALLOWED_STEPS_CODE,
    FACTS_SURVEY,
    FORMAT,
    PLAN,
    REFLECT_AND_ACT,
    SYSTEM_PROMPT,
)
from .steps import THOUGHTS, FactsSurvey, Plan


class GaiaAgent(Agent):
    name: str = "gaia_agent_v4"

    @classmethod
    def create(cls, llm: LLM, actions: tuple[Step, ...], plain_code: bool = False, **kwargs):
        steps_prompt = ALLOWED_STEPS_CODE if plain_code else ACT_STEPS
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
                name="act",
                system_prompt=SYSTEM_PROMPT,
                guidance=FORMAT,  # REFLECT_AND_ACT,
                steps_prompt=steps_prompt,
                steps=steps,
                next_node="act",
            ),
        ]
        return super().create(llm, nodes=nodes, max_iterations=2, **kwargs)
