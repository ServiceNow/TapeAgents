from tapeagents.agent import Agent
from tapeagents.core import Step
from tapeagents.llms import LLM
from tapeagents.nodes import StandardNode

from .prompts import (
    ALLOWED_STEPS,
    ALLOWED_STEPS_CODE,
    FACTS_SURVEY,
    FORMAT,
    PLAN,
    START,
    SYSTEM_PROMPT,
)
from .steps import THOUGHTS, FactsSurvey, Plan


class GaiaAgent(Agent):
    name: str = "gaia_agent_v3"

    @classmethod
    def create(cls, llm: LLM, actions: tuple[Step, ...], plain_code: bool = False, **kwargs):
        steps_prompt = ALLOWED_STEPS_CODE if plain_code else ALLOWED_STEPS
        steps = actions + THOUGHTS
        sp = SYSTEM_PROMPT
        nodes = [
            StandardNode(name="plan", system_prompt=sp, guidance=PLAN, agent_steps=Plan),
            StandardNode(name="facts_survey", system_prompt=sp, guidance=FACTS_SURVEY, agent_steps=FactsSurvey),
            StandardNode(name="start", system_prompt=sp, guidance=START, steps_prompt=steps_prompt, agent_steps=steps),
            StandardNode(
                name="act",
                system_prompt=sp,
                guidance=FORMAT,
                steps_prompt=steps_prompt,
                agent_steps=steps,
                next_node="act",
            ),
        ]
        return super().create(llm, nodes=nodes, max_iterations=2, **kwargs)
