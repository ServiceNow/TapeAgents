import random
from tapeagents.agent import Agent
from tapeagents.core import Tape
from tapeagents.llm_function import LLMFunctionNode


def add_demos(agent: Agent, tapes: list[Tape], max_n_demos: int, seed: int = 1):
    """Extract demos for function templates the given tapes.
    
    When there is too many demos, select random ones.
    
    """
    demos = {template_name: [] for template_name in agent.templates}
    for tape in tapes:
        for node, index in agent.parse(tape):
            if isinstance(node, LLMFunctionNode):
                demos[node.template_name].append(node.extract_demo(agent, tape, index))
    rng = random.Random(seed)
    agent_copy = agent.model_copy(deep=True)
    for template_name, template in agent_copy.templates.items():
        k = min(max_n_demos, len(demos[template_name]))
        template.demos = rng.sample(demos[template_name], k)
    return agent_copy
    