"""Module to optimize agents"""

import logging
import random
from typing import Callable

from .agent import Agent
from .dialog_tape import Tape
from .llm_function import LLMFunctionNode

logger = logging.getLogger(__name__)


def optimize_demos(
    agent: Agent,
    good_tapes: list[Tape],
    val_dataset: list,
    n_demos: int,
    random_sample: int,
    seed: int,
    metric_fn: Callable[[list, list[Tape]], float],
    run_agent_fn: Callable[[Agent, list[Tape]], list[Tape]],
) -> Agent:
    """
    Try `random_sample` times to `add_demos()` (see below), measure validation set performance, and keep the best agent.
    """
    best_agent = agent
    best_metric = 0

    for i in range(random_sample):
        # Add demos to the agent with a different seed for each attempt
        new_agent = add_demos(best_agent, good_tapes, n_demos, seed=seed + i)
        # Run agent on the validation set to get metric to optimize
        final_tapes = run_agent_fn(new_agent, val_dataset)
        metric = metric_fn(val_dataset, final_tapes, f"optimization_{i}")
        if metric > best_metric:
            best_metric = metric
            best_agent = new_agent
    return best_agent


def add_demos(agent: Agent, tapes: list[Tape], max_n_demos: int, seed: int = 1) -> Agent:
    """
    Extract demos for function templates from the given tapes.

    When there is too many demos, select random ones.
    """
    demos = {template_name: [] for template_name in agent.templates}
    for tape in tapes:
        for node, index in agent.get_node_runs(tape):
            if isinstance(node, LLMFunctionNode):
                demos[node.template_name].append(node.extract_demo(agent, tape, index))
    rng = random.Random(seed)
    agent_copy = agent.model_copy(deep=True)
    for template_name, template in agent_copy.templates.items():
        k = min(max_n_demos, len(demos[template_name]))
        template.demos = rng.sample(demos[template_name], k)
    return agent_copy
