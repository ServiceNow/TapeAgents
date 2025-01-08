"""Module to optimize agents"""

import logging
import random
from typing import Callable

from pydantic import BaseModel

from .agent import Agent
from .dialog_tape import Tape
from .llm_function import LLMFunctionNode

logger = logging.getLogger(__name__)


class OptimizationResult(BaseModel):
    id: int
    agent: Agent
    metric: float


def optimize_demos(
    agent: Agent,
    good_tapes: list[Tape],
    n_demos: int,
    n_iterations: int,
    seed: int,
    metric_fn: Callable[[list[Tape], str], float],
    run_agent_fn: Callable[[Agent, list[Tape]], list[Tape]],
    post_run_agent_fn: Callable[[OptimizationResult], None] | None,
) -> OptimizationResult:
    """
    Iteratively adds demonstration tapes to the agent, runs the agent, and evaluates its performance
    to find the optimal set of demonstrations that maximize the agent's performance metric.

    Parameters:
    agent (Agent): The initial agent to be optimized.
    good_tapes (list[Tape]): A list of good demonstration tapes.
    n_demos (int): The number of demonstrations to add to the agent.
    n_iterations (int): The number of tries to select the best set of demoinstration.
    seed (int): The initial seed for random number generation, incremented in each iteration to ensure variability per iteration.
    metric_fn (Callable[[list[Tape]], float]): A function to compute the performance metric from a list of tapes.
    run_agent_fn (Callable[[Agent, list[Tape]], list[Tape]]): A function to run the agent and return the resulting tapes.
    post_run_agent_fn (Callable[[OptimizationResult], None]): A callback function to be called after each iteration.

    Returns:
    OptimizationResult: Agents and metrics for each iterations, including the best performing one.
    """
    best_metric = 0
    best_agent = agent
    best_agent_id = 0

    for i in range(n_iterations):
        logger.debug(f"Optimization iteration {i}")
        # Add demos to the agent with a different seed for each attempt
        new_agent = add_demos(agent, good_tapes, n_demos, seed=seed + i)
        # Run agent on the validation set to get metric to optimize
        final_tapes = run_agent_fn(new_agent)
        metric = metric_fn(final_tapes, f"optimization_{i}")
        # Update best agent if necessary
        if metric > best_metric:
            best_metric = metric
            best_agent = new_agent
            best_agent_id = i
        # Callback results
        if post_run_agent_fn:
            post_run_agent_fn(OptimizationResult(id=i, agent=new_agent, metric=metric))
    # Return the best agent
    return OptimizationResult(id=best_agent_id, agent=best_agent, metric=best_metric)


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
