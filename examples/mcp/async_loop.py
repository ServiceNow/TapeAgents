import asyncio
import logging
import os
import time

import aiohttp
import hydra
from omegaconf import DictConfig

from examples.gaia_agent.eval import load_dataset, tape_correct, task_to_observations
from examples.gaia_agent.steps import GaiaTape
from tapeagents.core import StopStep
from tapeagents.io import save_json_tape
from tapeagents.orchestrator import EnvironmentGroup, async_execute_with_env

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="async_mcp",
)
def main(cfg: DictConfig) -> None:
    asyncio.run(amain(cfg))


async def amain(cfg: DictConfig) -> None:
    cfg.llm.base_url = os.environ["BASE_URL"]
    tasks = load_dataset(cfg.split)

    dt = time.perf_counter()
    connector = aiohttp.TCPConnector(limit=50000, limit_per_host=50000, keepalive_timeout=1.0)
    timeout = aiohttp.ClientTimeout(total=3600.0, connect=3600.0, sock_read=3600.0)
    coroutines = []
    results = []
    if cfg.only_tasks:
        selected_tasks = [(level, task_num) for level, task_num in cfg.only_tasks]
    else:
        selected_tasks = [
            (level, task_num) for level, level_tasks in tasks.items() for task_num, task in enumerate(level_tasks)
        ]
    try:
        async with EnvironmentGroup(cfg.environment, n_envs=cfg.n_envs) as env_manager:
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                for level, task_num in selected_tasks:
                    logging.info(f"Schedule task {level=}, {task_num=}")
                    task = tasks[level][task_num]
                    start_tape = GaiaTape(steps=task_to_observations(task))  # type: ignore
                    start_tape.metadata.id = f"l{level}_task{task_num:03d}"
                    start_tape.metadata.task = task
                    coroutines.append(async_execute_with_env(env_manager, cfg.agent, start_tape, session))
                logger.info(f"Solving {len(coroutines)} tasks")
                results = await asyncio.gather(*coroutines)
    except Exception as e:
        logger.error(f"Error during execution: {e}")

    total_steps = 0
    tapes_with_errors = 0
    solved = 0
    for tape in results:
        stop_steps = [step for step in tape if isinstance(step, StopStep)]
        tape.metadata.result = "" if not stop_steps else stop_steps[-1].model_dump().get("answer", "")
        solved += int(tape_correct(tape))
        total_steps += len(tape.steps)
        if tape.metadata.error:
            tapes_with_errors += 1
    logger.info(f"Average tape length: {(total_steps / len(results)):.2f}")
    logger.info(f"Tapes with error: {(tapes_with_errors / len(results)):.2f} ({tapes_with_errors} of {len(results)})")
    logger.info(f"Accuracy: {(solved / len(results)):.2f} ({solved} of {len(results)})")
    for tape in results:
        save_json_tape(tape, os.path.join(cfg.exp_path, "tapes"), tape.metadata.parent_id)
    logger.info(f"Execution time: {time.perf_counter() - dt:.2f} seconds")


if __name__ == "__main__":
    main()
