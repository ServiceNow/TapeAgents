import asyncio
import logging
import os
import time

import aiohttp
import hydra
from omegaconf import DictConfig

from examples.gaia_agent.eval import load_dataset, task_to_observations
from examples.gaia_agent.steps import GaiaTape
from tapeagents.io import save_json_tape
from tapeagents.orchestrator import execute_with_config

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
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for level, task_num in cfg.only_tasks:
            logging.info(f"Schedule task {level=}, {task_num=}")
            task = tasks[level][task_num]
            start_tape = GaiaTape(steps=task_to_observations(task))  # type: ignore
            start_tape.metadata.parent_id = f"l{level}_task{task_num:03d}"
            coroutines.append(execute_with_config(cfg, start_tape, session))
        logger.info(f"Solving {len(coroutines)} tasks")
        results = await asyncio.gather(*coroutines)

    for tape in results:
        save_json_tape(tape, os.path.join(cfg.exp_path, "tapes"), tape.metadata.parent_id)
        print(tape.metadata.parent_id, tape[-1])
    logger.info(f"Execution time: {time.perf_counter() - dt:.2f} seconds")


if __name__ == "__main__":
    main()
