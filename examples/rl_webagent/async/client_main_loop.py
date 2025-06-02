import asyncio
import logging
import os
import time

import aiohttp
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from examples.gaia_agent.eval import tape_correct
from examples.rl_webagent.agent import WebAgent
from examples.rl_webagent.scripts.orchestrate_rl import load_webtasks_debug
from examples.rl_webagent.steps import WebTape
from tapeagents.core import StopStep
from tapeagents.io import save_json_tape
from tapeagents.orchestrator import async_execute_agent
from tapeagents.remote_environment import AsyncRemoteEnvironment

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path=".",
    config_name="client_main_loop",
)
def main(cfg: DictConfig) -> None:
    asyncio.run(amain(cfg))


async def run_agent_with_remote_env(cfg: DictConfig, task: dict, session: aiohttp.ClientSession) -> WebTape:
    environment: AsyncRemoteEnvironment = instantiate(cfg.environment)  # type: ignore
    async with environment.acontext(session, wait_for_env=True) as env:
        # TODO: figure things out below...
        tape, metadata = await env.start_task({"task_entrypoint": task["task"], "seed": task["seed"]})
        actions = await environment.a_actions()
        tools_description = await environment.a_tools_description()
        logger.info(f"Available tools: {tools_description}")
        agent: WebAgent = instantiate(cfg.agent, known_actions=actions, tools_description=tools_description)
        tape = await async_execute_agent(agent, tape, env, session)
        return tape


async def amain(cfg: DictConfig) -> None:
    # cfg.llm.base_url = os.environ["BASE_URL"]
    os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(cfg.output_dir, "tapedata.sqlite")
    os.environ["MINIWOB_URL"] = cfg.environment_variables.miniwob_url
    # os.environ["SNOW_INSTANCE_URL"] = cfg.environment_variables.snow_instance_url
    # os.environ["SNOW_INSTANCE_UNAME"] = cfg.environment_variables.snow_instance_uname
    # os.environ["SNOW_INSTANCE_PWD"] = cfg.environment_variables.snow_instance_pwd

    ### Step 1: load datasets ###
    # train_samples, test_samples = load_webtasks(train_split=cfg.train_split, seeds=cfg.seeds)
    train_samples, test_samples = load_webtasks_debug()  # TODO: load all tasks when ready

    dt = time.perf_counter()
    timeout = 3600.0
    connector = aiohttp.TCPConnector(limit=1000, limit_per_host=1000, force_close=True)
    timeout = aiohttp.ClientTimeout(total=timeout, connect=timeout, sock_read=timeout)
    coroutines = []
    results = []

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for task in train_samples:
            logging.info(f"Schedule task {task['task'].get_task_id()} with seed {task['seed']}")
            coroutines.append(run_agent_with_remote_env(cfg, task, session))
        logger.info(f"Solving {len(coroutines)} tasks")
        results = await asyncio.gather(*coroutines)

    # TODO: breakpoint here to debug results

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
    print("Make sure that you run environment server first using `uv run examples/rl_webagent/async/env_server.py`")
    main()
