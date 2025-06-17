import asyncio
import logging
import os
import random
import time

import aiohttp
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from examples.rl_webagent.agent import WebAgent
from examples.rl_webagent.scripts.orchestrate_rl import load_webtasks
from examples.rl_webagent.steps import WebTape
from tapeagents.core import Observation
from tapeagents.io import save_json_tape
from tapeagents.orchestrator import async_execute_agent
from tapeagents.remote_environment import AsyncRemoteEnvironment

logger = logging.getLogger(__name__)


def abt_to_json(tasks: list[dict]) -> list[dict]:
    """
    Convert AbstractBrowserTask tasks to dicts that are JSON serializable by only giving the dataset, task_id, and seed.
    """
    return [{"dataset": task["dataset"], "task": task["task"].get_task_id(), "seed": task["seed"]} for task in tasks]


async def run_agent_with_remote_env(
    cfg: DictConfig, task: dict, session: aiohttp.ClientSession, max_loops: int
) -> WebTape:
    environment: AsyncRemoteEnvironment = instantiate(cfg.environment)  # type: ignore
    async with environment.acontext(session, wait_for_env=True) as env:
        start_attempts = cfg.start_attempts
        t = time.perf_counter()
        while True:
            try:
                tape_dict, _ = await env.start_task(task)
                break
            except Exception as e:
                start_attempts -= 1
                if start_attempts <= 0:
                    raise e
                logger.warning(f"Failed to start task, retry after 5 seconds: {e}")
                await asyncio.sleep(5)
        start_time = time.perf_counter() - t
        logger.info(f"Task {task['task']}/{task['seed']} started in {start_time:.2f} seconds")
        tape: WebTape = WebTape(**tape_dict)  # convert http response dict to WebTape object
        t = time.perf_counter()
        try:
            actions = await env.a_actions()
            tools_description = await env.a_tools_description()
            llms = None
            try:
                # try to load multiple LLMs from the configuration and choose one randomly
                llms = instantiate(cfg.llms)
                logger.info(f"Loaded {len(llms)} LLMs from configuration.")
                llm = random.choice(llms)
            except Exception:
                pass
            logger.info(f"Using LLM: {llm.base_url}")
            agent: WebAgent = instantiate(cfg.agent, known_actions=actions, tools_description=tools_description)
            agent.llms["default"] = llm
            tape = await async_execute_agent(agent, tape, env, session, max_loops=max_loops)
        except Exception as e:
            logger.error(f"Error occurred while running agent: {e}")
            tape.metadata.error = str(e)
        tape.metadata.result = {"execution_time": time.perf_counter() - t, "start_time": start_time}
    # save the tape as we go
    save_json_tape(tape, os.path.join(cfg.exp_path, "tapes"), tape.metadata.parent_id)
    return tape


async def amain(cfg: DictConfig) -> None:
    os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(cfg.exp_path, "tapedata.sqlite")
    os.environ["MINIWOB_URL"] = cfg.environment_variables.miniwob_url
    # os.environ["SNOW_INSTANCE_URL"] = cfg.environment_variables.snow_instance_url
    # os.environ["SNOW_INSTANCE_UNAME"] = cfg.environment_variables.snow_instance_uname
    # os.environ["SNOW_INSTANCE_PWD"] = cfg.environment_variables.snow_instance_pwd

    ### Step 1: load datasets ###
    train_samples, test_samples = load_webtasks(train_split=cfg.train_split, seeds=cfg.seeds)
    # train_samples, test_samples = load_webtasks_debug()  # TODO: load all tasks when ready
    # Convert AbstractBrowserTask to json parsable dicts
    train_samples = abt_to_json(train_samples)
    test_samples = abt_to_json(test_samples)
    samples = test_samples
    dt = time.perf_counter()
    timeout = cfg.requests_timeout
    timeout = aiohttp.ClientTimeout(total=timeout, connect=timeout, sock_read=timeout)
    coroutines = []
    results = []

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for task in samples:
            logging.info(f"Schedule task {task['task']} with seed {task['seed']}")
            coroutines.append(run_agent_with_remote_env(cfg, task, session, max_loops=cfg.max_loops))
        logger.info(f"Solving {len(coroutines)} tasks")
        results: list[WebTape] = await asyncio.gather(*coroutines)

    logger.info(f"Saved {len(results)} tapes to {os.path.join(cfg.exp_path, 'tapes')}")

    ### Print some statistics
    total_steps = 0
    errs = 0
    no_reward = 0
    acc = []
    rewards = []
    for tape in results:
        total_steps += len(tape.steps)
        last_obs = [step for step in tape if isinstance(step, Observation)][-1]
        success = last_obs.metadata.other.get("reward", 0.0) > 0.5
        acc.append(success)
        if tape.metadata.error:
            errs += 1
            logger.warning(f"Error in tape {tape.metadata.id}: {tape.metadata.error}")
        elif (
            "info" in last_obs.metadata.other
            and "task_info" in last_obs.metadata.other["info"]
            and "REWARD_GLOBAL" in last_obs.metadata.other["info"]["task_info"]
        ):
            rewards.append(last_obs.metadata.other["info"]["task_info"]["REWARD_GLOBAL"])
        else:
            no_reward += 1
            logger.warning(f"No reward found in last observation of tape {tape.metadata.id}")

    logger.info(f"Average tape length: {(total_steps / len(results)):.2f}")
    logger.info(f"Total execution time: {time.perf_counter() - dt:.2f} seconds")
    logger.info(f"Average time per tape: {(time.perf_counter() - dt) / len(results):.2f} seconds")
    logger.info(f"Accuracy: {sum(acc) / len(acc) if acc else 0:.2f}")
    logger.info(f"Average reward: {sum(rewards) / len(rewards) if rewards else 0:.2f}")
    logger.info(f"Number of tapes with no reward: {no_reward}")
    logger.info(f"Failed tapes: {errs}")

    ### TODO: continue to copy things from orchestrate_rl.py / switch to pipelinerl


@hydra.main(
    version_base=None,
    config_path=".",
    config_name="client_main_loop",
)
def main(cfg: DictConfig) -> None:
    asyncio.run(amain(cfg))


if __name__ == "__main__":
    print("Make sure that you run environment server first using `uv run examples/rl_webagent/async/env_server.py`")
    main()
