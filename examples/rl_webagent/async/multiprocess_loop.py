import logging
import multiprocessing
import os
import time

import hydra
from omegaconf import DictConfig, OmegaConf
from termcolor import colored

from examples.gaia_agent.steps import GaiaTape
from examples.rl_webagent.scripts.orchestrate_rl import load_webtasks
from tapeagents.core import Observation
from tapeagents.io import save_json_tape
from tapeagents.orchestrator import execute_agent, get_agent_and_env_from_config
from tapeagents.parallel_processing import process_pool_processor

logger = logging.getLogger(__name__)


def abt_to_json(tasks: list[dict]) -> list[dict]:
    """
    Convert AbstractBrowserTask tasks to dicts that are JSON serializable by only giving the dataset, task_id, and seed.
    """
    return [{"dataset": task["dataset"], "task": task["task"].get_task_id(), "seed": task["seed"]} for task in tasks]


def execute_single_task(cfg: DictConfig):
    """Worker function to execute a single task with its own agent and environment"""
    logging.basicConfig(
        format="%(asctime)s - PID_%(process)d - Thread_%(threadName)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
        force=True,  # forget previous handlers
    )
    logger.info(f"Processing task {cfg.task}")
    assert cfg.task, "Task must be provided in the configuration"

    try:
        agent, env = get_agent_and_env_from_config(cfg)
        tape, _ = env.start_task(cfg.task)  # type: ignore
        t = time.perf_counter()
        try:
            tape = execute_agent(agent, tape, env, max_loops=cfg.max_loops)
        except Exception as e:
            logger.error(f"Error occurred while running agent: {e}")
        logger.info("agent finished")
        tape.metadata.result = {"execution_time": time.perf_counter() - t}
        logger.info("save tape")
        save_json_tape(tape, os.path.join(cfg.exp_path, "tapes"), tape.metadata.parent_id)  # type: ignore
        logger.info("close environment")
        env.close()
    except Exception as e:
        logger.exception(f"Error during task execution: {e}")
        raise e
    return tape


@hydra.main(
    version_base=None,
    config_path=".",
    config_name="multiprocess_loop",
)
def main(cfg: DictConfig) -> None:
    os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(cfg.exp_path, "tapedata.sqlite")
    os.environ["MINIWOB_URL"] = cfg.environment_variables.miniwob_url

    train_samples, test_samples = load_webtasks(train_split=cfg.train_split, seeds=cfg.seeds)
    # train_samples, test_samples = load_webtasks_debug()  # TODO: load all tasks when ready
    # Convert AbstractBrowserTask to json parsable dicts
    train_samples = abt_to_json(train_samples)
    test_samples = abt_to_json(test_samples)

    task_args = []
    interpolated_cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    for task in test_samples:
        task_cfg = OmegaConf.create(interpolated_cfg_dict)
        task_cfg.task = task
        task_args.append(task_cfg)

    logger.info(f"Processing {len(task_args)} tasks with {cfg.workers} workers")

    dt = time.perf_counter()

    results = []
    for result in process_pool_processor(
        stream=task_args,
        worker_func=execute_single_task,
        n_workers=cfg.workers,
        keep_order=False,
    ):
        if isinstance(result, Exception):
            logger.error(f"Task failed with exception: {result}")
            error_tape = GaiaTape(steps=[])
            error_tape.metadata.error = str(result)
            results.append(error_tape)
        else:
            results.append(result)
        logger.info(colored(f"{len(results)} of {len(task_args)} tasks processed so far", "blue"))

    ### Print some statistics
    total_steps = 0
    errs = 0
    acc = []
    rewards = []
    for tape in results:
        total_steps += len(tape.steps)
        last_obs = [step for step in tape if isinstance(step, Observation)][-1]
        success = last_obs.metadata.other.get("reward", 0.0) > 0.5
        acc.append(success)
        if (
            "info" in last_obs.metadata.other
            and "task_info" in last_obs.metadata.other["info"]
            and "REWARD_GLOBAL" in last_obs.metadata.other["info"]["task_info"]
        ):
            rewards.append(last_obs.metadata.other["info"]["task_info"]["REWARD_GLOBAL"])
        else:
            errs += 1
            logger.warning(f"No reward found in last observation of tape {tape.metadata.id}")

    logger.info(f"Average tape length: {(total_steps / len(results)):.2f}")
    logger.info(f"Total execution time: {time.perf_counter() - dt:.2f} seconds")
    logger.info(f"Average time per tape: {(time.perf_counter() - dt) / len(results):.2f} seconds")
    logger.info(f"Accuracy: {sum(acc) / len(acc) if acc else 0:.2f}")
    logger.info(f"Average reward: {sum(rewards) / len(rewards) if rewards else 0:.2f}")
    logger.info(f"Failed tapes: {errs}")


if __name__ == "__main__":
    # Set start method for multiprocessing (required for some systems)
    multiprocessing.set_start_method("spawn", force=True)
    main()
