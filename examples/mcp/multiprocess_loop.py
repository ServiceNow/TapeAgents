import atexit
import logging
import multiprocessing
import os
import time

import hydra
from omegaconf import DictConfig, OmegaConf

from examples.gaia_agent.eval import load_dataset, tape_correct, task_to_observations
from examples.gaia_agent.steps import GaiaTape
from tapeagents.agent import Agent
from tapeagents.core import StopStep
from tapeagents.environment import Environment
from tapeagents.io import save_json_tape
from tapeagents.orchestrator import get_agent_and_env_from_config, main_loop
from tapeagents.parallel_processing import process_pool_processor

logger = logging.getLogger(__name__)

agent: Agent = None  # type: ignore
env: Environment = None  # type: ignore


def initialize_env_agent(cfg_dict: dict):
    cfg = DictConfig(cfg_dict)

    global agent, env
    agent, env = get_agent_and_env_from_config(cfg)

    # Initialize the environment
    try:
        env.reset()
    except Exception as e:
        logger.error(f"Error initializing environment: {e}")
        raise

    def env_close():
        """Ensure the environment is closed properly on exit"""
        if env:
            try:
                env.close()
                logger.info("Environment closed successfully.")
            except Exception as e:
                logger.error(f"Error closing environment: {e}")

    atexit.register(env_close)  # Register cleanup function to close the environment on exit


def execute_single_task(args):
    """Worker function to execute a single task with its own agent and environment"""
    level, task_num, task = args
    logging.basicConfig(
        format="%(asctime)s - PID_%(process)d - Thread_%(threadName)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
        force=True,  # forget previous handlers
    )
    logger.info(f"Processing task L{level}:{task_num:03d}")

    # Create separate agent and environment for this task
    global agent, env

    # Create start tape
    start_tape = GaiaTape(steps=task_to_observations(task))  # type: ignore

    # Execute the main loop
    final_tape = start_tape
    env.reset()  # Reset the environment for each task
    try:
        for event in main_loop(agent, start_tape, env, max_loops=50):
            if event.agent_event and event.agent_event.final_tape:
                final_tape = event.agent_event.final_tape
            elif event.env_tape:
                final_tape = event.env_tape

    except Exception as e:
        logger.exception(f"Error processing task {level}_{task_num}: {e}")
        final_tape.metadata.error = str(e)

    final_tape.metadata.task = task
    final_tape.metadata.parent_id = f"l{level}_task{task_num:03d}"
    stop_steps = [step for step in final_tape if isinstance(step, StopStep)]
    final_tape.metadata.result = "" if not stop_steps else stop_steps[-1].model_dump().get("answer", "")
    return final_tape


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="async_mcp",
)
def main(cfg: DictConfig) -> None:
    cfg.llm.base_url = os.environ["BASE_URL"]
    tasks = load_dataset(cfg.split)

    if cfg.only_tasks:
        selected_tasks = [(level, task_num) for level, task_num in cfg.only_tasks]
    else:
        selected_tasks = [
            (level, task_num) for level, level_tasks in tasks.items() for task_num, task in enumerate(level_tasks)
        ]

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    task_args = []
    for level, task_num in selected_tasks:
        task = tasks[level][task_num]
        task_args.append((level, task_num, task))

    logger.info(f"Processing {len(task_args)} tasks with {cfg.n_envs} workers")

    dt = time.perf_counter()

    results = []
    for result in process_pool_processor(
        stream=task_args,
        worker_func=execute_single_task,
        n_workers=cfg.n_envs,
        keep_order=True,
        initializer=initialize_env_agent,
        initargs=(cfg_dict,),
    ):
        if isinstance(result, Exception):
            logger.error(f"Task failed with exception: {result}")
            error_tape = GaiaTape(steps=[])
            error_tape.metadata.error = str(result)
            results.append(error_tape)
        else:
            results.append(result)

    total_steps = 0
    tapes_with_errors = 0
    solved = 0
    for tape in results:
        solved += int(tape_correct(tape))
        total_steps += len(tape.steps)
        if tape.metadata.error:
            tapes_with_errors += 1

    logger.info(f"Average tape length: {(total_steps / len(results)):.2f}")
    logger.info(f"Tapes with error: {(tapes_with_errors / len(results)):.2f} ({tapes_with_errors} of {len(results)})")
    logger.info(f"Accuracy: {(solved / len(results)):.2f} ({solved} of {len(results)})")

    os.makedirs(os.path.join(cfg.exp_path, "tapes"), exist_ok=True)
    for tape in results:
        save_json_tape(tape, os.path.join(cfg.exp_path, "tapes"), tape.metadata.parent_id)

    logger.info(f"Execution time: {time.perf_counter() - dt:.2f} seconds")


if __name__ == "__main__":
    # Set start method for multiprocessing (required for some systems)
    multiprocessing.set_start_method("spawn", force=True)
    main()
