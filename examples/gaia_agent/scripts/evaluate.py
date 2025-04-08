import json
import logging
import os
import time

import hydra
from hydra.utils import instantiate
from joblib import Parallel, delayed
from omegaconf import DictConfig

from tapeagents.io import save_json_tape, save_tape_images
from tapeagents.llms import TrainableLLM
from tapeagents.orchestrator import get_agent_and_env_from_config
from tapeagents.tools.container_executor import init_code_sandbox

from ..eval import get_exp_config_dict, load_dataset, solve_task

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../../conf",
    config_name="gaia_agent",
)
def main(cfg: DictConfig) -> None:
    tasks = load_dataset(cfg.split)
    init_code_sandbox(cfg.exp_path)
    dt = time.perf_counter()
    n_workers = cfg.batch or 1
    tapes_dir = os.path.join(cfg.exp_path, "tapes")
    if len(cfg.only_tasks):
        tasks = cfg.only_tasks
    else:
        tasks = [
            (level, task_num)
            for level, level_tasks in tasks.items()
            for task_num, _ in enumerate(level_tasks)
            if not task_already_solved(task_num, level, tapes_dir, cfg.retry_unsolved)
        ]
    logger.info(f"Solve {len(tasks)} tasks using {n_workers} workers")
    Parallel(n_jobs=n_workers, prefer="processes")(
        [delayed(task_worker)(cfg, level, task_num) for level, task_num in tasks]
    )
    dt = time.perf_counter() - dt
    logger.info(f"Done, elapsed time: {dt:.2f} sec")


def validate_config(cfg, llm, tapes_dir):
    if os.path.exists(tapes_dir):
        old_exp_cfg = get_exp_config_dict(cfg.exp_path)
        assert (
            old_exp_cfg["llm"]["model_name"] == llm.model_name
        ), f"Exp dir model name mismatch: old {old_exp_cfg['llm']['model_name']}, new {llm.model_name}"
        assert old_exp_cfg["split"] == cfg.split, f"Exp split: old {old_exp_cfg['split']}, new {cfg.split}"
    os.makedirs(tapes_dir, exist_ok=True)


def task_already_solved(i: int, level: int, tapes_dir: str, retry_unsolved: bool) -> bool:
    tape_name = f"l{level}_task{i:03d}"
    tape_path = os.path.join(tapes_dir, f"{tape_name}.json")
    result = None
    tape_exists = os.path.exists(tape_path)
    solved = tape_exists
    if retry_unsolved and tape_exists:
        with open(tape_path) as f:
            tape_dict = json.load(f)
        result = tape_dict["metadata"]["result"]
        solved = result not in ["", None, "None", "none", "null"]
        if not solved:
            old_file_idx = 0
            while os.path.exists(f"{tape_path}.{old_file_idx}"):
                old_file_idx += 1
            os.rename(tape_path, f"{tape_path}.{old_file_idx}")

    return solved


def task_worker(cfg: DictConfig, level: int, task_num: int):
    os.makedirs(os.path.join(cfg.exp_path, "logs"), exist_ok=True)
    log_file = os.path.join(cfg.exp_path, "logs", f"evaluate.{os.getpid()}.log")
    log_handler = logging.FileHandler(log_file)
    log_handler.setLevel(logging.INFO)
    logging.basicConfig(
        format="%(asctime)s - PID_%(process)d - Thread_%(threadName)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[log_handler, logging.StreamHandler()],
        force=True,  # forget previous handlers
    )
    timers = {}

    t = time.perf_counter()
    tasks = load_dataset(cfg.split)
    task = tasks[level][task_num]
    timers["load_task"] = time.perf_counter() - t

    t = time.perf_counter()
    llm: TrainableLLM = instantiate(cfg.llm)
    timers["instantiate_llm"] = time.perf_counter() - t

    os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(cfg.exp_path, "tapedata.sqlite")
    tapes_dir = os.path.join(cfg.exp_path, "tapes")
    validate_config(cfg, llm, tapes_dir)
    images_dir = os.path.join(cfg.exp_path, "attachments", "images")
    os.makedirs(images_dir, exist_ok=True)

    t = time.perf_counter()
    agent, env = get_agent_and_env_from_config(cfg)
    timers["create_agent_env"] = time.perf_counter() - t

    tape = solve_task(task, agent, env, level, task_num, tapes_dir)

    t = time.perf_counter()
    env.close()
    timers["close_env"] = time.perf_counter() - t

    tape.metadata.other["timers"] |= timers
    save_json_tape(tape, tapes_dir, f"l{level}_task{task_num:03d}")
    save_tape_images(tape, images_dir)
    logger.info(f"Saved to {tapes_dir}")


if __name__ == "__main__":
    main()
