import json
import logging
import os
import time

import hydra
from hydra.utils import instantiate
from joblib import Parallel, delayed
from omegaconf import DictConfig

from tapeagents.config import is_debug_mode
from tapeagents.io import save_json_tape, save_tape_images
from tapeagents.llms import TrainableLLM
from tapeagents.tools.container_executor import ContainerExecutor, maybe_get_code_sandbox

from ..agent import GaiaAgent
from ..environment import get_env
from ..eval import get_exp_config_dict, load_dataset, solve_task

logger = logging.getLogger(__name__)
if is_debug_mode():
    logging.basicConfig(level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)


@hydra.main(
    version_base=None,
    config_path="../../../conf",
    config_name="gaia_openai",
)
def main(cfg: DictConfig) -> None:
    """
    Solve Gaia tasks from the each level of the dataset, save the results to
    the separate files per level. If needed continue solving the unsolved tasks in the
    next run.
    """
    code_sandbox = maybe_get_code_sandbox(cfg.exp_path)
    tasks = load_dataset(cfg.split)
    dt = time.perf_counter()
    n_workers = cfg.batch or 0
    tapes_dir = os.path.join(cfg.exp_path, "tapes")
    tasks = [
        delayed(task_worker)(cfg, level, task_num, code_sandbox)
        for level, level_tasks in tasks.items()
        for task_num, _ in enumerate(level_tasks)
        if not task_already_solved(task_num, level, tapes_dir)
    ]
    logger.info(f"Evaluate {len(tasks)} unsolved tasks using {n_workers} workers")
    Parallel(n_jobs=n_workers, prefer="processes")(tasks)
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


def task_already_solved(i: int, level: int, tapes_dir: str) -> bool:
    tape_name = f"l{level}_task{i:03d}"
    tape_path = os.path.join(tapes_dir, f"{tape_name}.json")
    result = None
    if os.path.exists(tape_path):
        with open(tape_path) as f:
            tape_dict = json.load(f)
        result = tape_dict["metadata"]["result"]
    return os.path.exists(tape_path) and result not in ["", None, "None"]


def task_worker(cfg: DictConfig, level: int, task_num: int, code_sandbox: ContainerExecutor | None):
    tasks = load_dataset(cfg.split)
    task = tasks[level][task_num]
    llm: TrainableLLM = instantiate(cfg.llm)
    agent = GaiaAgent.create(llm, **cfg.agent)
    os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(cfg.exp_path, "tapedata.sqlite")
    tapes_dir = os.path.join(cfg.exp_path, "tapes")
    validate_config(cfg, llm, tapes_dir)
    images_dir = os.path.join(cfg.exp_path, "images")
    os.makedirs(images_dir, exist_ok=True)

    tapes_dir = os.path.join(cfg.exp_path, "tapes")
    images_dir = os.path.join(cfg.exp_path, "images")
    task_name = f"l{level}_task{task_num:03d}"
    env = get_env(cfg.exp_path, code_sandbox=code_sandbox, **cfg.env)

    for tape in solve_task(task, agent, env, level):
        save_json_tape(tape, tapes_dir, task_name)
        save_tape_images(tape, images_dir)
        logger.info(f"Task {task_name} solved, saved to {tapes_dir}")
    env.close()


if __name__ == "__main__":
    main()
