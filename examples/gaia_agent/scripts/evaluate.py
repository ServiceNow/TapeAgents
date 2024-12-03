import json
import logging
import os
import time

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from tapeagents.config import is_debug_mode
from tapeagents.io import save_json_tape, save_tape_images
from tapeagents.llms import TrainableLLM
from tapeagents.parallel_processing import choose_processor
from tapeagents.tools.container_executor import ContainerExecutor

from ..agent import GaiaAgent
from ..environment import GaiaEnvironment
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
    os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(cfg.exp_path, "tapedata.sqlite")
    code_path = os.path.join(cfg.exp_path, "code")
    os.makedirs(code_path, exist_ok=True)
    os.makedirs(cfg.env.attachment_dir, exist_ok=True)
    llm: TrainableLLM = instantiate(cfg.llm)
    try:
        code_sandbox = ContainerExecutor(work_dir=os.path.join(cfg.exp_path, "code"))
    except Exception as e:
        logger.error(f"Failed to create code sandbox: {e}")
        code_sandbox = None
    agent = GaiaAgent.create(llm, **cfg.agent)
    tasks = load_dataset(cfg.split)
    tapes_dir = os.path.join(cfg.exp_path, "tapes")
    validate_config(cfg, llm, tapes_dir)
    images_dir = os.path.join(cfg.exp_path, "images")
    os.makedirs(images_dir, exist_ok=True)

    dt = time.perf_counter()
    n_workers = cfg.batch or 0
    processor = choose_processor(n_workers)
    args = [
        (agent, llm, cfg.env, code_sandbox, task, cfg.exp_path, i, level)
        for level, level_tasks in tasks.items()
        for i, task in enumerate(level_tasks)
        if not task_already_solved(i, level, tapes_dir)
    ]
    if cfg.get("n_tasks"):
        args = args[: cfg.n_tasks]  # run only the first n_tasks
    logger.info(f"Evaluate {len(args)} unsolved tasks using {n_workers} workers")
    for tape_ready in processor(args, task_worker):
        if isinstance(tape_ready, Exception):
            raise tape_ready
        if is_debug_mode():
            break
    dt = time.perf_counter() - dt
    logger.info(f"Done, elapsed time: {dt:.2f} sec")
    # if code_sandbox:
    #    code_sandbox.stop()


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
    return os.path.exists(tape_path) and result not in ["", None]


def task_worker(args: tuple) -> int:
    agent, llm, cfg_env, code_sandbox, task, exp_path, i, level = args
    tapes_dir = os.path.join(exp_path, "tapes")
    images_dir = os.path.join(exp_path, "images")
    tape_name = f"l{level}_task{i:03d}"
    env = GaiaEnvironment(vision_lm=llm, code_sandbox=code_sandbox, **cfg_env)

    tape = solve_task(task, agent, env, level)
    save_json_tape(tape, tapes_dir, tape_name)
    save_tape_images(tape, images_dir)
    logger.info(f"Task {tape_name} solved, saved to {tapes_dir}")
    env.browser.flush_log(os.path.join(exp_path, "browser_log.jsonl"))
    return 1


if __name__ == "__main__":
    main()
