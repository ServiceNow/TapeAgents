import json
import logging
import os
import time

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from tapeagents.io import load_tapes, save_json_tape
from tapeagents.llms import TrainableLLM
from tapeagents.parallel_processing import lazy_thread_pool_processor

from ..agent import GaiaAgent
from ..environment import GaiaEnvironment
from ..eval import get_exp_config_dict, load_dataset, solve_task
from ..tape import GaiaTape

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


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
    llm: TrainableLLM = instantiate(cfg.llm)
    env = GaiaEnvironment(vision_lm=llm, **cfg.env)
    agent = GaiaAgent.create(llm, **cfg.agent)
    tasks = load_dataset(cfg.data_dir)
    tapes_dir = os.path.join(cfg.exp_path, "tapes")
    if os.path.exists(tapes_dir):
        old_exp_cfg = get_exp_config_dict(cfg.exp_path)
        assert (
            old_exp_cfg["llm"]["model_name"] == llm.model_name
        ), f"Exp dir model name mismatch: old {old_exp_cfg['llm']['model_name']}, new {llm.model_name}"
        assert (
            old_exp_cfg["data_dir"] == cfg.data_dir
        ), f"Exp dir data: old {old_exp_cfg['data_dir']}, new {cfg.data_dir}"
    os.makedirs(tapes_dir, exist_ok=True)

    browser_log_path = os.path.join(cfg.exp_path, "browser_log.jsonl")
    if os.path.exists(browser_log_path):
        with open(browser_log_path) as f:
            items = [json.loads(line) for line in f]
            for item in items:
                env.browser._add_to_cache(item["k"], item["v"])
            logger.info(f"Loaded {len(items)} cached queries from browser log")

    dt = time.perf_counter()
    for level, level_tasks in tasks.items():
        logger.info(f"Start level {level} with {len(level_tasks)} tasks")
        done = 0
        if cfg.batch > 1:
            logger.info(f"Using batch processing with {cfg.batch} workers")
            inputs = [
                (
                    agent,
                    env,
                    i,
                    task,
                    cfg.n_attempts,
                    tapes_dir,
                    browser_log_path,
                    level,
                )
                for i, task in enumerate(level_tasks)
            ]
            for tape_ready in lazy_thread_pool_processor(inputs, task_worker, cfg.batch):
                if isinstance(tape_ready, Exception):
                    raise tape_ready
                else:
                    done += tape_ready
        else:
            for i, task in enumerate(level_tasks):
                tape_ready = task_worker((agent, env, i, task, cfg.n_attempts, tapes_dir, browser_log_path, level))
                done += tape_ready
        logger.info(f"Level {level} done, {done} tapes produced")

    flush_browser_log(browser_log_path, env)
    dt = time.perf_counter() - dt
    logger.info(f"Done, elapsed time: {dt:.2f} sec")


def task_worker(inputs: tuple) -> int:
    agent, env, i, task, n_attempts, tapes_dir, browser_log_path, level = inputs
    tape_name = f"l{level}_task{i}"
    tape_path = os.path.join(tapes_dir, f"{tape_name}.json")
    if os.path.exists(tape_path):
        tape: GaiaTape = load_tapes(GaiaTape, tape_path)[0]  # type: ignore
        last_step = tape.steps[-1]
        model_answer = last_step.answer if last_step.kind == "gaia_answer_action" else None
        if model_answer:
            logger.info(f"Skip task {tape_name}, already solved")
            return 0
    tape = solve_task(task, agent, env, n_attempts)
    tape.metadata.level = level
    save_json_tape(tape, tapes_dir, tape_name)
    logger.info(f"Task {tape_name} solved, saved to {tape_path}")
    flush_browser_log(browser_log_path, env)
    return 1


def flush_browser_log(browser_log_path: str, env: GaiaEnvironment):
    with open(browser_log_path, "w") as wf:
        for k, v in env.browser._log.items():
            wf.write(json.dumps({"k": k, "v": v}) + "\n")
        env.browser._log = {}


if __name__ == "__main__":
    main()
