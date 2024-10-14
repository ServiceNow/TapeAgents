import json
import logging
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from tapeagents.io import load_tapes, save_json_tape
from tapeagents.llms import TrainableLLM

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

    for level, level_tasks in tasks.items():
        outfile = os.path.join(cfg.exp_path, f"l{level}_{llm.model_name.split('/')[-1]}_run.json")
        logger.info(f"Start level {level} with {len(level_tasks)} tasks, save to {outfile}")
        new_tapes = []
        for i, task in enumerate(level_tasks):
            tape_name = f"l{level}_task{i}"
            tape_path = os.path.join(tapes_dir, f"{tape_name}.json")
            if os.path.exists(tape_path):
                tape: GaiaTape = load_tapes(GaiaTape, tape_path)[0]  # type: ignore
                last_step = tape.steps[-1]
                model_answer = last_step.answer if last_step.kind == "gaia_answer_action" else None
                if model_answer:
                    logger.info(f"Skip task {tape_name}, already solved")
                    continue
            tape = solve_task(task, agent, env, cfg.n_attempts)
            new_tapes.append(tape)
            save_json_tape(tape, tapes_dir, tape_name)
            logger.info(f"Task {tape_name} solved, saved to {tape_path}")
            flush_browser_log(browser_log_path, env)
        logger.info(f"Level {level} done, {len(new_tapes)} tapes saved")

    flush_browser_log(browser_log_path, env)
    logger.info("Done")


def flush_browser_log(browser_log_path: str, env: GaiaEnvironment):
    with open(browser_log_path, "w") as wf:
        for k, v in env.browser._log.items():
            wf.write(json.dumps({"k": k, "v": v}) + "\n")
        env.browser._log = {}


if __name__ == "__main__":
    main()
