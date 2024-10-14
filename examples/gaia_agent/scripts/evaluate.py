import json
import logging
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from termcolor import colored

from examples.gaia_agent.steps import GaiaAnswer
from examples.gaia_agent.tape import GaiaTape
from tapeagents.io import save_json_tape
from tapeagents.llms import CachedLLM

from ..agent import GaiaAgent
from ..environment import GaiaEnvironment
from ..eval import GaiaResults, load_dataset, load_results, save_results, solve_task

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
    llm: CachedLLM = instantiate(cfg.llm)
    env = GaiaEnvironment(vision_lm=llm, **cfg.env)
    if cfg.load_webcache_from_run:
        with open(cfg.load_webcache_from_run) as f:
            old_results_dict = json.load(f)
            webcache = old_results_dict["web_cache"]
        env.browser._cache |= webcache
        logger.info(
            colored(
                f"Updated webcache with {len(webcache)} items from the old result. New cache size {len(env.browser._cache)}",
                "yellow",
            )
        )
    agent = GaiaAgent.create(llm, **cfg.agent)
    tasks = load_dataset(cfg.data_dir)
    tapes_dir = os.path.join(cfg.exp_path, "tapes")
    os.makedirs(tapes_dir, exist_ok=True)
    if cfg.task_id is not None:
        logger.info(f"Solve only task {cfg.task_id} from the level 1")
        task = tasks[1][cfg.task_id - 1]
        solve_task(task, agent, env, cfg.n_attempts)
        return

    for level, level_tasks in tasks.items():
        results = GaiaResults()
        os.makedirs(cfg.exp_path, exist_ok=True)
        os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(cfg.exp_path, "tapedata.sqlite")
        outfile = os.path.join(cfg.exp_path, f"l{level}_{llm.model_name.split('/')[-1]}_run.json")
        logger.info(f"Start level {level} with {len(level_tasks)} tasks, save to {outfile}")
        solved = {}
        if os.path.exists(outfile):
            results = load_results(outfile)
            env.browser._cache |= results.web_cache
            llm._log = results.prompts
            llm.reindex_log()
            logger.info(f"Loaded previous solutions for {len(results.tapes)} tasks, continue")
            for i in range(len(level_tasks)):
                old_tape = results.tapes[i] if i < len(results.tapes) else None
                if not old_tape:
                    continue
                last_step = old_tape["steps"][-1]
                predicted = last_step["answer"] if last_step["kind"] == "gaia_answer_action" else None
                if predicted:
                    solved[i] = old_tape
                    save_json_tape(GaiaTape.model_validate(old_tape), tapes_dir, f"l{level}_task{i}")
        logger.info(f"Already solved {len(solved)} tasks out of {len(level_tasks)}")

        tapes = []
        for i, task in enumerate(level_tasks):
            if i in solved:
                logger.info(f"Task L{level}:{i + 1} already solved, skip")
                tape = solved[i]
                tapes.append(tape)
            else:
                tape = solve_task(task, agent, env, cfg.n_attempts)
                save_json_tape(tape, tapes_dir, f"l{level}_task{i}")
                tapes.append(tape.model_dump())
            results.tapes = tapes
            results.prompts = llm._log
            results.web_cache |= env.browser._log
            save_results(results, outfile)
            logger.info(f"Saved {len(results.tapes)} tapes to {outfile}")
        logger.info(f"Level {level} done, {len(results.tapes)} tapes saved")
    logger.info("Done")


if __name__ == "__main__":
    main()
