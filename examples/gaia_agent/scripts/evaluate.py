import json
import logging
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from termcolor import colored

from tapeagents.llms import CachedLLM

from ..agent import GaiaAgent
from ..environment import GaiaEnvironment
from ..eval import GaiaResults, load_dataset, load_results, save_results, solve_task

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../../conf/tapeagent",
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

    if cfg.task_id is not None:
        logger.info(f"Solve only task {cfg.task_id} from the level 1")
        task = tasks[1][cfg.task_id - 1]
        solve_task(task, agent, env, cfg.n_attempts)
        return

    for i, level in tasks.items():
        results = GaiaResults()
        os.makedirs(cfg.exp_path, exist_ok=True)
        os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(cfg.exp_path, "tapedata.sqlite")
        outfile = os.path.join(cfg.exp_path, f"l{i}_{llm.model_name.split('/')[-1]}_run.json")
        logger.info(f"Start level {i} with {len(level)} tasks, save to {outfile}")
        if os.path.exists(outfile):
            results = load_results(outfile)
            env.browser._cache |= results.web_cache
            llm._log = results.prompts
            llm.reindex_log()
            logger.info(f"Loaded previous solutions for {len(results.tapes)} tasks, continue")

        unsolved_tasks = level[len(results.tapes) :]
        for task in unsolved_tasks:
            tape = solve_task(task, agent, env, cfg.n_attempts)
            results.tapes.append(tape.model_dump())
            results.prompts = llm._log
            results.web_cache |= env.browser._log
            save_results(results, outfile)
            logger.info(f"Saved {len(results.tapes)} tapes to {outfile}")
        logger.info(f"Level {i} done, {len(results.tapes)} tapes saved")
    logger.info("Done")


if __name__ == "__main__":
    main()
