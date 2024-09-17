import logging
import os

from termcolor import colored

from tapeagents.examples.gaia_agent.agent import GaiaAgent
from tapeagents.examples.gaia_agent.environment import GaiaEnvironment
from tapeagents.examples.gaia_agent.eval import load_dataset, load_results
from tapeagents.examples.gaia_agent.tape import GaiaTape
from tapeagents.llms import ReplayLLM
from tapeagents.runtime import replay_tapes
from tapeagents.tools import BasicToolbox
from tapeagents.utils import diff_dicts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(fname: str, dataset_path: str = ""):
    """
    Replay Gaia tapes from the file with the results, compare the results with the original ones,
    and print the statistics of the replay.
    """
    results = load_results(fname)

    prompts = results.prompts
    model_name = results.model
    params = results.llm_config

    data_dir = dataset_path or results.dataset_path
    assert data_dir, "Dataset path is not provided"
    tasks = load_dataset(data_dir)
    llm = ReplayLLM(
        prompts=prompts,
        model_name=model_name,
        context_size=params.get("context_size", 32000),
    )
    vision_lm = None  # None or llm
    tools = BasicToolbox(vision_lm=vision_lm, only_cached_webpages=True, safe_calculator=False)
    tools._cache = results.web_cache
    env = GaiaEnvironment(tools)
    agent = GaiaAgent(llms={"default": llm}, short_steps=True)

    logger.info(f"Web Cache {len(results.web_cache)}")
    logger.info(f"Prompts {len(prompts)}")

    tapes = []
    for i, task in enumerate(tasks[1]):
        old_task = results.tapes[i]["metadata"]["task"]
        if old_task["file_name"] and os.path.basename(old_task["file_name"]) == os.path.basename(task["file_name"]):
            old_task["file_name"] = task["file_name"]
        if task != old_task:
            logger.info(colored(f"Task {i} mismatch", "red"))
            logger.info(diff_dicts(task, old_task))
            break
        tapes.append(GaiaTape.model_validate(results.tapes[i]))

    logger.info(f"Validate {len(tapes)} tapes")
    replay_tapes(agent, tapes, env, pause_on_error=True, reuse_observations=True)

    logger.info(f"Date {results.datetime}")
    logger.info(f"Commit {results.commit}")
    logger.info(f"Results {len(results.tapes)}")
    logger.info(f"Web Cache {len(results.web_cache)}")
    logger.info(f"Prompts {len(prompts)}")
    logger.info(f"LM {model_name}, params {params}")


if __name__ == "__main__":
    main(
        "../gaia/tapes/l1_gpt-4o-2024-05-13_2024-07-12_commonsteps5.json",
        "../gaia/dataset/validation/",
    )
