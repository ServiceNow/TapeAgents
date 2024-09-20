import logging
import os
import sys

from termcolor import colored

from tapeagents.core import LLMCall, LLMMessage, Prompt
from tapeagents.examples.gaia_agent.agent import GaiaAgent
from tapeagents.examples.gaia_agent.environment import GaiaEnvironment
from tapeagents.examples.gaia_agent.eval import load_dataset, load_results
from tapeagents.examples.gaia_agent.tape import GaiaTape
from tapeagents.llms import ReplayLLM
from tapeagents.runtime import replay_tapes
from tapeagents.utils import diff_dicts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(fname: str, dataset_path: str = ""):
    """
    Replay Gaia tapes from the file with the results, compare the results with the original ones,
    and print the statistics of the replay.
    """
    results = load_results(fname)

    prompts = results.prompts
    llm_calls = [
        LLMCall(prompt=Prompt.model_validate(prompt), completion=LLMMessage(), cached=False)
        for prompt in results.prompts
    ]
    model_name = results.model
    params = results.llm_config

    data_dir = dataset_path or results.dataset_path
    assert data_dir, "Dataset path is not provided"
    tasks = load_dataset(data_dir)
    llm = ReplayLLM(
        llm_calls=llm_calls,
        model_name=model_name,
        context_size=params.get("context_size", 32000),
    )
    env = GaiaEnvironment(only_cached_webpages=True)
    env.browser._cache = results.web_cache
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
    assert len(sys.argv) > 1, "Provide the path to the results file"
    main(
        sys.argv[1],
        "../gaia/dataset/validation/",
    )
