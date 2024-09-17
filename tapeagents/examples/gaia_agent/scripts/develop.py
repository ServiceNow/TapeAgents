import datetime
import json
import logging
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from termcolor import colored

from tapeagents.develop import Develop
from tapeagents.examples.gaia_agent.agent import GaiaAgent
from tapeagents.examples.gaia_agent.environment import GaiaEnvironment
from tapeagents.examples.gaia_agent.eval import GaiaResults, load_dataset, load_results, save_results, solve_task
from tapeagents.examples.gaia_agent.steps import GaiaQuestion
from tapeagents.examples.gaia_agent.tape import GaiaTape
from tapeagents.examples.ghreat.critic import CriticTape
from tapeagents.llms import CachedLLM
from tapeagents.rendering import TapeBrowserRenderer
from tapeagents.tools import BasicToolbox

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

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
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    llm: CachedLLM = instantiate(cfg.llm)
    vision_lm = llm
    tools = BasicToolbox(vision_lm=vision_lm)
    env = GaiaEnvironment(tools)
    agent = GaiaAgent.create(llm, config=instantiate(cfg.agent_config))
    agent.max_iterations = 10
    tape = GaiaTape(steps=[GaiaQuestion(content="How many calories in 2 teaspoons of hummus")])
    Develop(agent, tape, TapeBrowserRenderer(), env).launch()


if __name__ == "__main__":
    main()
