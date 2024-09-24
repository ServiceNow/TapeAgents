import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from tapeagents.studio import Studio
from tapeagents.llms import CachedLLM
from tapeagents.rendering import TapeBrowserRenderer

from ..agent import GaiaAgent
from ..environment import GaiaEnvironment
from ..steps import GaiaQuestion
from ..tape import GaiaTape

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
    llm: CachedLLM = instantiate(cfg.llm)
    vision_lm = llm
    env = GaiaEnvironment(vision_lm=vision_lm)
    agent = GaiaAgent.create(llm, config=instantiate(cfg.agent_config))
    agent.max_iterations = 10
    tape = GaiaTape(steps=[GaiaQuestion(content="How many calories in 2 teaspoons of hummus")])
    Studio(agent, tape, TapeBrowserRenderer(), env).launch()


if __name__ == "__main__":
    main()
