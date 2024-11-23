import logging
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from examples.gaia_agent.v2 import GaiaPlanner
from tapeagents.renderers.camera_ready_renderer import CameraReadyRenderer
from tapeagents.studio import Studio

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
    config_path="../../../conf",
    config_name="gaia_openai",
)
def main(cfg: DictConfig) -> None:
    os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(cfg.exp_path, "tapedata.sqlite")
    llm = instantiate(cfg.llm)
    env = GaiaEnvironment(vision_lm=llm)
    agent = GaiaPlanner.create(llm)
    tape = GaiaTape(steps=[GaiaQuestion(content="How many calories in 2 teaspoons of hummus")])
    Studio(agent, tape, CameraReadyRenderer(), env).launch(server_name="0.0.0.0", server_port=cfg.studio.port)


if __name__ == "__main__":
    main()
