import logging
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from tapeagents.io import load_tapes
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
    attachment_dir = cfg.get("env").get("attachment_dir")
    os.makedirs(attachment_dir, exist_ok=True)
    env = GaiaEnvironment(vision_lm=llm, attachment_dir=attachment_dir)
    agent = GaiaAgent.create(llm, **cfg.agent)
    content = "How many calories in 2 teaspoons of hummus"
    if cfg.studio.tape:
        tape = load_tapes(GaiaTape, cfg.studio.tape, ".json")[0]
    else:
        # Uncomment the following line to test video question
        # content = "In the video https://www.youtube.com/watch?v=L1vXCYZAYYM, what is the highest number of bird species to be on camera simultaneously?"
        tape = GaiaTape(steps=[GaiaQuestion(content=content)])
    Studio(agent, tape, CameraReadyRenderer(), env).launch(server_name="0.0.0.0", static_dir=attachment_dir)


if __name__ == "__main__":
    main()
