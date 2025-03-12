import logging
import os

import hydra
from omegaconf import DictConfig

from tapeagents.config import ATTACHMENT_DEFAULT_DIR
from tapeagents.io import load_tapes
from tapeagents.orchestrator import get_agent_and_env_from_config
from tapeagents.renderers.camera_ready_renderer import CameraReadyRenderer
from tapeagents.studio import Studio
from tapeagents.tools.container_executor import init_code_sandbox

from ..steps import GaiaQuestion, GaiaTape

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../../conf",
    config_name="gaia_agent",
)
def main(cfg: DictConfig) -> None:
    os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(cfg.exp_path, "tapedata.sqlite")
    init_code_sandbox(cfg.exp_path)
    agent, env = get_agent_and_env_from_config(cfg)
    content = "How many calories in 2 teaspoons of hummus"
    if cfg.studio.tape:
        tape = load_tapes(GaiaTape, cfg.studio.tape, ".json")[0]
    else:
        # Uncomment the following line to test video question
        # content = "In the video https://www.youtube.com/watch?v=L1vXCYZAYYM, what is the highest number of bird species to be on camera simultaneously?"
        tape = GaiaTape(steps=[GaiaQuestion(content=content)])
    Studio(agent, tape, CameraReadyRenderer(), env).launch(
        server_name="0.0.0.0",
        static_dir=f"{cfg.exp_path}/{ATTACHMENT_DEFAULT_DIR}",
    )


if __name__ == "__main__":
    main()
