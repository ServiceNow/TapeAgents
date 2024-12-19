import logging
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from tapeagents.io import load_tapes
from tapeagents.renderers.camera_ready_renderer import CameraReadyRenderer
from tapeagents.studio import Studio

from ..environment import FormFillerEnvironment
from ..student import StudentAgent
from ..tape import FormFillerContext, FormFillerTape
from ..teacher import TeacherAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="run_formfiller_agent")
def main(cfg: DictConfig):
    llm = instantiate(cfg.llm)
    user_dialogues_path = cfg.user_dialogues_path

    if cfg.agent_type == "student":
        agent_type = StudentAgent
    elif cfg.agent_type == "teacher":
        agent_type = TeacherAgent
    else:
        raise ValueError(f"Unknown agent type: {cfg.agent_type}")

    logger.info(f"Agent type: {cfg.agent_type}")

    tapes = load_tapes(FormFillerTape, user_dialogues_path)

    agent = agent_type.create(llm, templates=cfg.templates)

    os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(".", "tapedata.sqlite")
    llm = instantiate(cfg.llm)
    tape = tapes[0]
    assert isinstance(tape.context, FormFillerContext)
    env = FormFillerEnvironment.from_spec(tape.context.env_spec)

    Studio(agent, tape, CameraReadyRenderer(), env).launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
