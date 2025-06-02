import logging
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from tapeagents.remote_environment import EnvironmentServer

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=".", config_name="env_server")
def main(cfg: DictConfig):
    os.environ["MINIWOB_URL"] = cfg.environment_variables.miniwob_url

    env_server: EnvironmentServer = instantiate(cfg.environment_server)
    env_server.launch(cfg.environment)


if __name__ == "__main__":
    main()
