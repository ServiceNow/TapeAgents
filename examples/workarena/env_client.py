import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from tapeagents.remote_environment import RemoteEnvironment
from tapeagents.tools.browser import OpenUrlAction

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=".", config_name="env_client")
def main(cfg: DictConfig):
    env: RemoteEnvironment = instantiate(cfg.environment)
    env.reset()
    logger.info("Environment reset complete")
    actions = env.actions()
    logger.info(f"Available actions: {actions}")
    action = OpenUrlAction(url="https://servicenow.com")
    obs = env.step(action)
    logger.info(f"Observation after step: {type(obs)}: {obs}")


if __name__ == "__main__":
    main()
