import asyncio
import logging

import aiohttp
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from tapeagents.remote_environment import AsyncRemoteEnvironment
from tapeagents.tools.browser import OpenUrlAction

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=".", config_name="env_client")
def main(cfg: DictConfig):
    asyncio.run(amain(cfg))


async def amain(cfg: DictConfig):
    environment: AsyncRemoteEnvironment = instantiate(cfg.environment)
    async with aiohttp.ClientSession() as session:
        async with environment.acontext(session) as env:
            logger.info("Environment ready.")
            actions = await env.a_actions()
            logger.info(f"Available actions: {actions}")
            action = OpenUrlAction(url="https://servicenow.com")
            obs = await env.astep(action)
            logger.info(f"Observation after step: {type(obs)}: {obs}")
            raise Exception("This is a test exception to test env closing.")


if __name__ == "__main__":
    print("Make sure that you run environment server first using `uv run examples/workarena/env_server.py`")
    main()
