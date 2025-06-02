import logging
import os

import gradio as gr
import hydra
import nest_asyncio
import uvicorn
from browsergym.miniwob import ALL_MINIWOB_TASKS
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from omegaconf import DictConfig

from tapeagents.llms import LLM
from tapeagents.studio import Studio

from ..agent import WebAgent
from ..environment import WebEnvironment
from .tape_browser import WebRender

nest_asyncio.apply()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../../conf",
    config_name="webagent_demo",
)
def main(cfg: DictConfig) -> None:
    os.makedirs(cfg.exp_path, exist_ok=True)
    os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(cfg.exp_path, "tapedata.sqlite")
    os.environ["MINIWOB_URL"] = cfg.environment_variables.miniwob_url
    # os.environ["SNOW_INSTANCE_URL"] = cfg.environment_variables.snow_instance_url
    # os.environ["SNOW_INSTANCE_UNAME"] = cfg.environment_variables.snow_instance_uname
    # os.environ["SNOW_INSTANCE_PWD"] = cfg.environment_variables.snow_instance_pwd

    llm: LLM = hydra.utils.instantiate(cfg.llm)
    env = WebEnvironment(**cfg.env)
    agent = WebAgent.create(llm)
    tape, _ = env.start_task({"task": ALL_MINIWOB_TASKS[0], "seed": cfg.seeds[0]})
    blocks = Studio(agent, tape, WebRender(exp_dir=""), env).blocks
    logger.info(f"Starting FastAPI server with static dir {cfg.exp_path}")
    app = FastAPI()
    app.mount("/static", StaticFiles(directory=cfg.exp_path), name="static")
    app = gr.mount_gradio_app(app, blocks, path="/")
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
