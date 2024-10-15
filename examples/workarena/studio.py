import logging
import os

import gradio as gr
import hydra
import nest_asyncio
import uvicorn
from browsergym.workarena import ALL_WORKARENA_TASKS
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from omegaconf import DictConfig

from tapeagents.llms import LLM
from tapeagents.studio import Studio

from .agent import WorkArenaAgent, WorkArenaBaseline
from .environment import WorkArenaEnvironment
from .tape_browser import WorkArenaRender

nest_asyncio.apply()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="workarena_demo",
)
def main(cfg: DictConfig) -> None:
    os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(cfg.exp_path, "tapedata.sqlite")
    llm: LLM = hydra.utils.instantiate(cfg.llm)
    env = WorkArenaEnvironment(**cfg.env)
    if cfg.agent == "baseline":
        agent = WorkArenaBaseline.create(llm)
    else:
        logger.info("Use guided agent")
        agent = WorkArenaAgent.create(llm)
    tape, _ = env.start_task(ALL_WORKARENA_TASKS[0], seed=cfg.seeds[0])
    blocks = Studio(agent, tape, WorkArenaRender(exp_dir=""), env).blocks
    logger.info(f"Starting FastAPI server with static dir {cfg.exp_path}")
    app = FastAPI()
    app.mount("/static", StaticFiles(directory=cfg.exp_path), name="static")
    app = gr.mount_gradio_app(app, blocks, path="/")
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
