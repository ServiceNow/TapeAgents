import datetime
import logging
import sys

from omegaconf import DictConfig

from examples.form_filler.environment import FormFillerEnvironment
from tapeagents.agent import Agent
from tapeagents.core import Action, FinalStep, Observation, Tape
from tapeagents.environment import CodeExecutionEnvironment, Environment
from tapeagents.llms import LLM, LiteLLM
from tapeagents.orchestrator import main_loop
from tapeagents.renderers.basic import BasicRenderer
from tapeagents.renderers.camera_ready_renderer import CameraReadyRenderer
from tapeagents.renderers.pretty import PrettyRenderer
from tapeagents.team import TeamAgent, TeamTape
from tapeagents.tools.container_executor import ContainerExecutor
from tapeagents.view import Call, Respond

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def make_world(llm: LLM | None = None, env: Environment | None = None) -> tuple[Agent, Tape, Environment]:
    llm = llm or LiteLLM(model_name="gpt-4o", parameters={"timeout": 15.0})

    with open("config.yaml") as f:
        agent_config = DictConfig.load('reference_teacher.yaml') 
    agent = instantiate()
    env = FormFillerEnvironment.from_spec()


    start_tape = TeamTape(context=None, steps=[])
    now = f"{datetime.datetime.now():%Y%m%d%H%M%S}"
    env = env or CodeExecutionEnvironment(ContainerExecutor(work_dir=f"outputs/data_science/{now}"))
    return org, start_tape, env


def main(studio: bool):
    agent, start_tape, env = make_world()
    if studio:
        from tapeagents.studio import Studio

        Studio(agent, start_tape, make_renderers(), env).launch()
    else:
        final_tape = main_loop(agent, start_tape, env).get_final_tape()
        with open("final_tape.json", "w") as f:
            f.write(final_tape.model_dump_json(indent=2))    