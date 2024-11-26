import json
import logging
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from tapeagents.container_executor import ContainerExecutor
from tapeagents.io import save_json_tape
from tapeagents.llms import LiteLLM, TrainableLLM
from tapeagents.observe import retrieve_llm_call
from tapeagents.orchestrator import main_loop

from ..environment import GaiaEnvironment
from ..eval import load_dataset, task_to_question_step
from ..tape import GaiaTape
from ..v2 import GaiaPlanner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../../conf",
    config_name="gaia_openai",
)
def main(cfg: DictConfig) -> None:
    dset = load_dataset(cfg.data_dir)
    tapes_dir = f"{cfg.exp_path}/tapes"
    os.makedirs(tapes_dir, exist_ok=True)
    os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(cfg.exp_path, "tapedata.sqlite")
    tape_name = f"debug_{cfg.level}_{cfg.task}"
    tasks = dset[cfg.level]
    task = tasks[cfg.task]
    llm: TrainableLLM = instantiate(cfg.llm)
    try:
        code_sandbox = ContainerExecutor(work_dir=os.path.join(cfg.exp_path, "code"))
    except Exception as e:
        logger.error(f"Failed to create code sandbox: {e}")
        code_sandbox = None
    env = GaiaEnvironment(vision_lm=llm, code_sandbox=code_sandbox, safe_calculator=False)
    planner = GaiaPlanner.create(llm)
    tape = GaiaTape(steps=[task_to_question_step(task, env)])
    metadata = tape.metadata
    metadata.task = task
    metadata.level = cfg.level
    for event in main_loop(planner, tape, env, max_loops=50):
        if event.agent_event and event.agent_event.step:
            step = event.agent_event.step
            tape = tape.append(step)
            save_json_tape(tape, tapes_dir, tape_name)
            llm_call = retrieve_llm_call(step.metadata.prompt_id)
            logger.info(f"{len(tape)} RUN {step.metadata.agent}:{step.metadata.node}")
            if llm_call:
                for i, m in enumerate(llm_call.prompt.messages):
                    logger.info(f"PROMPT M{i+1}: {json.dumps(m, indent=2)}")
            logger.info(f"{len(tape)} STEP of {step.metadata.agent}:{step.metadata.node}")
            logger.info(step.llm_view())
            # input("Press Enter to continue...")
            print("-" * 140)
        elif event.observation:
            step = event.observation
            tape = tape.append(step)
            save_json_tape(tape, tapes_dir, tape_name)
            logger.info(f"OBSERVATION: {step.kind}")
            # input("Press Enter to continue...")
            print("-" * 140)
        elif event.agent_event and event.agent_event.final_tape is not None:
            logger.info("RUN END")
        elif event.env_tape is not None:
            logger.info("ENV END")
        else:
            logger.info(f"EVENT: {event.status}")

    tape.metadata = metadata
    save_json_tape(tape, tapes_dir, tape_name)
    logger.info(f"Saved tape to {tapes_dir}/{tape_name}.json")

    if code_sandbox:
        code_sandbox.stop()


if __name__ == "__main__":
    main()
