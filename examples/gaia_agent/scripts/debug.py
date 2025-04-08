import json
import logging
import os

import hydra
from omegaconf import DictConfig

from tapeagents.io import save_json_tape
from tapeagents.observe import retrieve_llm_calls
from tapeagents.orchestrator import get_agent_and_env_from_config, main_loop

from ..eval import load_dataset, task_to_observations
from ..steps import GaiaMetadata, GaiaTape

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../../conf",
    config_name="agent_debug",
)
def main(cfg: DictConfig) -> None:
    dset = load_dataset("validation")
    tapes_dir = f"{cfg.exp_path}/tapes"
    os.makedirs(tapes_dir, exist_ok=True)
    os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(cfg.exp_path, "tapedata.sqlite")
    level, task = cfg.only_tasks[0]
    tape_name = f"debug_{level}_{task}"
    tasks = dset[level]
    task = tasks[task]
    agent, env = get_agent_and_env_from_config(cfg)
    tape = GaiaTape(steps=task_to_observations(task))
    tape.metadata = GaiaMetadata.model_validate(tape.metadata.model_dump() | {"task": task, "level": level})
    step_count = 0
    for event in main_loop(agent, tape, env, max_loops=50):
        if event.agent_event and event.agent_event.step:
            step = event.agent_event.step
            step_count += 1
            llm_calls = retrieve_llm_calls(step.metadata.prompt_id)
            logger.info(f"{step_count} RUN {step.metadata.agent}:{step.metadata.node}")
            if llm_calls:
                for i, m in enumerate(llm_calls[0].prompt.messages):
                    logger.info(f"PROMPT M{i+1}: {json.dumps(m, indent=2)}")
            logger.info(f"{step_count} STEP of {step.metadata.agent}:{step.metadata.node}")
            logger.info(step.llm_view())
            input("Press Enter to continue...")
            print("-" * 140)
        elif event.observation:
            step = event.observation
            step_count += 1
            logger.info(f"OBSERVATION: {step.kind}")
            input("Press Enter to continue...")
            print("-" * 140)
        elif new_tape := (event.agent_tape or event.env_tape):
            tape = new_tape
            save_json_tape(tape, tapes_dir, tape_name)
            logger.info(f"Saved tape to {tapes_dir}/{tape_name}.json")
        elif event.agent_event and event.agent_event.final_tape is not None:
            logger.info("RUN END")
        elif event.env_tape is not None:
            logger.info("ENV END")
        else:
            logger.info(f"EVENT: {event.status}")

    save_json_tape(tape, tapes_dir, tape_name)
    logger.info(f"Saved tape to {tapes_dir}/{tape_name}.json")


if __name__ == "__main__":
    main()
