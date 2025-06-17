import json
import logging
import os

import hydra
from browsergym.miniwob import ALL_MINIWOB_TASKS
from hydra.utils import instantiate
from omegaconf import DictConfig
from termcolor import colored

from examples.rl_webagent.utils import StepRepeatMonitor
from tapeagents.core import Action
from tapeagents.io import save_json_tape
from tapeagents.llms import LLM
from tapeagents.observe import retrieve_llm_calls
from tapeagents.orchestrator import main_loop

from ..agent import WebAgent
from ..environment import WebEnvironment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../../conf",
    config_name="webagent_demo",
)
def main(cfg: DictConfig) -> None:
    os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(cfg.exp_path, "tapedata.sqlite")
    os.environ["MINIWOB_URL"] = cfg.environment_variables.miniwob_url
    # os.environ["SNOW_INSTANCE_URL"] = cfg.environment_variables.snow_instance_url
    # os.environ["SNOW_INSTANCE_UNAME"] = cfg.environment_variables.snow_instance_uname
    # os.environ["SNOW_INSTANCE_PWD"] = cfg.environment_variables.snow_instance_pwd

    tapes_dir = f"{cfg.exp_path}/tapes"
    os.makedirs(tapes_dir, exist_ok=True)

    task = ALL_MINIWOB_TASKS[1]
    seed = cfg.seeds[0]

    llm: LLM = instantiate(cfg.llm)
    env = WebEnvironment(**cfg.env)
    agent = WebAgent.create(llm)

    tape, metadata = env.start_task({"task": task, "seed": seed})
    metadata["seed"] = seed
    tape.metadata.result = metadata

    tape_name = f"debug_{task.get_task_id()}_seed{seed}"

    logger.info(colored(f"Start task {task.get_task_id()} seed {seed}: {metadata['goal']}", "cyan"))
    step_count = 0
    reapeated_action_monitor = StepRepeatMonitor(max_repeats=4)
    for event in main_loop(agent, tape, env, max_loops=50):
        if event.agent_event and event.agent_event.step:
            step = event.agent_event.step
            step_count += 1
            # avoid repeating the same action more than 4 times
            if isinstance(step, Action):
                step_view = step.llm_view()
                if reapeated_action_monitor.should_stop(step_view):
                    logger.error(
                        f"Repeated action '{step_view}' detected {reapeated_action_monitor.repeat_count} times"
                    )
                    break
            # print step and prompt messages
            llm_calls = retrieve_llm_calls(step.metadata.prompt_id)
            logger.info(f"{step_count} RUN {step.metadata.agent}:{step.metadata.node}")
            if llm_calls:
                for i, m in enumerate(llm_calls[0].prompt.messages):
                    logger.info(colored(f"PROMPT M{i + 1}: {json.dumps(m, indent=2)}", "red"))
            logger.info(f"{step_count} STEP of {step.metadata.agent}:{step.metadata.node}")
            logger.info(colored(step.llm_view(), "cyan"))
            input("Press Enter to continue...")
            print("-" * 140)
        elif event.observation:
            step = event.observation
            step_count += 1
            logger.info(colored(f"OBSERVATION: {step.kind}", "green"))
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
