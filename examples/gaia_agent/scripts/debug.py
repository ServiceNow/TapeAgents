import json
import logging
import os

from tapeagents.io import save_json_tape
from tapeagents.llms import LiteLLM
from tapeagents.observe import retrieve_llm_calls
from tapeagents.orchestrator import main_loop

from ..agent import GaiaAgent
from ..environment import GaiaEnvironment
from ..eval import load_dataset, task_to_question_step
from ..tape import GaiaTape
from ..v2 import GaiaPlanner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(exp_dir, level, task_num):
    dset = load_dataset("validation")
    tapes_dir = f"{exp_dir}/tapes"
    os.makedirs(tapes_dir, exist_ok=True)
    os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(exp_dir, "tapedata.sqlite")
    tape_name = f"debug_{level}_{task_num}"
    tasks = dset[level]
    task = tasks[task_num]
    llm = LiteLLM(
        model_name="gpt-4o-mini-2024-07-18",
        context_size=128000,
        use_cache=True,
        parameters={"temperature": 0.0},
    )
    env = GaiaEnvironment(vision_lm=llm, safe_calculator=False)
    # agent = GaiaPlanner.create(llm)
    agent = GaiaAgent.create(llm)
    tape = GaiaTape(steps=[task_to_question_step(task, env)])
    metadata = tape.metadata
    metadata.task = task
    metadata.level = level
    for event in main_loop(agent, tape, env, max_loops=50):
        if event.agent_event and event.agent_event.step:
            step = event.agent_event.step
            tape = tape.append(step)
            save_json_tape(tape, tapes_dir, tape_name)
            llm_calls = retrieve_llm_calls(step.metadata.prompt_id)
            logger.info(f"{len(tape)} RUN {step.metadata.agent}:{step.metadata.node}")
            if llm_calls:
                for i, m in enumerate(llm_calls[0].prompt.messages):
                    logger.info(f"PROMPT M{i+1}: {json.dumps(m, indent=2)}")
            logger.info(f"{len(tape)} STEP of {step.metadata.agent}:{step.metadata.node}")
            logger.info(step.llm_view())
            input("Press Enter to continue...")
            print("-" * 140)
        elif event.observation:
            step = event.observation
            tape = tape.append(step)
            save_json_tape(tape, tapes_dir, tape_name)
            logger.info(f"OBSERVATION: {step.kind}")
            input("Press Enter to continue...")
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


if __name__ == "__main__":
    exp_dir = "outputs/gaia/runs/old_debug1/"
    level = 1
    task = 1
    main(exp_dir, level, task)
