import logging
import os

import hydra
from browsergym.workarena import ALL_WORKARENA_TASKS
from omegaconf import DictConfig
from termcolor import colored

from tapeagents.llms import LLM
from tapeagents.runtime import main_loop

from .agent import WorkArenaAgent, WorkArenaBaseline
from .environment import WorkArenaEnvironment
from .steps import WorkArenaAction

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../conf/tapeagent",
    config_name="workarena_openai",
)
def main(cfg: DictConfig) -> None:
    llm: LLM = hydra.utils.instantiate(cfg.llm)
    env = WorkArenaEnvironment(**cfg.env)
    if cfg.agent == "baseline":
        agent = WorkArenaBaseline.create(llm)
    else:
        logger.info("Use guided agent")
        agent = WorkArenaAgent.create(llm)
    tapes_dir = os.path.join(cfg.exp_path, "tapes")
    os.makedirs(tapes_dir, exist_ok=True)
    os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(cfg.exp_path, "tapedata.sqlite")
    last_action = None
    repeated_action_cnt = 0
    for seed in cfg.seeds:
        for i, task in enumerate(ALL_WORKARENA_TASKS):
            task_name = f"task{i}_seed{seed}_{task.get_task_id()}"
            fname = f"{task_name}.json"
            if os.path.exists(os.path.join(tapes_dir, fname)):
                logger.info(f"Skipping task {i+1}, already solved")
                continue
            tape, metadata = env.start_task(task, seed)
            metadata["seed"] = seed
            metadata["number"] = i
            logger.info(colored(f"Start task {i+1} seed {seed}: {metadata['goal']}", "cyan"))
            loop = 0
            logger.info(colored(f"Loop {loop+1}", "cyan"))
            for event in main_loop(agent, tape, env, max_loops=20):  # type: ignore
                if event.agent_event and event.agent_event.step:
                    step = event.agent_event.step
                    if isinstance(step, WorkArenaAction):
                        step_view = step.llm_view()
                        if step_view == last_action:
                            repeated_action_cnt += 1
                            if repeated_action_cnt > 4:
                                logger.error("Repeated action detected more than 4 time, stop the task")
                                break
                        else:
                            repeated_action_cnt = 0
                        last_action = step_view
                    tape = tape.append(step)  # type: ignore
                if event.observation:
                    tape = tape.append(event.observation)  # type: ignore
                    loop += 1
                    logger.info(colored(f"Loop {loop+1}", "cyan"))
                save_tape(tapes_dir, f"{fname}.tmp", tape, metadata)
            success, result = env.validate_task(tape)
            metadata["success"] = success
            metadata.update(result)
            env.finish_task(task_name)
            os.unlink(os.path.join(tapes_dir, f"{fname}.tmp"))  # remove temporary file
            save_tape(tapes_dir, fname, tape, metadata)
            logger.info(f"Saved tape to {fname}")


def save_tape(tapes_dir, fname, tape, metadata: dict):
    tape.metadata.result = metadata
    fpath = os.path.join(tapes_dir, fname)
    with open(fpath, "w") as f:
        f.write(tape.model_dump_json(indent=4))


if __name__ == "__main__":
    main()
