import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import hydra
import torch
from browsergym.miniwob import ALL_MINIWOB_TASKS
from omegaconf import DictConfig
from termcolor import colored

from examples.rl_webagent.utils import VLLMServiceManager
from tapeagents.core import Action
from tapeagents.io import save_json_tape
from tapeagents.llms import TrainableLLM
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

    tapes_dir = os.path.join(cfg.exp_path, "tapes")
    os.makedirs(tapes_dir, exist_ok=True)

    if os.path.exists(os.path.join(cfg.exp_path, "task_successes.json")):
        with open(os.path.join(cfg.exp_path, "task_successes.json"), "r") as f:
            task_successes = json.load(f)
    else:
        task_successes = defaultdict(list)

    with VLLMServiceManager(
        exp_path=Path(cfg.exp_path),
        service_name="actor",
        model_name_or_path=cfg.model_path,
        port=8080,
        verbose=True,
        cuda_device=",".join([str(i) for i in range(torch.cuda.device_count())]),
        **(dict(cfg.vllm_config.vllm_kwargs) | dict(cfg.vllm_config.actor_vllm_kwargs)),
    ) as vllm_service_manager:
        llms = [
            TrainableLLM(
                base_url=base_url,
                model_name=cfg.model_path,
                tokenizer_name=cfg.model_path,
                parameters=cfg.llm.parameters,
                use_cache=False,
                observe_llm_calls=False,
            )
            for base_url in vllm_service_manager.get_base_urls()
        ]
        env = WebEnvironment(**cfg.env)
        agent = WebAgent.create(llms[0])

        last_action = None
        repeated_action_cnt = 0
        for seed in cfg.seeds:
            for i, task in enumerate(ALL_MINIWOB_TASKS):
                task_name = f"task{i}_seed{seed}_{task.get_task_id()}"
                fname = f"{task_name}.json"
                if os.path.exists(os.path.join(tapes_dir, fname)):
                    logger.info(f"Skipping task {i+1}, already solved")
                    continue
                tmp_fpath = os.path.join(tapes_dir, f"{fname}.tmp")
                tape, metadata = env.start_task({"task": task, "seed": seed})
                metadata["seed"] = seed
                metadata["number"] = i
                logger.info(colored(f"Start task {i+1} seed {seed}: {metadata['goal']}", "cyan"))
                loop = 0
                logger.info(colored(f"Loop {loop+1}", "cyan"))
                try:
                    for event in main_loop(agent, tape, env, max_loops=20):  # type: ignore
                        if event.agent_event and event.agent_event.step:
                            step = event.agent_event.step
                            # avoid repeating the same action more than 4 times
                            if isinstance(step, Action):
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
                        tape.metadata.result = metadata
                        save_json_tape(tape, tmp_fpath)
                    success, result = env.validate_task(tape)
                    final_reward = 1.0 if success else 0.0
                    tape.metadata.result.update({"success": success, **result, "final_reward": final_reward})
                except Exception as e:
                    logger.exception(e, stack_info=True)
                    logger.error(f"Failed to run task: {e}")
                    tape.metadata.result.update({"success": False, "reward": 0.0, "final_reward": 0.0, "error": str(e)})

                env.finish_task()
                os.unlink(tmp_fpath)  # remove temporary file
                logger.info(colored(f"reward: {tape.metadata.result['reward']}", "cyan"))
                save_json_tape(tape, tapes_dir, fname)
                logger.info(f"Saved tape to {fname}")

                # save task_successes
                if task.get_task_id() not in task_successes:
                    task_successes[task.get_task_id()] = []
                task_successes[task.get_task_id()].append(tape.metadata.result["reward"])

                overall_r = sum([sum(task_successes[task_id]) for task_id in task_successes]) / sum(
                    [len(task_successes[task_id]) for task_id in task_successes]
                )
                logger.info(colored(f"average reward: {overall_r}", "cyan"))

                with open(os.path.join(cfg.exp_path, "task_successes.json"), "w") as f:
                    json.dump(task_successes, f)

                # save average reward
                with open(os.path.join(cfg.exp_path, "task_avg_reward.json"), "w") as f:
                    json.dump(
                        {task_id: sum(rewards) / len(rewards) for task_id, rewards in task_successes.items()},
                        f,
                    )
                logger.info(f"Saved stats to {cfg.exp_path}")


if __name__ == "__main__":
    main()
    # to do RL, check examples/rl_gsm8k/orchestrate_rl.py
    # then, instead of run_batch() I need to create my own version to support agent & environemnt batch interactions
    # for that I can take inspiration from examples/gaia_agent/scripts/evaluate.py and look at:
    # Parallel(n_jobs=n_workers, prefer="processes")(
    #     [delayed(task_worker)(cfg, level, task_num) for level, task_num in tasks]
    # )
