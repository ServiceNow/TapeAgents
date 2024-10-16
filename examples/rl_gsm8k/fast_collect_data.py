import copy
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable, Dict, List, Optional, TextIO, Tuple

import hydra
import numpy as np
import psutil
import requests
import torch
from datasets import load_dataset
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from termcolor import colored, cprint
from tqdm import tqdm

import wandb
from examples.rl_gsm8k.math_agent import (
    AnswerAction,
    MathAgent,
    MathEnvironment,
    MathTape,
    Task,
    extract_result_value,
    save_tape,
    solve_task,
)
from tapeagents.batch import batch_main_loop
from tapeagents.core import AgentResponseParsingFailureAction, StepMetadata, TapeMetadata, TrainingText
from tapeagents.finetune.finetune import load_config, run_finetuning_loop
from tapeagents.finetune.logging_ import flatten_dict_config, init_wandb
from tapeagents.llms import TrainableLLM

logger = logging.getLogger(__name__)


# Replace the existing logging setup with this:
def setup_logging(log_dir: Path):
    os.makedirs(log_dir, exist_ok=True)

    # Create formatters
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create handlers
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    debug_handler = RotatingFileHandler(log_dir / "debug.log", maxBytes=10 * 1024 * 1024, backupCount=5)
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(file_formatter)

    info_handler = RotatingFileHandler(log_dir / "info.log", maxBytes=10 * 1024 * 1024, backupCount=5)
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(file_formatter)

    error_handler = RotatingFileHandler(log_dir / "error.log", maxBytes=10 * 1024 * 1024, backupCount=5)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)

    # Configure root logger
    logger.setLevel(logging.DEBUG)  # Capture all levels
    logger.addHandler(console_handler)
    logger.addHandler(debug_handler)
    logger.addHandler(info_handler)
    logger.addHandler(error_handler)


def terminate_with_children(process_id: int) -> None:
    try:
        process = psutil.Process(process_id)
        children = process.children(recursive=True)
        # terminate child processes
        for child in children:
            child.terminate()
        _, still_alive = psutil.wait_procs(children, timeout=3)
        for child in still_alive:
            child.kill()

        # terminate parent process
        process.terminate()
        process.wait(timeout=3)
    except psutil.NoSuchProcess:
        print(f"No process found with PID: {process_id}")
    except psutil.AccessDenied:
        print(f"Insufficient privileges to terminate process with PID: {process_id}")
    except Exception as e:
        print(f"An error occurred: {e}")


def wait_for_service(process, url, headers=None, timeout=120):
    start_time = time.time()
    while True:
        try:
            response = requests.get(url, headers=headers, timeout=1)
            if response.status_code == 200:
                logger.info(f"-> Service at {url} is ready!")
                return True
        except requests.exceptions.RequestException as e:
            pass

        if time.time() - start_time > timeout:
            logger.error(f"-> Timeout waiting for service at {url}")
            return False

        if process.poll() is not None:
            logger.error(f"-> Process terminated while waiting for service at {url}")
            return False

        logger.info(f"-> Waiting for service at {url}")
        time.sleep(5)


def serve_vllm_local_model(
    model_name_or_path: str | Path,
    stdout_file_path: str | Path,
    stderr_file_path: str | Path,
    port: int = 8080,
    verbose: bool = True,
    cuda_device: str = "0",
    host: str = "localhost",
    **kwargs,
) -> Tuple[subprocess.Popen[str], Optional[TextIO], Optional[TextIO]]:
    tensor_parallel_size = cuda_device.count(",") + 1
    kwargs_str = ""
    if kwargs:
        kwargs_str = " ".join([f"{k} {v} " for k, v in kwargs.items()])
    cmd = (
        f"OUTLINES_CACHE_DIR=/tmp/.outlines_{cuda_device}_{int(time.time())} "  # outlines cache is a source of constant problem, see https://github.com/vllm-project/vllm/issues/4193
        f"CUDA_VISIBLE_DEVICES={cuda_device} "
        f"python -m vllm.entrypoints.openai.api_server "
        f"--model {model_name_or_path} "
        f"--tensor-parallel-size {tensor_parallel_size} "
        f"--port {port} "
        "--disable-frontend-multiprocessing "
        "--dtype bfloat16 "
        "--download-dir /mnt/llmd/base_models/ " + kwargs_str
    )
    if tensor_parallel_size > 1:  # needs spawn for vllm 0.6.1
        cmd = "VLLM_WORKER_MULTIPROC_METHOD=spawn " + cmd

    if verbose:
        cprint(f"Server launcher cmd: {cmd}", "light_green")

    process_args = {
        "shell": True,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }

    stdout_file = stderr_file = None
    if verbose:
        stdout_file = open(stdout_file_path, "w")
        stderr_file = open(stderr_file_path, "w")
        process_args["stdout"] = stdout_file
        process_args["stderr"] = stderr_file

    try:
        process = subprocess.Popen(cmd, **process_args)
    except Exception as e:
        print(f"Error occurred: {e}")
        raise TimeoutError(f"execution_timeout while waiting for {model_name_or_path} service to start")

    vllm_url = f"http://{host}:{port}/health"
    headers = {"User-Agent": "vLLM Client"}
    if wait_for_service(process, vllm_url, headers=headers, timeout=8000):
        cprint(f"Student {model_name_or_path} model loaded on port {port}", "magenta")
    else:
        cprint(
            f"execution_timeout while waiting for {model_name_or_path} service to start on port {port}",
            "magenta",
        )
        terminate_with_children(process.pid)  # type: ignore
        process.wait()  # type: ignore
        if stdout_file:
            stdout_file.close()
        if stderr_file:
            stderr_file.close()
        raise TimeoutError(f"execution_timeout while waiting for {model_name_or_path} service to start")

    return process, stdout_file, stderr_file


def load_state(state_path):
    if state_path.exists():
        with open(state_path, "r") as f:
            return json.load(f)
    else:
        return {"iteration": 0}


def save_state(state, state_path):
    with open(state_path, "w") as f:
        json.dump(state, f)


def clean_up(target_path: Path, state: Dict, state_path: str | Path) -> None:
    os.makedirs(target_path, exist_ok=True)

    def remove_dir(directory: Path):
        if directory.exists() and directory.is_dir():
            shutil.rmtree(directory)

    # Reset the state iteration steps
    state["iteration"] = 0
    save_state(state, state_path)

    logger.info("Cleaning up checkpoints and training state")

    # List of directories to remove
    directories = [
        target_path / "debug.log",
        target_path / "error.log",
        target_path / "info.log",
        target_path / "dialogue_trace.log",
        target_path / "rl_forks_train",
        target_path / "train",
        target_path / "rl_forks_eval",
        target_path / "eval",
        target_path / "finetune" / "current",
        target_path / "finetune" / "logs",
        target_path / "finetune" / "intermediate",
        target_path / "finetune" / "training_state",
    ]

    for directory in directories:
        remove_dir(directory)
        logger.info(f"{directory} removed.")


@hydra.main(config_path="../../conf/", config_name="fast_rl_gsm8k")
def main(cfg: DictConfig):
    random.seed(42)
    exp_path = Path(cfg.output_dir)
    setup_logging(exp_path)
    run = init_wandb(cfg, exp_path, flatten_dict_config(cfg))
    if run is None:
        raise ValueError("Failed to initialize wandb run")
    state_path = exp_path / "rl_state.json"
    state = load_state(state_path)
    # optionally clean all data at start time
    if cfg.force_restart:
        clean_up(exp_path, state, state_path)

    dataset = load_dataset("openai/gsm8k", "main", split="train")
    samples = [s for s in dataset]
    logging.info(f"Loaded {len(samples)} samples")

    env = MathEnvironment()

    tapes_dir = os.path.join(exp_path, "tapes")
    logger.info(f"Saving tapes to {tapes_dir}")
    os.makedirs(tapes_dir, exist_ok=True)
    os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(exp_path, "llm_calls.sqlite")
    conf_dir = exp_path / "conf"
    os.makedirs(conf_dir, exist_ok=True)
    finetune_path = exp_path / "finetune"
    while state["iteration"] < cfg.max_iterations:
        if os.path.exists(finetune_path / "current"):
            assistant_model_path = str(finetune_path / "current")
        else:
            assistant_model_path = cfg.model_path

        assistant_process, stdout_file, stderr_file = serve_vllm_local_model(
            model_name_or_path=assistant_model_path,
            stdout_file_path=exp_path / "assistant_vllm_stdout.log",
            stderr_file_path=exp_path / "assistant_vllm_stderr.log",
            port=8080,
            verbose=True,
            cuda_device=",".join([str(i) for i in range(torch.cuda.device_count())]),
        )

        llm = TrainableLLM(
            base_url="http://127.0.0.1:8080",
            model_name=str(assistant_model_path),
            tokenizer_name=str(assistant_model_path),
            parameters=cfg.llm.parameters,
            use_cache=False,
        )

        agent = MathAgent.create(llm=llm)

        rewards = []
        no_errors = []
        successes = []
        training_samples: list[TrainingText] = []
        start_make_training_data = time.time()
        reward_stats = defaultdict(list)
        step_stats = defaultdict(list)

        try:
            sub_samples = random.sample(samples, cfg.max_agent_forks // cfg.attempts)
            tapes = []
            for sample in sub_samples:
                start_step = Task(task=sample["question"], metadata=StepMetadata(other=extract_result_value(sample)))  # type: ignore
                tape = MathTape(steps=[start_step], context=None)
                tapes.append(tape)

            # tapes = tapes * cfg.attempts
            tapes = [copy.deepcopy(tape) for tape in tapes for _ in range(cfg.attempts)]
            new_tapes = []
            for new_tape in batch_main_loop(agent, tapes, env, max_loops=10):
                if any([isinstance(step, AgentResponseParsingFailureAction) for step in new_tape.steps]):
                    new_tape_filtered = copy.deepcopy(new_tape)
                    new_tape_filtered.steps = []
                    for step in new_tape.steps:
                        new_tape_filtered.steps.append(step)
                        if isinstance(step, AgentResponseParsingFailureAction):
                            break
                    no_error = 0
                    new_tape = new_tape_filtered
                    reward = -1
                    success = 0
                else:
                    no_error = 1
                    if (
                        isinstance(new_tape.steps[-1], AnswerAction)
                        and new_tape.steps[-1].value == new_tape.steps[0].metadata.other["value"]
                    ):
                        reward = 1
                        success = 1
                    else:
                        reward = 0
                        success = 0

                reward_stats[new_tape.metadata.parent_id].append(reward)
                step_stats[new_tape.metadata.parent_id].append(len(new_tape.steps))
                new_tapes.append(new_tape)
                rewards.append(reward)
                no_errors.append(no_error)
                successes.append(success)

                try:
                    for trace in agent.make_training_data(new_tape):
                        trace.rewards = [reward]
                        trace.fork_id = new_tape.metadata.parent_id
                        training_samples.append(trace)
                except Exception as e:
                    logger.error(colored(f"Failed to make training data: {e}", "red"))
                    logger.error(new_tape)

            max_rewards = np.mean([max(stats) for stats in reward_stats.values() if stats])
            min_rewards = np.mean([min(stats) for stats in reward_stats.values() if stats])
            var_rewards = np.mean([np.var(stats) for stats in reward_stats.values() if stats])

            max_steps = np.mean([max(stats) for stats in step_stats.values() if stats])
            min_steps = np.mean([min(stats) for stats in step_stats.values() if stats])
            var_steps = np.mean([np.var(stats) for stats in step_stats.values() if stats])
            mean_steps = np.mean([np.mean(stats) for stats in step_stats.values() if stats])

        except Exception as e:
            logger.error(colored(f"Failed to solve task: {e}", "red"))
            raise e
        finally:
            terminate_with_children(assistant_process.pid)
            assistant_process.wait()  # type: ignore
            if stdout_file:
                stdout_file.close()
            if stderr_file:
                stderr_file.close()

        end_make_training_data = time.time()

        logger.info(f"Collected {len(training_samples)} training samples")
        logger.info(f"Rewards: {np.mean(rewards)}")
        wandb.log(
            {
                "rewards": np.mean(rewards),
                "max_rewards": max_rewards,
                "min_rewards": min_rewards,
                "var_rewards": var_rewards,
                "steps": mean_steps,
                "max_steps": max_steps,
                "min_steps": min_steps,
                "var_steps": var_steps,
                "no_error": np.mean(no_errors),
                "success": np.mean(successes),
                "execution_time/make_training_data": end_make_training_data - start_make_training_data,
            },
            step=state["iteration"],
        )

        start_basemodel_logprobs = time.time()
        new_training_samples: list[TrainingText] = []
        if assistant_model_path == cfg.model_path:
            for trace in training_samples:
                trace.ref_logprobs = trace.old_logprobs
                new_training_samples.append(trace)
        else:
            base_model_process, basemodel_stdout_file, basemodel_stderr_file = serve_vllm_local_model(
                model_name_or_path=cfg.model_path,
                stdout_file_path=exp_path / "basemodel_vllm_stdout.log",
                stderr_file_path=exp_path / "basemodel_vllm_stderr.log",
                port=8080,
                verbose=True,
                cuda_device=",".join([str(i) for i in range(torch.cuda.device_count())]),
            )
            try:
                basemodel_llm = TrainableLLM(
                    base_url="http://127.0.0.1:8080",
                    model_name=cfg.model_path,
                    tokenizer_name=cfg.model_path,
                    parameters=dict(temperature=0.7),
                )

                basemodel_agent = MathAgent.create(llm=basemodel_llm)
                for trace in training_samples:
                    try:
                        trace.ref_logprobs = basemodel_agent.llm.get_log_probs(trace.prompt_text, trace.output_text)
                    except Exception as e:
                        logger.error(colored(f"Failed to get ref log probs: {e}", "red"))
                        continue
                    new_training_samples.append(trace)

            except Exception as e:
                logger.error(colored(f"Failed to get ref log probs: {e}", "red"))
                raise e
            finally:
                # Make sure the weights of the model to be trained are not loaded on the GPU
                terminate_with_children(base_model_process.pid)
                base_model_process.wait()
                if basemodel_stdout_file:
                    basemodel_stdout_file.close()
                if basemodel_stderr_file:
                    basemodel_stderr_file.close()
        end_basemodel_logprobs = time.time()
        wandb.log(
            {
                "execution_time/basemodel_logprobs": end_basemodel_logprobs - start_basemodel_logprobs,
            },
            step=state["iteration"],
        )
        rollout_dir = exp_path / "rollouts" / str(state["iteration"])
        os.makedirs(rollout_dir, exist_ok=True)
        with open(rollout_dir / "data.jsonl", "w") as f:
            for trace in training_samples:
                f.write(trace.model_dump_json() + "\n")
                f.flush()

        finetune_cfg = cfg.copy()

        finetune_cfg.finetune.save_checkpoint_steps = max(
            cfg.max_agent_forks // (cfg.finetune.train_batch_size * cfg.finetune.gradient_accumulation_passes),
            1,
        )
        interrupt_train_steps = int((state["iteration"] + 1) * finetune_cfg.finetune.save_checkpoint_steps)
        finetune_cfg.finetune.interrupt_train_steps = interrupt_train_steps
        finetune_cfg.output_dir = str(finetune_path)
        finetune_cfg.finetune.data = {"data_parts_train": [{"path": str(rollout_dir)}]}
        finetune_cfg.finetune.wandb_id = run.id + "_finetune"
        finetune_cfg.finetune.wandb_name = run.name + "_finetune"
        finetune_cfg.finetune.wandb_resume = "always"
        config_path = conf_dir / f"{state['iteration']}.yaml"
        OmegaConf.save(finetune_cfg, config_path)

        finetune_command = (
            "cd /home/toolkit/TapeAgents/tapeagents && "
            "PYTHONPATH=/home/toolkit/TapeAgents:/home/toolkit/TapeAgents/tapeagents/src "
            "conda run -n tapeagents --no-capture-output "
            f"accelerate launch --mixed_precision=bf16 --num_processes {str(torch.cuda.device_count())} "
            "--config_file /home/toolkit/TapeAgents/conf/deepspeed/accelerate_local.yaml "
            f"run_finetune.py --config-dir {str(conf_dir)} "
            f"--config-name {str(state['iteration'])} hydra.run.dir={str(finetune_path)}"
        )
        logger.info(f"Executing finetune command: {finetune_command}")

        start_finetune = time.time()
        error_code = subprocess.call(finetune_command, shell=True)
        end_finetune = time.time()

        if error_code != 0:
            logger.error(f"Finetuning failed with error code {error_code}")
            sys.exit(1)

        wandb.log(
            {
                "execution_time/finetune": end_finetune - start_finetune,
            },
            step=state["iteration"],
        )
        state["iteration"] += 1
        save_state(state, state_path)


if __name__ == "__main__":
    main()