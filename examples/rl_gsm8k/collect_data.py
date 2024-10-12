import json
import logging
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, TextIO, Tuple
import wandb

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

from examples.rl_gsm8k.math_agent import (
    MathAgent,
    MathEnvironment,
    extract_result_value,
    save_tape,
    solve_task,
)
from tapeagents.core import TrainingText
from tapeagents.finetune.finetune import load_config, run_finetuning_loop
from tapeagents.llms import TrainableLLM
from tapeagents.finetune.logging_ import init_wandb, flatten_dict_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


@hydra.main(config_path="../../conf/", config_name="rl_gsm8k")
def main(cfg: DictConfig):
    random.seed(42)
    exp_path = Path(cfg.output_dir)
    run = init_wandb(cfg, exp_path, flatten_dict_config(cfg))
    if run is None:
        raise ValueError("Failed to initialize wandb run")
    state_path = exp_path / "rl_state.json"
    state = load_state(state_path)
    # optionally clean all data at start time
    if cfg.force_restart:
        clean_up(exp_path, state, state_path)
    attempts = 10
    # Serve the vLLM model

    stdout_path = exp_path / "vllm_stdout.log"
    stderr_path = exp_path / "vllm_stderr.log"
    port = 8080

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
    for _ in range(10):
        if os.path.exists(finetune_path / "current"):
            assistant_model_path = str(finetune_path / "current")
        else:
            assistant_model_path = cfg.model_path

        assistant_process, stdout_file, stderr_file = serve_vllm_local_model(
            model_name_or_path=assistant_model_path,
            stdout_file_path=stdout_path,
            stderr_file_path=stderr_path,
            port=port,
            verbose=True,
        )

        llm = TrainableLLM(
            base_url="http://127.0.0.1:8080",
            model_name=str(assistant_model_path),
            tokenizer_name=str(assistant_model_path),
            parameters=dict(temperature=0.7),
            use_cache=False,
        )

        agent = MathAgent.create(llm=llm)

        training_samples = []
        rewards = []
        tapes = []
        try:
            # use batch_main_loop
            sub_samples = random.sample(samples, cfg.max_agent_forks//attempts)
            for i, sample in enumerate(tqdm(sub_samples)):
                sample = extract_result_value(sample)
                for _ in range(attempts):
                    tape = solve_task(agent, env, sample)
                    tapes.append((i, tape))
                    reward = 1 if tape.metadata.result["solved"] else 0
                    rewards.append(reward)

                if i % 10 == 0 and i > 0:
                    logger.info(
                        f"{i}: Current accuracy: {np.mean(rewards):.3f}, prompt tokens used: {agent.llm.token_count}"
                    )
            logger.info(f"Accuracy: {np.mean(rewards):.3f}, prompt tokens used: {agent.llm.token_count}")
            wandb.log({"accuracy": np.mean(rewards), "prompt_tokens_used": agent.llm.token_count}, step=state["iteration"])
            for i, tape in tapes:
                for trace in agent.make_training_data(tape):
                    print(f"TRACE {i}")
                    print("CONTEXT", trace.prompt_text)
                    print("COMPLETION", trace.output_text)

                    trace.fork_id = i
                    trace.rewards = [reward]
                    training_samples.append(trace)
        except Exception as e:
            logger.error(colored(f"Failed to solve task: {e}", "red"))
        finally:
            # Make sure the weights of the model to be trained are not loaded on the GPU
            terminate_with_children(assistant_process.pid)
            assistant_process.wait()  # type: ignore
            if stdout_file:
                stdout_file.close()
            if stderr_file:
                stderr_file.close()

        # do we need to launch the base_model?
        if assistant_model_path == cfg.model_path:
            for trace in training_samples:
                trace.ref_logprobs = trace.old_logprobs
        else:
            base_model_process, basemodel_stdout_file, basemodel_stderr_file = serve_vllm_local_model(
                model_name_or_path=cfg.model_path,
                stdout_file_path=stdout_path,
                stderr_file_path=stderr_path,
                port=port,
                verbose=True,
            )
            try:
                basemodel_llm = TrainableLLM(
                    base_url="http://127.0.0.1:8080",
                    model_name=cfg.model_path,
                    tokenizer_name=cfg.model_path,
                    parameters=dict(temperature=0.7),
                )

                basemodel_agent = MathAgent.create(llm=basemodel_llm)
                base_model_traces = []
                for tape in tapes:
                    for i, trace in enumerate(basemodel_agent.make_training_data(tape)):
                        trace.fork_id = i
                        base_model_traces.append(trace)
                for trace, base_model_trace in zip(training_samples, base_model_traces):
                    trace.ref_logprobs = base_model_trace.old_logprobs

            except Exception as e:
                logger.error(colored(f"Failed to solve task: {e}", "red"))
            finally:
                # Make sure the weights of the model to be trained are not loaded on the GPU
                terminate_with_children(base_model_process.pid)
                base_model_process.wait()
                if basemodel_stdout_file:
                    basemodel_stdout_file.close()
                if basemodel_stderr_file:
                    basemodel_stderr_file.close()

        rollout_dir = exp_path / "rollouts" / str(state["iteration"])
        os.makedirs(rollout_dir, exist_ok=True)
        with open(rollout_dir / "data.jsonl", "w") as f:
            for trace in training_samples:
                f.write(trace.model_dump_json() + "\n")
                f.flush()

        finetune_cfg = cfg.copy()

        finetune_cfg.finetune.save_checkpoint_steps = max(
            cfg.max_agent_forks // attempts // (cfg.finetune.train_batch_size * cfg.finetune.gradient_accumulation_passes),
            1,
        )
        interrupt_train_steps = int((state["iteration"] + 1) * finetune_cfg.finetune.save_checkpoint_steps)
        finetune_cfg.finetune.interrupt_train_steps = interrupt_train_steps
        finetune_cfg.output_dir = str(finetune_path)
        finetune_cfg.finetune.data = {"data_parts_train": [{"path": str(rollout_dir)}]}
        config_path = conf_dir / f"{state['iteration']}.yaml"
        OmegaConf.save(finetune_cfg, config_path)

        finetune_command = (
            "cd /home/toolkit/TapeAgents/tapeagents && "
            "PYTHONPATH=/home/toolkit/TapeAgents:/home/toolkit/TapeAgents/tapeagents/src "
            "conda run -n tapeagents --no-capture-output "
            "accelerate launch --mixed_precision=bf16 --num_processes 1 "
            "--config_file /home/toolkit/TapeAgents/conf/deepspeed/accelerate_local.yaml "
            f"run_finetune.py --config-dir {str(conf_dir)} "
            f"--config-name {str(state['iteration'])} hydra.run.dir={str(finetune_path)}"
        )
        logger.info(f"Executing finetune command: {finetune_command}")

        error_code = subprocess.call(finetune_command, shell=True)

        if error_code != 0:
            logger.error(f"Finetuning failed with error code {error_code}")
            sys.exit(1)

        state["iteration"] += 1
        save_state(state, state_path)


if __name__ == "__main__":
    main()
