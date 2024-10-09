import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, TextIO, Tuple
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import psutil
import requests
from datasets import load_dataset
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


@hydra.main(config_path="../../conf/", config_name="rl_gsm8k")
def main(cfg: DictConfig): 
    exp_path = Path("outputs/gsm8k/tuning/llama31_8b_train")
    attempts=2
    # Serve the vLLM model
    model_path = "/mnt/llmd/base_models/Meta-Llama-3.1-8B-Instruct"
    stdout_path = exp_path / "vllm_stdout.log"
    stderr_path = exp_path / "vllm_stderr.log"
    port = 8080

    dataset = load_dataset("openai/gsm8k", "main", split="train")
    samples = [s for s in dataset]
    np.random.seed(42)
    np.random.shuffle(samples)  # type: ignore
    # for sample in dataset:
    #    samples.append(sample)
    #    if len(samples) == 2:
    #        break
    logging.info(f"Loaded {len(samples)} samples")

    llm = TrainableLLM(
        base_url="http://127.0.0.1:8080",
        model_name="/mnt/llmd/base_models/Meta-Llama-3.1-8B-Instruct",
        tokenizer_name="/mnt/llmd/base_models/Meta-Llama-3.1-8B-Instruct",
        parameters=dict(temperature=0.7),
    )

    agent = MathAgent.create(llm=llm)
    env = MathEnvironment()

    tapes_dir = os.path.join(exp_path, "tapes")
    logger.info(f"Saving tapes to {tapes_dir}")
    os.makedirs(tapes_dir, exist_ok=True)
    os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(exp_path, "llm_calls.sqlite")

    assistant_process, stdout_file, stderr_file = serve_vllm_local_model(
        model_name_or_path=model_path,
        stdout_file_path=stdout_path,
        stderr_file_path=stderr_path,
        port=port,
        verbose=True,
    )

    training_samples = []
    rewards = []
    try:
        # batch_main_loop
        with open(exp_path / "tapes.jsonl", "w") as f:
            for i, sample in enumerate(tqdm(samples[:10])):
                sample = extract_result_value(sample)
                for j in range(attempts):
                    tape = solve_task(agent, env, sample)
                    reward = 1 if tape.metadata.result["solved"] else 0
                    rewards.append(reward)
                    for i, trace in enumerate(agent.make_training_data(tape)):
                        print(f"TRACE {i}")
                        print("CONTEXT", trace.prompt_text)
                        print("COMPLETION", trace.output_text)

                        trace.reward = reward
                        trace.ref_log_probs = trace.log_probs
                        training_samples.append(trace)
                        f.write(trace.model_dump_json() + "\n")
                        f.flush()
    except Exception as e:
        logger.error(colored(f"Failed to solve task: {e}", "red"))
    finally:
        if stdout_file:
            stdout_file.close()
        if stderr_file:
            stderr_file.close()
        # Kill the vLLM process
        if assistant_process:
            assistant_process.terminate()
            assistant_process.wait()

        if i % 10 == 0 and i > 0:
            logger.info(f"{i}: Current accuracy: {np.mean(rewards):.3f}, prompt tokens used: {agent.llm.token_count}")
    cfg_finetune = cfg.finetune.copy()
    cfg_finetune.output_dir = str(exp_path / "finetune")
    run_finetuning_loop(cfg=cfg_finetune, training_samples=training_samples)
    logger.info(f"Accuracy: {np.mean(rewards):.3f}, prompt tokens used: {agent.llm.token_count}")
    logger.info(f"{len(rewards)} tapes saved to {tapes_dir}")


if __name__ == "__main__":
    main()
