import logging
import os
import sys
import contextlib
import json
import logging
import os
import shutil
import socket
import subprocess
import threading
import time
import numpy as np
from datasets import load_dataset
from termcolor import colored
from tqdm import tqdm
import time
import hydra
import numpy as np
import psutil
import requests
import torch
import wandb
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from pprint import pformat
from typing import Callable, Dict, List, Optional, TextIO, Tuple
from termcolor import cprint
from tqdm import tqdm

from tapeagents.llms import TrainableLLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        # f"VLLM_PORT={vllm_port} "
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


def main(exp_path: str, attempts: int = 1):
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    samples = [s for s in dataset]
    np.random.seed(42)
    np.random.shuffle(samples)  # type: ignore
    logging.info(f"Loaded {len(samples)} samples")

    llm = TrainableLLM(
        base_url="https:/localhost:8080",
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        tokenizer_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        parameters=dict(temperature=0.7),
    )
    agent = MathAgent.create(llm)
    env = MathEnvironment()

    tapes_dir = os.path.join(exp_path, "tapes")
    logger.info(f"Saving tapes to {tapes_dir}")
    os.makedirs(tapes_dir, exist_ok=True)
    os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(exp_path, "llm_calls.sqlite")

    solved = []
    for i, sample in enumerate(tqdm(samples)):
        sample = extract_result_value(sample)
        for j in range(attempts):
            tape_file = os.path.join(tapes_dir, f"task{i}_attempt{j+1}.json")
            if os.path.exists(tape_file):
                logger.info(f"Task {i} attempt {j+1} already solved, skipping")
                continue
            try:
                tape = solve_task(agent, env, sample, tape_file)
                solved.append(int(tape.metadata.result["solved"]))
                save_tape(tape_file, tape)
            except Exception as e:
                logger.error(colored(f"Failed to solve task, attempt {j+1}: {e}", "red"))
                solved.append(0)
        if i % 10 == 0 and i > 0:
            logger.info(f"{i}: Current accuracy: {np.mean(solved):.3f}, prompt tokens used: {agent.llm.token_count}")
    logger.info(f"Accuracy: {np.mean(solved):.3f}, prompt tokens used: {agent.llm.token_count}")
    logger.info(f"{len(solved)} tapes saved to {tapes_dir}")


if __name__ == "__main__":
    exp_path = sys.argv[1] if len(sys.argv) > 1 else "gsm8k/tuning/llama31_70b_train"
    main(exp_path)
