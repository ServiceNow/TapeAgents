import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional, TextIO, Tuple

import numpy as np
import psutil
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from termcolor import cprint

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


@retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=2, min=10))
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
        # f"--pipeline-parallel-size {tensor_parallel_size} "
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


def setup_logging(output_dir):
    print(f"Setting up logging to {output_dir}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # Create the output directory if it doesn't exist

    # Define log file paths
    info_log = output_dir / "info.log"
    debug_log = output_dir / "debug.log"
    error_log = output_dir / "error.log"

    # Clear any existing handlers
    logger = logging.getLogger()  # get root logger
    logger.handlers = []  # Clear existing handlers
    logger.setLevel(logging.DEBUG)  # Ensure all levels are captured at the root level

    # Create file handlers for each log level
    info_handler = logging.FileHandler(info_log)
    info_handler.setLevel(logging.INFO)

    debug_handler = logging.FileHandler(debug_log)
    debug_handler.setLevel(logging.DEBUG)

    error_handler = logging.FileHandler(error_log)
    error_handler.setLevel(logging.ERROR)

    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.INFO)

    # Create formatters and set them to the handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    info_handler.setFormatter(formatter)
    debug_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(info_handler)
    logger.addHandler(debug_handler)
    logger.addHandler(error_handler)
    logger.addHandler(stdout_handler)


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
    # list of files to remove
    files = [
        target_path / "assistant_vllm_stdout.log",
        target_path / "assistant_vllm_stderr.log",
        target_path / "basemodel_vllm_stdout.log",
        target_path / "basemodel_vllm_stderr.log",
        target_path / "debug.log",
        target_path / "error.log",
        target_path / "info.log",
    ]

    for file in files:
        if file.exists():
            # erase the content but not the file
            with open(file, "w"):
                pass
            logger.info(f"{file} erased.")

    # List of directories to remove
    directories = [
        target_path / "llm_calls.sqlite",
        target_path / "dialogue_trace.log",
        target_path / "rollouts",
        target_path / "tapes",
        target_path / "conf",
        target_path / "finetune" / "current",
        target_path / "finetune" / "logs",
        target_path / "finetune" / "intermediate",
        target_path / "finetune" / "training_state",
    ]

    for directory in directories:
        remove_dir(directory)
        logger.info(f"{directory} removed.")


def calculate_stats(stats):
    return {
        "max": np.mean([max(stats) for stats in stats.values() if stats]),
        "min": np.mean([min(stats) for stats in stats.values() if stats]),
        "var": np.mean([np.var(stats) for stats in stats.values() if stats]),
        "mean": np.mean([np.mean(stats) for stats in stats.values() if stats]),
    }
