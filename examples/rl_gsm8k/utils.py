import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional, TextIO, Union

import numpy as np
import psutil
import requests
import torch
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class VLLMServiceManager:
    def __init__(
        self,
        model_name_or_path: Union[str, Path],
        stdout_file_path: Union[str, Path],
        stderr_file_path: Union[str, Path],
        port: int = 8080,
        verbose: bool = True,
        cuda_device: str = "0",
        host: str = "localhost",
        **kwargs,
    ):
        self.model_name_or_path = model_name_or_path
        self.stdout_file_path = stdout_file_path
        self.stderr_file_path = stderr_file_path
        self.port = port
        self.verbose = verbose
        self.cuda_device = cuda_device
        self.host = host
        self.kwargs = kwargs
        self.process: Optional[subprocess.Popen] = None
        self.stdout_file: Optional[TextIO] = None
        self.stderr_file: Optional[TextIO] = None
        self.stats = {}

    def _terminate_with_children(self, process_id: int) -> None:
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
            logger.warning(f"Could not terminate process with PID: {process_id}, not found")
        except psutil.AccessDenied:
            logger.error(f"Insufficient privileges to terminate process with PID: {process_id}")
        except Exception as e:
            logger.error(f"An error occurred while terminating process: {e}")

    def _wait_for_service(self, process: subprocess.Popen, url, headers=None, timeout=120) -> bool:
        start_time = time.time()
        while True:
            try:
                response = requests.get(url, headers=headers, timeout=1)
                if response.status_code == 200:
                    logger.info(f"-> Service at {url} is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass

            if time.time() - start_time > timeout:
                logger.error(f"-> Timeout waiting for service at {url}")
                return False

            if process.poll() is not None:
                logger.error(f"-> Service process has terminated")
                return False

            logger.info(f"-> Waiting for service at {url}")
            time.sleep(5)

    @retry(stop=stop_after_attempt(1), wait=wait_exponential(multiplier=2, min=10))
    def _start_service(self) -> None:
        tensor_parallel_size = self.cuda_device.count(",") + 1
        kwargs_str = " ".join([f"{k} {v}" for k, v in self.kwargs.items()]) if self.kwargs else ""

        cmd = (
            f"OUTLINES_CACHE_DIR=/tmp/.outlines_{self.cuda_device}_{int(time.time())} "
            f"CUDA_VISIBLE_DEVICES={self.cuda_device} "
            f"python -m vllm.entrypoints.openai.api_server "
            f"--model {self.model_name_or_path} "
            f"--tensor-parallel-size {tensor_parallel_size} "
            f"--port {self.port} "
            "--disable-frontend-multiprocessing "
            "--dtype bfloat16 "
            f"{kwargs_str}"
        )

        if tensor_parallel_size > 1:
            cmd = "VLLM_WORKER_MULTIPROC_METHOD=spawn " + cmd

        if self.verbose:
            logger.info(f"Server launcher cmd: {cmd}")

        process_args = {
            "shell": True,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }

        if self.verbose:
            self.stdout_file = open(self.stdout_file_path, "w")
            self.stderr_file = open(self.stderr_file_path, "w")
            process_args["stdout"] = self.stdout_file
            process_args["stderr"] = self.stderr_file

        try:
            self.process = subprocess.Popen(cmd, **process_args)
        except Exception as e:
            logger.error(f"Error occurred: {e}")
            raise e

        vllm_url = f"http://{self.host}:{self.port}/health"
        headers = {"User-Agent": "vLLM Client"}

        start_waiting = time.time()
        if self._wait_for_service(self.process, vllm_url, headers=headers, timeout=8000):
            logger.info(f"Student {self.model_name_or_path} model loaded on port {self.port}")
        else:
            self._cleanup()
            raise Exception("Failed to start the service")
        end_waiting = time.time()
        self.stats["starting_time"] = end_waiting - start_waiting

    def _cleanup(self) -> None:
        if self.process and self.process.pid:
            self._terminate_with_children(self.process.pid)
            self.process.wait()

        if self.stdout_file:
            self.stdout_file.close()
        if self.stderr_file:
            self.stderr_file.close()

    def __enter__(self) -> "VLLMServiceManager":
        self._start_service()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._cleanup()
    
    def get_stats(self):
        return self.stats


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


def launch_training(config_dir: str, config_name: str, accelerate_cfg_path: str, deepspeed_cfg_path: str):
    """
    Launch training process with proper GPU configuration and error handling.

    Args:
        config_path (str): Path to the training config file
        cfg: Configuration object containing accelerate_cfg_path

    Returns:
        float: Training duration in seconds

    Raises:
        ValueError: If no GPUs are available
        RuntimeError: If training process fails
    """
    # Check GPU availability
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise ValueError("No GPUs available for finetuning")

    # Construct command based on GPU count
    base_cmd = [
        "accelerate",
        "launch",
        "--mixed_precision=bf16",
        "--config_file",
        accelerate_cfg_path,
        "examples/rl_gsm8k/run_finetune.py",
        "--config-dir",
        config_dir,
        "--config-name",
        config_name,
    ]

    if num_gpus > 1:
        #TODO: better handling of multi-gpu training: accelerate or deepspeed
        base_cmd[2:2] = [
            "--use_deepspeed",
            #"--multi_gpu",
            "--num_processes",
            str(num_gpus),
            "--deepspeed_config_file",
            deepspeed_cfg_path,
        ]

    logger.info(f"Launching training with command: {' '.join(base_cmd)}")
    try:
        result = subprocess.run(
            base_cmd,
            check=True,  # Raises CalledProcessError if return code != 0
            text=True,
            capture_output=False,
        )

    except subprocess.CalledProcessError as e:
        # Capture both stdout and stderr for debugging
        error_msg = (
            f"Training process failed with exit code {e.returncode}\n" f"stdout: {e.stdout}\n" f"stderr: {e.stderr}"
        )
        raise RuntimeError(error_msg) from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error during training: {str(e)}") from e
