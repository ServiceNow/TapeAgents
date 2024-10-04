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
from tapeagents.environment import EmptyEnvironment
from tapeagents.llms import TrainableLLM
from tapeagents.environment import Environment
from tapeagents.dialog_tape import AssistantStep, DialogTape, SystemStep, UserStep
from tapeagents.batch import batch_main_loop
from tapeagents.io import save_tapes
from examples.rlaif.chat_agent import (
    ChatAgent,
)
from tapeagents.core import LLMOutput, PartialStep, Prompt, Tape, TapeMetadata, TrainingText

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_conversation_to_steps(conversation):
    steps = []
    for message in conversation:
        if message['role'] == 'user':
            steps.append(UserStep(content=message['content']))
        elif message['role'] == 'assistant':
            steps.append(AssistantStep(content=message['content']))
        elif message['role'] == 'system':
            steps.append(SystemStep(content=message['content']))
    return steps

def create_dialog_tape_from_sample(sample):
    steps = convert_conversation_to_steps(sample['conversation'])
    return DialogTape(
        context=None,
        steps=steps
    )


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


def main(exp_path: Path):
    dataset = load_dataset("allenai/WildChat", split="train")
    samples = []
    for i, sample in enumerate(dataset):
        if i == 10:
            break
        samples.append(sample)

    np.random.seed(42)
    np.random.shuffle(samples)  # type: ignore
    logging.info(f"Loaded {len(samples)} samples")

    llm = TrainableLLM(
        base_url="http://127.0.0.1:8080",
        model_name="/mnt/llmd/base_models/Meta-Llama-3.1-8B-Instruct",
        tokenizer_name="/mnt/llmd/base_models/Meta-Llama-3.1-8B-Instruct",
        parameters=dict(temperature=0.7),
    )
    agent = ChatAgent.create(llm)

    tapes_dir = os.path.join(exp_path, "tapes")
    logger.info(f"Saving tapes to {tapes_dir}")
    os.makedirs(tapes_dir, exist_ok=True)
    os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(exp_path, "llm_calls.sqlite")

    with save_tapes(exp_path / "tapes.yaml") as dumper:
        for i, sample in enumerate(tqdm(samples)):
            try:
                for i, message in enumerate(sample['conversation']):
                    if message['role'] != 'assistant':
                        continue

                    steps = convert_conversation_to_steps(sample['conversation'])
                    tape = DialogTape(
                                context=None,
                                steps=steps[:i], # drop the last assistant step
                            )
                    tape_file = os.path.join(tapes_dir, f"task{i}.json")
                    for event in agent.run(tape):
                        if event.partial_step:
                            assert isinstance(step := event.partial_step.step, AssistantStep)
                            print(step.content[n_printed:], end="", flush=True)
                            n_printed = len(step.content)
                        if event.final_tape:
                            tape = event.final_tape
                            print()
                            print(f"Received new tape of length {len(tape)}")
                        print("--- CHECK TRACES ---")

                traces: list[TrainingText] = []
                for i, trace in enumerate(agent.make_training_data(tape)):
                    print(f"TRACE {i}")
                    print("CONTEXT", trace.prompt_text)
                    print("COMPLETION", trace.output_text)
                    traces.append(trace)

            except Exception as e:
                logger.error(colored(f"Failed to solve task: {e}", "red"))


if __name__ == "__main__":
    exp_path = sys.argv[1] if len(sys.argv) > 1 else "wildchat/rl/llama31_8b_train"
    exp_path = Path(exp_path)
    main(exp_path)
