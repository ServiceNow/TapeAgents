import copy
import json
import logging
import multiprocessing
import os
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
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
from tenacity import retry, stop_after_attempt, wait_exponential
from termcolor import colored, cprint
from tqdm import tqdm
from tapeagents.observe import retrieve_all_llm_calls
import wandb
from examples.gsm8k_tuning.math_agent import (
    AnswerAction,
    MathAgent,
    MathEnvironment,
    MathTape,
    Task,
    extract_result_value,
)
from tapeagents.batch import batch_main_loop
from tapeagents.core import AgentResponseParsingFailureAction, StepMetadata, TapeMetadata, TrainingText
from tapeagents.finetune.finetune import load_config, run_finetuning_loop
from tapeagents.finetune.logging_ import flatten_dict_config, init_wandb
from tapeagents.io import save_json_tape
from tapeagents.llms import TrainableLLM
from tapeagents.observe import start_sqlite_writer, stop_sqlite_writer

logger = logging.getLogger(__name__)


def get_log_probs(agent, trace):
    try:
        trace.ref_logprobs = agent.llm.get_log_probs(trace.prompt_text, trace.output_text)  # type: ignore
        return trace
    except Exception as e:
        logger.error(colored(f"Failed to get ref log probs: {e}", "red"))
        return None


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


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2))
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


def create_tapes(samples):
    tapes = []
    for sample in samples:
        start_step = Task(task=sample["question"], metadata=StepMetadata(other=extract_result_value(sample)))
        tape = MathTape(steps=[start_step], context=None)
        tapes.append(tape)
    return tapes


def process_dataset(agent, tapes, cfg, env, tapes_dir, dataset_name):
    start_process_tapes = time.time()
    os.makedirs(tapes_dir, exist_ok=True)
    reward_stats = defaultdict(list)
    step_stats = defaultdict(list)
    no_errors_stats = defaultdict(list)
    success_stats = defaultdict(list)
    training_samples: List[TrainingText] = []

    logger.info("Starting main loop")
    start_make_new_tapes = time.time()
    new_tapes = list(batch_main_loop(agent, tapes, env, max_loops=cfg.max_loops, n_workers=cfg.n_workers))
    llm_calls = retrieve_all_llm_calls(os.environ["TAPEAGENTS_SQLITE_DB"] )
    end_make_new_tapes = time.time()

    def process_tape(new_tape, agent, dataset_name, tapes_dir):
        if any([isinstance(step, AgentResponseParsingFailureAction) for step in new_tape.steps]):
            new_tape_filtered = copy.deepcopy(new_tape)
            new_tape_filtered.steps = [
                step for step in new_tape.steps if not isinstance(step, AgentResponseParsingFailureAction)
            ]
            no_error, reward, success = 0, -1, 0
            new_tape = new_tape_filtered
        else:
            no_error = 1
            if (
                isinstance(new_tape.steps[-1], AnswerAction)
                and new_tape.steps[-1].value == new_tape.steps[0].metadata.other["value"]
            ):
                reward, success = 1, 1
            else:
                reward, success = 0, 0

        save_json_tape(new_tape, os.path.join(tapes_dir, f"{new_tape.metadata.id}.json"))

        training_samples = []
        if dataset_name == "train":
            prompt_ids = [step.metadata.prompt_id for step in new_tape.steps if step.metadata.prompt_id]
            sub_llm_calls = [call for call in llm_calls if call.prompt.id in prompt_ids]
            for trace in agent.make_training_data(new_tape, sub_llm_calls):
                trace.old_logprobs = agent.llm.get_log_probs(trace.prompt_text, trace.output_text)
                trace.rewards = [reward]
                trace.fork_id = new_tape.metadata.parent_id
                training_samples.append(trace)

        return new_tape, reward, len(new_tape.steps), success, no_error, training_samples

    logger.info("Starting data creation")
    start_make_data = time.time()
    with ThreadPoolExecutor(max_workers=torch.cuda.device_count()) as executor:
        process_tape_partial = partial(process_tape, agent=agent, dataset_name=dataset_name, tapes_dir=tapes_dir)
        futures = [executor.submit(process_tape_partial, new_tape) for new_tape in new_tapes]
        # Wrap futures with tqdm for progress tracking
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tapes", unit="tape"):
            new_tape, reward, steps, success, no_error, tape_training_samples = future.result()
            reward_stats[new_tape.metadata.parent_id].append(reward)
            step_stats[new_tape.metadata.parent_id].append(steps)
            success_stats[new_tape.metadata.parent_id].append(success)
            no_errors_stats[new_tape.metadata.parent_id].append(no_error)
            new_tapes.append(new_tape)
            training_samples.extend(tape_training_samples)

    end_make_data = time.time()

    end_process_tapes = time.time()

    stats = {
        **{f"{dataset_name}_{k}_reward": v for k, v in calculate_stats(reward_stats).items()},
        **{f"{dataset_name}_{k}_steps": v for k, v in calculate_stats(step_stats).items()},
        **{f"{dataset_name}_{k}_success": v for k, v in calculate_stats(success_stats).items()},
        **{f"{dataset_name}_{k}_no_errors": v for k, v in calculate_stats(no_errors_stats).items()},
        **{
            f"execution_time/{dataset_name}_make_new_tapes": end_make_new_tapes - start_make_new_tapes,
            f"execution_time/{dataset_name}_make_data": end_make_data - start_make_data,
            f"execution_time/{dataset_name}_tapes_per_second": len(new_tapes)
            / (end_process_tapes - start_process_tapes),
        },
    }
    return new_tapes, training_samples, stats


@hydra.main(config_path="../../conf/", config_name="fast_rl_gsm8k")
def main(cfg: DictConfig):
    multiprocessing.set_start_method("spawn")
    random.seed(42)
    exp_path = Path(cfg.output_dir)
    setup_logging(exp_path)
    cfg.finetune.wandb_id = exp_path.name
    run = init_wandb(cfg, exp_path, flatten_dict_config(cfg))
    if run is None:
        raise ValueError("Failed to initialize wandb run")
    state_path = exp_path / "rl_state.json"
    state = load_state(state_path)
    # optionally clean all data at start time
    if cfg.force_restart:
        clean_up(exp_path, state, state_path)

    train_dataset = load_dataset("openai/gsm8k", "main", split="train")
    train_samples = [s for s in train_dataset]
    test_dataset = load_dataset("openai/gsm8k", "main", split="test")
    test_samples = [s for s in test_dataset]
    logging.info(f"Loaded {len(train_samples)} training samples")
    logging.info(f"Loaded {len(test_samples)} test samples")

    env = MathEnvironment()

    os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(exp_path, "llm_calls.sqlite")
    conf_dir = exp_path / "conf"
    os.makedirs(conf_dir, exist_ok=True)
    finetune_path = exp_path / "finetune"
    while state["iteration"] <= cfg.max_iterations:
        if os.path.exists(finetune_path / "current"):
            assistant_model_path = str(finetune_path / "current")
        else:
            assistant_model_path = cfg.model_path

        start_serving_assistant = time.time()
        assistant_process, stdout_file, stderr_file = serve_vllm_local_model(
            model_name_or_path=assistant_model_path,
            stdout_file_path=exp_path / "assistant_vllm_stdout.log",
            stderr_file_path=exp_path / "assistant_vllm_stderr.log",
            port=8080,
            verbose=True,
            cuda_device=",".join([str(i) for i in range(torch.cuda.device_count())]),
        )
        end_serving_assistant = time.time()

        llm = TrainableLLM(
            base_url="http://127.0.0.1:8080",
            model_name=str(assistant_model_path),
            tokenizer_name=str(assistant_model_path),
            parameters=cfg.llm.parameters,
            use_cache=False,
        )

        test_llm = TrainableLLM(
            base_url="http://127.0.0.1:8080",
            model_name=str(assistant_model_path),
            tokenizer_name=str(assistant_model_path),
            parameters=cfg.test_llm.parameters,
            use_cache=False,
        )

        tapes_dir = exp_path / "tapes" / str(state["iteration"])
        os.makedirs(tapes_dir, exist_ok=True)

        try:
            sub_samples = random.sample(train_samples, cfg.max_agent_forks // cfg.attempts)
            train_tapes = create_tapes(sub_samples)
            train_tapes = [copy.deepcopy(tape) for tape in train_tapes for _ in range(cfg.attempts)]
            test_tapes = create_tapes(test_samples)
            train_agent = MathAgent.create(llm=llm)
            test_agent = MathAgent.create(llm=test_llm)

            datasets = [("train", train_agent, train_tapes)]
            if state["iteration"] % cfg.test_every_n_iterations == 0:
                datasets.append(("test", test_agent, test_tapes))
            all_results = {}

            for dataset_name, agent, tapes in datasets:
                tapes_dir = exp_path / "tapes" / dataset_name / str(state["iteration"])
                new_tapes, training_samples, stats = process_dataset(agent, tapes, cfg, env, tapes_dir, dataset_name)

                all_results[dataset_name] = {
                    "new_tapes": new_tapes,
                    "training_samples": training_samples,
                    "stats": stats,
                }

                # Log results
                logger.info(f"{dataset_name.capitalize()} Results:")
                for stat_name, stat_value in stats.items():
                    logger.info(f"{dataset_name}_{stat_name}: {stat_value}")

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

        logger.info(f"Collected {len(training_samples)} training samples")
        stats = all_results["train"]["stats"]
        if "test" in all_results:
            stats.update(all_results["test"]["stats"])
        wandb.log(
            {
                **{
                    "execution_time/serving_assistant": end_serving_assistant - start_serving_assistant,
                },
                **stats,
            },
            step=state["iteration"],
        )

        start_basemodel_logprobs = time.time()
        training_samples = all_results["train"]["training_samples"]
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

                with ThreadPoolExecutor() as executor:
                    futures = [executor.submit(get_log_probs, basemodel_agent, trace) for trace in training_samples]
                    for future in as_completed(futures):
                        result = future.result()
                        if result:
                            new_training_samples.append(result)

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
        for trace in training_samples:
            with open(rollout_dir / f"{trace.fork_id}.jsonl", "a") as f:
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
        # error_code = subprocess.call(finetune_command, shell=True)
        p = multiprocessing.Process(target=run_finetuning_loop, args=(finetune_cfg,))
        p.start()  # Start the subprocess
        p.join()  # Wait for the process to complete
        end_finetune = time.time()

        # if error_code != 0:
        #    logger.error(f"Finetuning failed with error code {error_code}")
        #    sys.exit(1)

        wandb.log(
            {
                "execution_time/finetune": end_finetune - start_finetune,
            },
            step=state["iteration"],
        )
        state["iteration"] += 1
        save_state(state, state_path)


if __name__ == "__main__":
    try:
        start_sqlite_writer()
        main()
    finally:
        stop_sqlite_writer()
