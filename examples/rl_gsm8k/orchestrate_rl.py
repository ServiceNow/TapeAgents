import copy
import logging
import multiprocessing
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from termcolor import colored
from tqdm import tqdm

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
from tapeagents.core import AgentResponseParsingFailureAction, StepMetadata, TrainingText
from tapeagents.finetune.context import accelerator
from tapeagents.finetune.finetune import run_finetuning_loop
from tapeagents.finetune.logging_ import flatten_dict_config, init_wandb
from tapeagents.io import save_json_tape
from tapeagents.llms import TrainableLLM
from tapeagents.observe import erase_sqlite, retrieve_all_llm_calls, start_sqlite_queue_writer, stop_sqlite_queue_writer

from .utils import (
    calculate_stats,
    clean_up,
    load_state,
    save_state,
    serve_vllm_local_model,
    setup_logging,
    terminate_with_children,
)

logger = logging.getLogger(__name__)


def annotate_trace_with_ref_log_probs(agent: MathAgent, trace: TrainingText) -> TrainingText | None:
    try:
        trace.ref_logprobs = agent.llm.get_log_probs(trace.prompt_text, trace.output_text)  # type: ignore
        return trace
    except Exception as e:
        logger.error(colored(f"Failed to get ref log probs: {e}", "red"))
        return None


def convert_samples_to_tapes(samples: list) -> list[MathTape]:
    """
    Creates MathTape objects from a list of sample dictionaries.

    Args:
        samples (list[dict]): List of dictionaries containing math problems, where each dict
            has 'question' and expected answer value. The list is created from a dataset.

    Returns:
        list[MathTape]: List of MathTape objects initialized with the math problems as Task steps.
            Each tape contains a single starting Task step with the question and expected answer value
            stored in metadata.
    """
    tapes = []
    for sample in samples:
        start_step = Task(task=sample["question"], metadata=StepMetadata(other=extract_result_value(sample)))
        tape = MathTape(steps=[start_step], context=None)
        tapes.append(tape)
    return tapes


def generate_training_data(
    agent: MathAgent,
    tapes: list[MathTape],
    cfg: DictConfig,
    env: MathEnvironment,
    tapes_dir: str | Path,
    dataset_name: str,
) -> Tuple[List[MathTape], List[TrainingText], Dict[str, float]]:
    """
    Generate complete tapes and training samples from a list of initialized tapes.

    Args:
        agent: Agent that interacts with the math environment
        tapes: List of tapes initialized with math problems
        cfg: Configuration
        env: Environment with tools
        tapes_dir: Directory to save processed episodes
        dataset_name: Name of dataset ('train' or other)

    Returns:
        Tuple containing:
        - List of completed MathTapes
        - List of training samples with rewards and logprobs
        - Dictionary of performance statistics and execution times
    """

    start_make_data = time.time()
    os.makedirs(tapes_dir, exist_ok=True)
    reward_stats = defaultdict(list)
    step_stats = defaultdict(list)
    no_errors_stats = defaultdict(list)
    success_stats = defaultdict(list)
    training_samples: List[TrainingText] = []

    logger.info("Starting main loop")
    start_sampling_from_llm = time.time()
    new_tapes = list(batch_main_loop(agent, tapes, env, max_loops=cfg.max_loops, n_workers=cfg.n_workers))
    end_sampling_from_llm = time.time()
    start_reading_sqlite = time.time()
    if dataset_name == "train":
        llm_calls = retrieve_all_llm_calls(os.environ["TAPEAGENTS_SQLITE_DB"])
    else:
        llm_calls = []
    end_reading_sqlite = time.time()

    def extract_tape_training_samples(
        new_tape: MathTape, agent: MathAgent, dataset_name: str, tapes_dir: str
    ) -> Tuple[MathTape, List[TrainingText], Dict[str, int]]:
        """
        Process a single tape to extract training samples and statistics.

        Args:
            new_tape: The tape to process containing math problem steps
            agent: MathAgent
            dataset_name: Name of dataset ('train' or 'test')
            tapes_dir: Directory to save processed tapes

        Returns:
            Tuple containing:
            - Processed MathTape
            - List of training samples with rewards and logprobs
            - Dictionary with statistics (reward, steps, success, no_errors)
        """
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

        training_samples: list[TrainingText] = []
        if dataset_name == "train":
            prompt_ids = [step.metadata.prompt_id for step in new_tape.steps if step.metadata.prompt_id]
            sub_llm_calls = [call for call in llm_calls if call.prompt.id in prompt_ids]
            # Sort sub_llm_calls to match the order of prompt_ids
            sub_llm_calls = sorted(sub_llm_calls, key=lambda call: prompt_ids.index(call.prompt.id))
            # Process LLM calls in reverse order to apply reward discounting
            # For each LLM interaction in the tape:
            # - Create a training sample from the prompt and output
            # - Get log probabilities of the output tokens
            # - Apply discounted reward based on position (earlier steps get more discount)
            # - Set parent tape ID for tracking
            for i, llm_call in enumerate(sub_llm_calls[::-1]):
                trace = agent.llm.make_training_text(llm_call.prompt, llm_call.output)
                trace.logprobs = agent.llm.get_log_probs(trace.prompt_text, trace.output_text)
                trace.reward = reward * (cfg.reward_discount**i)
                trace.parent_tape_id = (
                    f"{new_tape.metadata.parent_id}_{i}"
                    if cfg.normalize_reward_per_step
                    else new_tape.metadata.parent_id
                )
                training_samples.append(trace)
        tape_stats = {
            "reward": reward,
            "steps": len(new_tape.steps),
            "success": success,
            "no_errors": no_error,
        }
        return new_tape, training_samples, tape_stats

    logger.info("Starting data creation")
    start_annotate_tape = time.time()
    with ThreadPoolExecutor(max_workers=torch.cuda.device_count()) as executor:
        extract_tape_training_samples_partial = partial(
            extract_tape_training_samples, agent=agent, dataset_name=dataset_name, tapes_dir=tapes_dir
        )
        futures = [executor.submit(extract_tape_training_samples_partial, new_tape) for new_tape in new_tapes]
        # Wrap futures with tqdm for progress tracking
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tapes", unit="tape"):
            new_tape, tape_training_samples, tape_stats = future.result()
            reward_stats[new_tape.metadata.parent_id].append(tape_stats["reward"])
            step_stats[new_tape.metadata.parent_id].append(tape_stats["steps"])
            success_stats[new_tape.metadata.parent_id].append(tape_stats["success"])
            no_errors_stats[new_tape.metadata.parent_id].append(tape_stats["no_error"])
            new_tapes.append(new_tape)
            training_samples.extend(tape_training_samples)

    end_annotate_tape = time.time()

    end_make_data = time.time()

    stats = {
        **{f"{dataset_name}_{k}_reward": v for k, v in calculate_stats(reward_stats).items()},
        **{f"{dataset_name}_{k}_steps": v for k, v in calculate_stats(step_stats).items()},
        **{f"{dataset_name}_{k}_success": v for k, v in calculate_stats(success_stats).items()},
        **{f"{dataset_name}_{k}_no_errors": v for k, v in calculate_stats(no_errors_stats).items()},
        **{
            f"execution_time/{dataset_name}_sampling_from_llm": end_sampling_from_llm - start_sampling_from_llm,
            f"execution_time/{dataset_name}_annotate_tapes": end_annotate_tape - start_annotate_tape,
            f"execution_time/{dataset_name}_make_data": end_make_data - start_make_data,
            f"execution_time/{dataset_name}_tapes_made_per_second": len(new_tapes) / (end_make_data - start_make_data),
            f"execution_time/{dataset_name}_reading_sqlite": end_reading_sqlite - start_reading_sqlite,
        },
    }
    return new_tapes, training_samples, stats


@hydra.main(config_path="../../conf/", config_name="fast_rl_gsm8k")
def main(cfg: DictConfig):
    multiprocessing.set_start_method("spawn")  # necessary to use gpus in subprocesses
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
            train_tapes = convert_samples_to_tapes(sub_samples)
            train_tapes = [copy.deepcopy(tape) for tape in train_tapes for _ in range(cfg.attempts)]
            test_tapes = convert_samples_to_tapes(test_samples)
            train_agent = MathAgent.create(llm=llm)
            test_agent = MathAgent.create(llm=test_llm)

            datasets = [("train", train_agent, train_tapes)]
            if state["iteration"] % cfg.test_every_n_iterations == 0:
                datasets.append(("test", test_agent, test_tapes))
            all_results = {}

            for dataset_name, agent, tapes in datasets:
                tapes_dir = exp_path / "tapes" / dataset_name / str(state["iteration"])
                new_tapes, training_samples, stats = generate_training_data(
                    agent, tapes, cfg, env, tapes_dir, dataset_name
                )

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
        if "test" in all_results:  # test is only present every cfg.test_every_n_iterations
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
            # At the first itetration, Ref logprobs are the same as logprobs
            for trace in training_samples:
                trace.ref_logprobs = trace.logprobs
                new_training_samples.append(trace)
        else:
            # Load the base model to get the reference log probabilities
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
                    futures = [
                        executor.submit(annotate_trace_with_ref_log_probs, basemodel_agent, trace)
                        for trace in training_samples
                    ]
                    for future in as_completed(futures):
                        trace = future.result()
                        if trace:
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
                "execution_time/populating_ref_logprobs": end_basemodel_logprobs - start_basemodel_logprobs,
            },
            step=state["iteration"],
        )
        rollout_dir = exp_path / "rollouts" / str(state["iteration"])
        os.makedirs(rollout_dir, exist_ok=True)
        for trace in training_samples:
            with open(rollout_dir / f"{trace.parent_tape_id}.jsonl", "a") as f:
                f.write(trace.model_dump_json() + "\n")
                f.flush()

        finetune_cfg = cfg.copy()

        interrupt_train_steps = int((state["iteration"] + 1) * finetune_cfg.finetune.save_checkpoint_steps)
        finetune_cfg.finetune.interrupt_train_steps = interrupt_train_steps
        finetune_cfg.output_dir = str(finetune_path)
        finetune_cfg.finetune.data = {"data_parts_train": [{"path": str(rollout_dir)}]}
        finetune_cfg.finetune.wandb_id = run.id + "_finetune"
        finetune_cfg.finetune.wandb_name = run.name + "_finetune"
        finetune_cfg.finetune.wandb_resume = "always"
        config_path = conf_dir / f"{state['iteration']}.yaml"
        OmegaConf.save(finetune_cfg, config_path)

        start_finetune = time.time()
        # Set up accelerate command with environment variables
        p = multiprocessing.Process(target=run_finetuning_loop, args=(finetune_cfg,))
        p.start()  # Start the subprocess
        p.join()  # Wait for the process to complete
        end_finetune = time.time()
        # Check if the subprocess exited with an error
        if p.exitcode != 0:
            raise RuntimeError(f"Finetuning subprocess failed with exit code {p.exitcode}")

        wandb.log(
            {
                "execution_time/finetune": end_finetune - start_finetune,
            },
            step=state["iteration"],
        )
        state["iteration"] += 1
        save_state(state, state_path)
        erase_sqlite()


if __name__ == "__main__":
    try:
        start_sqlite_queue_writer()
        main()
    finally:
        stop_sqlite_queue_writer()
