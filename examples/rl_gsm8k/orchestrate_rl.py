import copy
import json
import logging
import multiprocessing
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from itertools import chain
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from termcolor import colored
from tqdm import tqdm

import wandb
from tapeagents.agent import Agent
from tapeagents.core import LLMCall, LLMOutputParsingFailureAction, StepMetadata, TrainingText
from tapeagents.finetune.data import MASKED_TOKEN_ID
from tapeagents.finetune.logging_ import flatten_dict_config, init_wandb
from tapeagents.llms import TrainableLLM

from .cot_math_agent import (
    CoTMathAgent,
    RLMathTape,
    Task,
)
from .deepseek_math_eval.answer_extraction import extract_last_single_answer, extract_math_answer
from .deepseek_math_eval.eval_script import eval_last_single_answer, eval_math
from .deepseek_math_eval.process_utils import process_eurus_test, process_gsm8k_test, process_math_test
from .utils import (
    VLLMServiceManager,
    calculate_stats,
    clean_up,
    launch_training,
    load_state,
    save_state,
    setup_logging,
)

logger = logging.getLogger(__name__)


def batch_annotate_traces_with_ref_logprobs(llm: TrainableLLM, traces: List[TrainingText]):
    prompt_token_ids = []
    completion_token_ids = []
    for trace in traces:
        prompt_token_ids.append(trace.input_ids[: -len(trace.logprobs)])
        completion_token_ids.append(trace.input_ids[-len(trace.logprobs) :])
    try:
        all_ref_logprobs = llm.get_batch_logprobs_token_ids(prompt_token_ids, completion_token_ids)
    except Exception as e:
        logger.error(f"Failed to get ref logprobs: {e}")
        return
    for trace, ref_logprobs in zip(traces, all_ref_logprobs):
        trace.ref_logprobs = [c["logprob"] for c in ref_logprobs["content"]]
        assert len(trace.ref_logprobs) == len(trace.logprobs), f"{len(trace.ref_logprobs)} != {len(trace.logprobs)}"


def convert_problems_to_tapes(problems: list, cfg: DictConfig) -> list[RLMathTape]:
    """
    Creates RLMathTape objects from a list of math problem dictionaries.

    Args:
        problems (list[dict]): List of dictionaries containing math problems, where each dict
            has 'question' and expected answer value. The list is created from a dataset.

    Returns:
        list[RLMathTape]: List of RLMathTape objects initialized with the math problems as Task steps.
            Each tape contains a single starting Task step with the question and expected answer value
            stored in metadata.
    """
    tapes: list[RLMathTape] = []
    for problem in tqdm(problems, desc="Converting problems to unique tapes", unit="problem"):
        start_step = Task(
            task=problem["task"],
            metadata=StepMetadata(
                other={
                    "value": problem["answer"],
                }
            ),
        )
        tape = RLMathTape(steps=[start_step], context=None)
        tapes.append(tape)
    return tapes


def extract_tape_training_samples(
    new_tape: RLMathTape, agent: CoTMathAgent, split_name: str, cfg: DictConfig
) -> Tuple[List[TrainingText], Dict[str, int]]:
    """
    Process a single tape to extract training samples and statistics.

    Args:
        new_tape: The tape to process containing math problem steps
        agent: CoTMathAgent
        split_name: Name of split ('train' or 'test')
        tapes_dir: Directory to save processed tapes
        cfg: Configuration
        llm_calls: List of LLM calls
        strict: check that every token matches between the vLLM and the HF tokenizer otherwise just compare their lengths

    Returns:
        Tuple containing:
        - List of training samples with rewards and logprobs
        - Dictionary with statistics (reward, steps, success, no_errors)
    """
    tape_prompt_tokens = 0
    tape_output_tokens = 0
    match cfg.dataset_name:
        case "math":
            eval_fn = eval_math
            extract_fn = extract_math_answer
        case "gsm8k":
            eval_fn = eval_last_single_answer
            extract_fn = extract_last_single_answer
        case "eurus":
            eval_fn = eval_math
            extract_fn = extract_math_answer
        case _:
            raise ValueError(f"Unknown dataset: {cfg.dataset_name}")

    if any([isinstance(step, LLMOutputParsingFailureAction) for step in new_tape.steps]):
        # LLM produced a step that was unparsable. Negative reward.
        no_error, reward, success = 0, -1, 0
    else:
        no_error = 1
        prediction = extract_fn(new_tape.steps[0].task, new_tape.steps[-1].reasoning, "cot")
        answer = new_tape.steps[0].metadata.other["value"]
        if eval_fn(
            {
                "prediction": prediction,
                "answer": answer,
            }
        ):
            # Correct answer
            reward, success = 1, 1
        else:
            # Incorrect answer or no answer
            reward, success = 0, 0

    training_samples: list[TrainingText] = []
    # For each LLM interaction in the tape:
    # - Create a training sample from the prompt and output
    # - Get log probabilities of the output tokens
    # - Set group ID for tracking
    for step in new_tape.steps:
        if "llm_call" not in step.metadata.other or step.metadata.other["llm_call"] is None:
            continue
        llm_call = step.metadata.other["llm_call"]

        if isinstance(llm_call, dict):
            llm_call = LLMCall(**llm_call)

        tape_prompt_tokens += llm_call.prompt_length_tokens
        tape_output_tokens += llm_call.output_length_tokens

        if split_name == "train":
            if llm_call.output_length_tokens >= cfg.llm.parameters.max_tokens:
                # Output is too long, ignore this sample
                # this will be recorded in output_tokens_overflow
                continue
            trace = agent.llm.make_training_text(llm_call.prompt, llm_call.output)

            input_ids = [lp.token_id for lp in llm_call.logprobs]
            labels = [lp.token_id for lp in llm_call.logprobs if lp.generated]
            # MASKED_TOKEN_ID is -100 and is the default "ignore_index" in nn.CrossEntropyLoss,
            # see https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
            labels = [MASKED_TOKEN_ID] * (len(input_ids) - len(labels)) + labels

            trace.input_ids = input_ids
            trace.labels = labels

            trace.reward = reward
            trace.logprobs = [lp.logprob for lp in llm_call.logprobs if lp.generated]
            trace.group_id = new_tape.metadata.parent_id
            training_samples.append(trace)

    tape_stats = {
        "reward": reward,
        "steps": len(new_tape.steps),
        "success": success,
        "no_error": no_error,
        "prompt_tokens": tape_prompt_tokens,
        "output_tokens": tape_output_tokens,
    }
    return training_samples, tape_stats


def batch_run_agent_replica(agent: CoTMathAgent, tapes: list[RLMathTape]) -> tuple[Agent, list[RLMathTape]]:
    final_tapes = agent.run_batch(tapes)
    # There is some statistics that we track in the agent in a mutable way
    return agent, final_tapes


def generate_training_data(
    agent_replicas: list[CoTMathAgent],
    tapes: list[RLMathTape],
    cfg: DictConfig,
    tapes_dir: Path,
    split_name: str,
) -> Tuple[list[CoTMathAgent], List[RLMathTape], List[TrainingText], Dict[str, float]]:
    """
    Generate complete tapes and training samples from a list of initialized tapes.

    Args:
        agent: Agent that interacts with the math environment
        tapes: List of tapes initialized with math problems
        cfg: Configuration
        tapes_dir: Directory to save processed episodes
        split_name: Name of split ('train' or other)

    Returns:
        Tuple containing:
        - List of completed RLMathTapes
        - List of training samples with rewards and logprobs
        - Dictionary of performance statistics and execution times
    """

    start_make_data = time.time()
    os.makedirs(tapes_dir, exist_ok=True)
    reward_stats = defaultdict(list)
    step_stats = defaultdict(list)
    no_errors_stats = defaultdict(list)
    success_stats = defaultdict(list)
    prompt_tokens_stats = defaultdict(list)
    output_tokens_stats = defaultdict(list)
    training_samples: List[TrainingText] = []

    logger.info(f"Run the agent on {cfg.dataset_name} {split_name}")

    start_making_tapes = time.time()
    with ProcessPoolExecutor(max_workers=len(agent_replicas)) as executor:
        replica_tapes = [tapes[i :: len(agent_replicas)] for i in range(len(agent_replicas))]
        results = list(executor.map(batch_run_agent_replica, agent_replicas, replica_tapes))
        final_tapes = list(chain(*[r[1] for r in results]))
        agent_replicas = [r[0] for r in results]
    logger.info(f"Making tapes took {time.time() - start_making_tapes}")

    for new_tape in tqdm(final_tapes, total=len(final_tapes), desc="Extracting training data from tapes", unit="tape"):
        tape_training_samples, tape_stats = extract_tape_training_samples(new_tape, agent_replicas[0], split_name, cfg)
        training_samples.extend(tape_training_samples)
        reward_stats[new_tape.metadata.parent_id].append(tape_stats["reward"])
        step_stats[new_tape.metadata.parent_id].append(tape_stats["steps"])
        success_stats[new_tape.metadata.parent_id].append(tape_stats["success"])
        no_errors_stats[new_tape.metadata.parent_id].append(tape_stats["no_error"])
        prompt_tokens_stats[new_tape.metadata.parent_id].append(tape_stats["prompt_tokens"])
        output_tokens_stats[new_tape.metadata.parent_id].append(tape_stats["output_tokens"])

    start_dump = time.time()
    with open(tapes_dir / "tapes.json", "w") as f:
        json.dump([tape.model_dump() for tape in final_tapes], f, indent=4)
    end_dump = time.time()

    end_make_data = time.time()

    stats = {
        **{f"{split_name}_{k}_reward": v for k, v in calculate_stats(reward_stats).items()},
        **{f"{split_name}_{k}_steps": v for k, v in calculate_stats(step_stats).items()},
        **{f"{split_name}_{k}_success": v for k, v in calculate_stats(success_stats).items()},
        **{f"{split_name}_{k}_no_errors": v for k, v in calculate_stats(no_errors_stats).items()},
        **{f"{split_name}_{k}_prompt_tokens": v for k, v in calculate_stats(prompt_tokens_stats).items()},
        **{f"{split_name}_{k}_output_tokens": v for k, v in calculate_stats(output_tokens_stats).items()},
        **{
            f"execution_time/{split_name}_dumping_tapes": end_dump - start_dump,
            f"execution_time/{split_name}_make_data": end_make_data - start_make_data,
            f"execution_time/{split_name}_tapes_made_per_second": len(final_tapes) / (end_make_data - start_make_data),
            f"{split_name}_prompt_tokens": sum([sum(pt) for pt in prompt_tokens_stats.values()]),
            f"{split_name}_output_tokens": sum([sum(ot) for ot in output_tokens_stats.values()]),
            f"{split_name}_output_tokens_overflow": sum(
                [(ot >= cfg.llm.parameters.max_tokens) for ot_list in output_tokens_stats.values() for ot in ot_list]
            )
            / len(final_tapes),
        },
    }
    return agent_replicas, final_tapes, training_samples, stats


@hydra.main(config_path="../../conf/", config_name="rl_gsm8k", version_base="1.3.2")
def main(cfg: DictConfig):
    multiprocessing.set_start_method("spawn")  # necessary to use gpus in subprocesses
    random.seed(42)
    exp_path = Path(cfg.output_dir)
    setup_logging(exp_path)
    logger.info(f"Current dir: {os.getcwd()}, output dir: {cfg.output_dir}")
    cfg.finetune.wandb_id = str(exp_path).replace("/", "_")
    run = init_wandb(cfg, exp_path, flatten_dict_config(cfg))
    if run is None:
        raise ValueError("Failed to initialize wandb run")
    state_path = exp_path / "rl_state.json"
    state = load_state(state_path)
    # optionally clean all data at start time
    if cfg.force_restart:
        clean_up(exp_path, state, state_path)

    match cfg.dataset_name:
        case "math":
            train_dataset_long_name = test_dataset_long_name = "hendrycks/competition_math"
            process_fn = process_math_test
        case "gsm8k":
            train_dataset_long_name = test_dataset_long_name = "openai/gsm8k"
            process_fn = process_gsm8k_test
        case "eurus":
            train_dataset_long_name = "PRIME-RL/Eurus-2-RL-Data"
            test_dataset_long_name = "alexpiche/math_test_cleaned"
            process_fn = process_eurus_test
        case _:
            raise ValueError(f"Unknown dataset: {cfg.dataset_name}")

    train_dataset = load_dataset(train_dataset_long_name, split="train", trust_remote_code=True)
    test_dataset = load_dataset(test_dataset_long_name, split="test", trust_remote_code=True)
    test_samples = [process_fn(s) for s in tqdm(test_dataset, desc="Processing test samples") if process_fn(s) is not None]
    train_samples = [process_fn(s) for s in tqdm(train_dataset, desc="Processing train samples") if process_fn(s) is not None]
    logger.info(f"Loaded {len(train_samples)} training samples")
    logger.info(f"Loaded {len(test_samples)} test samples")

    conf_dir = exp_path / "conf"
    os.makedirs(conf_dir, exist_ok=True)
    finetune_path = exp_path / "finetune"

    while state["iteration"] < cfg.max_iterations:
        logger.info(f"Starting iteration {state['iteration']}")
        start_iteration = time.time()
        if os.path.exists(finetune_path / "current"):
            assistant_model_path = str(finetune_path / "current")
        else:
            assistant_model_path = cfg.model_path

        try:
            all_results = {}
            with VLLMServiceManager(
                exp_path=exp_path,
                service_name="actor",
                model_name_or_path=assistant_model_path,
                port=8080,
                verbose=True,
                cuda_device=",".join([str(i) for i in range(torch.cuda.device_count())]),
                **(dict(cfg.vllm_config.vllm_kwargs) | dict(cfg.vllm_config.actor_vllm_kwargs)),
            ) as vllm_service_manager:
                sub_samples = random.sample(train_samples, cfg.max_agent_forks // cfg.attempts)
                train_tapes = convert_problems_to_tapes(sub_samples, cfg)
                train_tapes = [copy.deepcopy(tape) for tape in train_tapes for _ in range(cfg.attempts)]
                train_llms = [
                    TrainableLLM(
                        base_url=base_url,
                        model_name=str(assistant_model_path),
                        tokenizer_name=str(assistant_model_path),
                        parameters=cfg.llm.parameters,
                        use_cache=False,
                        collect_logprobs=True,
                        observe_llm_calls=False,
                        max_prompt_length=cfg.llm.max_prompt_length,
                    )
                    for base_url in vllm_service_manager.get_base_urls()
                ]

                test_llms = [
                    TrainableLLM(
                        base_url=base_url,
                        model_name=str(assistant_model_path),
                        tokenizer_name=str(assistant_model_path),
                        parameters=cfg.test_llm.parameters,
                        use_cache=False,
                        observe_llm_calls=False,
                        max_prompt_length=cfg.test_llm.max_prompt_length,
                    )
                    for base_url in vllm_service_manager.get_base_urls()
                ]

                train_agent_replicas = [CoTMathAgent.create(llm=llm) for llm in train_llms]

                splits = [("train", train_agent_replicas, train_tapes)]
                if state["iteration"] % cfg.test_every_n_iterations == 0 and cfg.test_every_n_iterations > 0:
                    test_tapes = convert_problems_to_tapes(test_samples, cfg)
                    test_agent_replicas = [CoTMathAgent.create(llm=llm) for llm in test_llms]
                    splits.append(("test", test_agent_replicas, test_tapes))
                for split_name, agent_replicas, tapes in splits:
                    tapes_dir = exp_path / "tapes" / split_name / str(state["iteration"])
                    agent_replicas_with_stats, new_tapes, split_training_samples, stats = generate_training_data(
                        agent_replicas, tapes, cfg, tapes_dir, split_name
                    )

                    llm_stats = agent_replicas_with_stats[0].llm.get_stats()
                    make_data_took = stats[f"execution_time/{split_name}_make_data"]
                    llm_stats = {f"llm/{split_name}_{k}": v for k, v in llm_stats.items()}
                    throughput_stats = {
                        f"{split_name}_prompt_tokens_per_sec": stats[f"{split_name}_prompt_tokens"] / make_data_took,
                        f"{split_name}_output_tokens_per_sec": stats[f"{split_name}_output_tokens"] / make_data_took,
                        f"{split_name}_total_tokens_per_sec": (
                            stats[f"{split_name}_prompt_tokens"] + stats[f"{split_name}_output_tokens"]
                        )
                        / make_data_took,
                    }
                    stats.update(llm_stats)
                    stats.update(throughput_stats)

                    all_results[split_name] = {
                        "new_tapes": new_tapes,
                        "training_samples": split_training_samples,
                        "stats": stats,
                    }

                    # Log results
                    logger.info(f"{cfg.dataset_name} {split_name} stats:")
                    for stat_name, stat_value in stats.items():
                        logger.info(f"{stat_name}: {stat_value}")
                assistant_vllm_stats = vllm_service_manager.get_stats()

        except Exception as e:
            logger.error(colored(f"Failed to solve task: {e}", "red"))
            raise e

        training_samples = all_results["train"]["training_samples"]
        logger.info(f"Collected {len(training_samples)} training samples")
        stats = all_results["train"]["stats"]
        if "test" in all_results:  # test is only present every cfg.test_every_n_iterations
            stats.update(all_results["test"]["stats"])
            time_evaluation = stats["execution_time/test_make_data"]
        else:
            time_evaluation = 0
        wandb.log(
            stats,
            step=state["iteration"],
        )

        try:
            with VLLMServiceManager(
                exp_path=exp_path,
                service_name="reference",
                model_name_or_path=cfg.model_path,
                port=8180,
                verbose=True,
                cuda_device=",".join([str(i) for i in range(torch.cuda.device_count())]),
                **(dict(cfg.vllm_config.vllm_kwargs) | dict(cfg.vllm_config.ref_vllm_kwargs)),
            ) as vllm_service_manager:
                ref_llms = [
                    TrainableLLM(
                        base_url=url,
                        model_name=cfg.model_path,
                        tokenizer_name=cfg.model_path,
                        parameters=dict(temperature=0.7),
                    )
                    for url in vllm_service_manager.get_base_urls()
                ]

                start_basemodel_logprobs = time.time()
                with ThreadPoolExecutor(
                    max_workers=cfg.get_logprobs_workers_per_gpu * torch.cuda.device_count()
                ) as executor:
                    chunk_size = 64
                    futures = []
                    for chunk_id, chunk_offset in enumerate(range(0, len(training_samples), chunk_size)):
                        ref_llm = ref_llms[chunk_id % len(ref_llms)]
                        chunk = training_samples[chunk_offset : chunk_offset + chunk_size]
                        futures.append(executor.submit(batch_annotate_traces_with_ref_logprobs, ref_llm, chunk))
                    # Reference logprobs are added in-place
                    futures = tqdm(as_completed(futures), total=len(futures), desc="Adding logprobs")
                    _ = [future.result() for future in futures]

                refmodel_vllm_stats = vllm_service_manager.get_stats()
                refmodel_starting_time = refmodel_vllm_stats["starting_time"]
                time_populating_ref_logprobs = time.time() - start_basemodel_logprobs

        except Exception as e:
            logger.error(colored(f"Failed to get ref log probs: {e}", "red"))
            raise e

        logprob_stats = {
            "execution_time/populating_ref_logprobs": time_populating_ref_logprobs,
            "execution_time/starting_assistantmodel_vllm": assistant_vllm_stats["starting_time"],
            "execution_time/starting_refmodel_vllm": refmodel_starting_time,
        }
        logger.info("Logprob population stats:")
        for stat_name, stat_value in logprob_stats.items():
            logger.info(f"{stat_name}: {stat_value}")
        wandb.log(logprob_stats, step=state["iteration"])
        rollout_dir = exp_path / "rollouts" / str(state["iteration"])
        os.makedirs(rollout_dir, exist_ok=True)
        with open(rollout_dir / "data.jsonl", "w") as f:
            for trace in training_samples:
                if cfg.use_rejection_sampling and trace.reward <= 0:
                    continue
                f.write(trace.model_dump_json() + "\n")
                f.flush()

        finetune_cfg = cfg.copy()

        checkpoint_steps = finetune_cfg.finetune.save_checkpoint_steps
        interrupt_train_steps = int((state["iteration"] + 1) * checkpoint_steps)

        finetune_cfg.finetune.interrupt_train_steps = interrupt_train_steps
        finetune_cfg.output_dir = str(finetune_path)
        finetune_cfg.finetune.data = {"data_parts_train": [{"path": str(rollout_dir)}]}
        finetune_cfg.finetune.wandb_id = run.id + "_finetune"
        finetune_cfg.finetune.wandb_name = run.name + "_finetune"
        finetune_cfg.finetune.wandb_resume = "always"
        config_path = conf_dir / f"{state['iteration']}.yaml"
        OmegaConf.save(finetune_cfg, config_path)

        start_finetune = time.time()
        launch_training(
            str(conf_dir),
            str(state["iteration"]),
            cfg.accelerate_cfg_path,
            use_deepspeed=cfg.use_deepspeed,  # defaults to False
        )
        time_finetune = time.time() - start_finetune
        time_iteration = time.time() - start_iteration
        wandb.log(
            {
                "execution_time/finetune": time_finetune,
                "execution_time/iteration": time_iteration,
                "execution_time/overhead": time_iteration
                - time_finetune
                - time_populating_ref_logprobs
                - time_evaluation
                - stats["execution_time/train_make_data"],
            },
            step=state["iteration"],
        )
        state["iteration"] += 1
        save_state(state, state_path)

    logger.info(f'Finished training after {state["iteration"]} iterations')


if __name__ == "__main__":
    main()
