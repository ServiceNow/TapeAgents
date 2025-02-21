import datetime
import logging
import copy
import json
import multiprocessing
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
import traceback
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from termcolor import colored
from tqdm import tqdm

import wandb

wandb.require("core")
from .cot_math_agent import (
    CoTMathAgent,
    MathEnvironment,
    RLMathTape,
    Task,
)
from .deepseek_math_eval.answer_extraction import extract_last_single_answer, extract_math_answer
from .deepseek_math_eval.eval_script import eval_last_single_answer, eval_math
from .deepseek_math_eval.process_utils import process_gsm8k_test, process_math_test
from .utils import (
    calculate_stats,
    clean_up,
    VLLMServiceManager,
    get_tokens_from_hf_tokenizer,
    launch_training,
    load_state,
    save_state,
    setup_logging,
)
from tapeagents.core import LLMOutputParsingFailureAction, StepMetadata, TrainingText
from tapeagents.llms import TrainableLLM
from tapeagents.orchestrator import main_loop
from .dist_utils import DistributedManager, init_wandb, flatten_dict_config

logger = logging.getLogger(__name__)


def annotate_traces_with_ref_logprobs(agent: CoTMathAgent, trace: TrainingText, strict: bool) -> TrainingText | None:
    try:
        prompt_token_ids, completion_token_ids = (
            trace.input_ids[: -len(trace.logprobs)],
            trace.input_ids[-len(trace.logprobs) :],
        )
        ref_logprobs = agent.llm.get_logprobs(prompt_token_ids, completion_token_ids)  # type: ignore
        trace.ref_logprobs = [c["logprob"] for c in ref_logprobs["content"]]
        assert len(trace.ref_logprobs) == len(trace.logprobs), f"{len(trace.ref_logprobs)} != {len(trace.logprobs)}"
        return trace
    except Exception as e:
        logger.error(f"Failed to get ref logprobs: {e}")
        if strict:
            raise e
        return None


def convert_problems_to_tapes(problems: list, cfg: DictConfig, split_name: str) -> list[RLMathTape]:
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
    for problem in tqdm(problems, desc=f"Converting {split_name} problems to unique tapes", unit="problem"):
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
) -> Tuple[RLMathTape, List[TrainingText], Dict[str, int]]:
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
    discarded = []
    tape_prompt_tokens = 0
    tape_output_tokens = 0
    match cfg.dataset_name:
        case "math":
            eval_fn = eval_math
            extract_fn = extract_math_answer
        case "gsm8k":
            eval_fn = eval_last_single_answer
            extract_fn = extract_last_single_answer
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
    if split_name == "train":
        # For each LLM interaction in the tape:
        # - Create a training sample from the prompt and output
        # - Get log probabilities of the output tokens
        # - Set group ID for tracking
        for step in new_tape.steps:
            if "llm_call" not in step.metadata.other or step.metadata.other["llm_call"] is None:
                continue
            llm_call = step.metadata.other["llm_call"]
            trace = agent.llm.make_training_text(llm_call.prompt, llm_call.output)

            input_ids = [lp["token_id"] for lp in llm_call.logprobs]
            labels = [lp["token_id"] for lp in llm_call.logprobs if lp["generated"]]
            # MASKED_TOKEN_ID is -100 and is the default "ignore_index" in nn.CrossEntropyLoss,
            # see https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
            labels = [MASKED_TOKEN_ID] * (len(input_ids) - len(labels)) + labels

            trace.input_ids = input_ids
            trace.labels = labels

            trace.reward = reward
            trace.logprobs = [lp["logprob"] for lp in llm_call.logprobs if lp["generated"]]
            trace.group_id = new_tape.metadata.parent_id
            tape_prompt_tokens += llm_call.prompt_length_tokens
            tape_output_tokens += llm_call.output_length_tokens
            training_samples.append(trace)

    tape_stats = {
        "reward": reward,
        "steps": len(new_tape.steps),
        "success": success,
        "no_error": no_error,
        "discarded": np.mean(discarded) if discarded else 0,
        "prompt_tokens": tape_prompt_tokens,
        "output_tokens": tape_output_tokens,
    }
    return new_tape, training_samples, tape_stats


def generate_training_data(
    agent: CoTMathAgent,
    tapes: list[RLMathTape],
    cfg: DictConfig,
    env: MathEnvironment,
    tapes_dir: Path,
    split_name: str,
) -> Tuple[List[RLMathTape], List[TrainingText], Dict[str, float]]:
    """
    Generate complete tapes and training samples from a list of initialized tapes.

    Args:
        agent: Agent that interacts with the math environment
        tapes: List of tapes initialized with math problems
        cfg: Configuration
        env: Environment with tools
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
    discarded_stats = defaultdict(list)
    training_samples: List[TrainingText] = []

    logger.info(f"Starting {cfg.dataset_name} {split_name} main loop")

    logger.info("Starting data creation")
    prompt_tokens = 0
    output_tokens = 0

    def generate_and_extract_tape_training_samples(
        tape: RLMathTape, agent: CoTMathAgent, env, split_name: str, cfg: DictConfig
    ):
        new_tape = main_loop(agent, tape, env, max_loops=cfg.max_loops).get_final_tape()
        assert new_tape.steps[1].reasoning == new_tape.steps[1].metadata.other["llm_call"].output.content
        return extract_tape_training_samples(new_tape, agent, split_name, cfg)

    with ThreadPoolExecutor(max_workers=cfg.n_workers_per_gpu * torch.cuda.device_count()) as executor:
        generate_and_extract_tape_training_samples_partial = partial(
            generate_and_extract_tape_training_samples,
            agent=agent,
            env=env,
            split_name=split_name,
            cfg=cfg,
        )
        futures = [executor.submit(generate_and_extract_tape_training_samples_partial, tape) for tape in tapes]
        # Wrap futures with tqdm for progress tracking
        new_tapes = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating tapes", unit="tape"):
            new_tape, tape_training_samples, tape_stats = future.result()
            new_tapes.append(new_tape)
            training_samples.extend(tape_training_samples)
            reward_stats[new_tape.metadata.parent_id].append(tape_stats["reward"])
            step_stats[new_tape.metadata.parent_id].append(tape_stats["steps"])
            success_stats[new_tape.metadata.parent_id].append(tape_stats["success"])
            no_errors_stats[new_tape.metadata.parent_id].append(tape_stats["no_error"])
            discarded_stats[new_tape.metadata.parent_id].append(tape_stats["discarded"])
            prompt_tokens += tape_stats["prompt_tokens"]
            output_tokens += tape_stats["output_tokens"]

    start_dump = time.time()
    with open(tapes_dir / "tapes.json", "w") as f:
        json.dump([tape.model_dump() for tape in new_tapes], f, indent=4)
    end_dump = time.time()

    end_make_data = time.time()

    stats = {
        **{f"{split_name}_{k}_reward": v for k, v in calculate_stats(reward_stats).items()},
        **{f"{split_name}_{k}_steps": v for k, v in calculate_stats(step_stats).items()},
        **{f"{split_name}_{k}_success": v for k, v in calculate_stats(success_stats).items()},
        **{f"{split_name}_{k}_no_errors": v for k, v in calculate_stats(no_errors_stats).items()},
        **{
            f"execution_time/{split_name}_dumping_tapes": end_dump - start_dump,
            f"execution_time/{split_name}_make_data": end_make_data - start_make_data,
            f"execution_time/{split_name}_tapes_made_per_second": len(new_tapes) / (end_make_data - start_make_data),
            f"{split_name}_discarded": np.mean([np.mean(v) for v in discarded_stats.values()]),
            f"{split_name}_prompt_tokens": prompt_tokens,
            f"{split_name}_output_tokens": output_tokens,
        },
    }
    # All nodes return their local results
    return new_tapes, training_samples, stats


def split_data_for_nodes(data, world_size, rank):
    """Split data into chunks for distributed processing."""
    if data is None:
        return None
        
    per_node = len(data) // world_size
    start_idx = rank * per_node
    # Last node gets any remaining samples
    end_idx = start_idx + per_node if rank < world_size - 1 else len(data)
    return data[start_idx:end_idx]


def safe_wandb_log(metrics, step, dist_manager: DistributedManager):
    if dist_manager.is_main_process():
        try:
            wandb.log(metrics, step=step)
        except Exception as e:
            logger.warning(f"Failed to log to wandb: {e}")


@hydra.main(config_path="../../conf/", config_name="rl_gsm8k", version_base="1.3.2")
def main(cfg: DictConfig):
    dist_manager = DistributedManager()
    
    multiprocessing.set_start_method("spawn")  # necessary to use gpus in subprocesses
    random.seed(42)
    exp_path = Path(cfg.output_dir)
    setup_logging(exp_path)
    logger.info(f"Current dir: {os.getcwd()}, output dir: {cfg.output_dir}")
    cfg.finetune.wandb_id = exp_path.name

    # Define directories
    conf_dir = exp_path / "conf"
    finetune_path = exp_path / "finetune"
    data_path = exp_path / "data"
    sync_dir = exp_path / "sync"

    run = init_wandb(cfg, exp_path, flatten_dict_config(cfg), dist_manager)

    state_path = exp_path / "rl_state.json"
    state = load_state(state_path)
    
    # Initialize data containers
    full_train_samples = None
    full_test_samples = None

    # distributed setup
    world_size = dist_manager.get_world_size()
    rank = dist_manager.get_rank()

    # Get dataset only on main process
    if dist_manager.is_main_process():
        if cfg.force_restart:
            clean_up(exp_path, state, state_path, dist_manager)

            match cfg.dataset_name:
                case "math":
                    dataset_long_name = "hendrycks/competition_math"
                    process_fn = process_math_test
                case "gsm8k":
                    dataset_long_name = "openai/gsm8k"
                    process_fn = process_gsm8k_test
                case _:
                    raise ValueError(f"Unknown dataset: {cfg.dataset_name}")

            train_dataset = load_dataset(dataset_long_name, "main", split="train", trust_remote_code=True)
            full_train_samples = [process_fn(s) for s in train_dataset]
            test_dataset = load_dataset(dataset_long_name, "main", split="test", trust_remote_code=True)
            full_test_samples = [process_fn(s) for s in test_dataset]
            
            # Save to shared storage
            os.makedirs(data_path, exist_ok=True)
            with open(data_path / "train_samples.json", "w") as f:
                json.dump(full_train_samples, f)
            with open(data_path / "test_samples.json", "w") as f:
                json.dump(full_test_samples, f)
            
            logger.info(f"Main process saved {len(full_train_samples)} training samples and {len(full_test_samples)} test samples")

    logger.info(f"Creating sync dir: {sync_dir}")
    os.makedirs(sync_dir, exist_ok=True)

    # Ensure data is written before other nodes try to read
    if not dist_manager.sync_nodes(message="after data save", sync_dir=sync_dir, rank=rank, world_size=world_size):
        raise RuntimeError("Failed sync after data save")

    if cfg.force_restart:
        # Wait for files to be available; important for distributed training
        max_retries = 10
        retry_delay = 5  # seconds
        
        for retry in range(max_retries):
            try:
                if not (data_path / "train_samples.json").exists() or not (data_path / "test_samples.json").exists():
                    if retry < max_retries - 1:
                        logger.warning(f"Rank {rank + 1}/{world_size}: Data files not found, retrying in {retry_delay} seconds (attempt {retry + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        continue
                    else:
                        raise FileNotFoundError("Data files not found after maximum retries")
                    
                with open(data_path / "train_samples.json", "r") as f:
                    full_train_samples = json.load(f)
                with open(data_path / "test_samples.json", "r") as f:
                    full_test_samples = json.load(f)
                break
            except Exception as e:
                if retry < max_retries - 1:
                    logger.warning(f"Rank {rank + 1}/{world_size}: Failed to load data files: {e}, retrying in {retry_delay} seconds")
                    time.sleep(retry_delay)
                else:
                    raise e
    else:
        try:
            rollout_dir = exp_path / "rollouts" / str(state["iteration"] - 1)
            with open(rollout_dir / "data.jsonl", "r") as f:
                full_train_samples = [json.loads(line) for line in f]
            full_test_samples = []
        except Exception as e:
            logger.error(f"Failed to load existing data: {e}")
            raise e

    # Split data for current node
    logger.info(f"Rank {rank + 1}/{world_size}: Total train samples before split: {len(full_train_samples)}")
    logger.info(f"Rank {rank + 1}/{world_size}: Total test samples before split: {len(full_test_samples)}")
    train_samples = split_data_for_nodes(full_train_samples, world_size, rank)
    test_samples = split_data_for_nodes(full_test_samples, world_size, rank)

    logger.info(f"Rank {rank + 1}/{world_size} loaded {len(train_samples)} training samples "
               f"and {len(test_samples)} test samples")

    # Create environment on all nodes
    env = MathEnvironment()
    os.makedirs(conf_dir, exist_ok=True)

    remove_leading_white_space = True if "deepseek" in cfg.model_path else False
    if remove_leading_white_space:
        logger.info("Removing leading white space from the model. This is necessary for DeepSeek models")

    while state["iteration"] < cfg.max_iterations:
        logger.info(f"Starting iteration {state['iteration']}")
        start_iteration = time.time()

        if os.path.exists(finetune_path / "current"):
            assistant_model_path = str(finetune_path / "current")
        else:
            assistant_model_path = cfg.model_path

        try:
            # Set NCCL settings for vLLM weight loading
            if dist_manager.get_world_size() > 1:
                os.environ["NCCL_CUMEM_ENABLE"] = "0"
                os.environ["NCCL_TIMEOUT"] = "7200"
                logger.info("Multi-node NCCL Environment Variables for vLLM:")
                for key in ["NCCL_CUMEM_ENABLE", "NCCL_TIMEOUT"]:
                    logger.info(f"{key}: {os.environ.get(key)}")

            with VLLMServiceManager(
                model_name_or_path=assistant_model_path,
                stdout_file_path=exp_path / f"assistant_vllm_stdout_rank{rank}.log",
                stderr_file_path=exp_path / f"assistant_vllm_stderr_rank{rank}.log",
                port=8080,
                gpus_per_model_instance=cfg.gpus_per_model_instance,
                verbose=True,
                cuda_device=",".join([str(i) for i in range(torch.cuda.device_count())]),
                **cfg.vllm_config.vllm_kwargs,
            ) as vllm_service_manager:
                # Each node already has its portion of samples from split_data_for_nodes
                sub_samples = random.sample(train_samples, cfg.max_agent_forks // (cfg.attempts * world_size))
                train_tapes = convert_problems_to_tapes(sub_samples, cfg, split_name="train")
                train_tapes = [copy.deepcopy(tape) for tape in train_tapes for _ in range(cfg.attempts)]

                llm = TrainableLLM(
                    base_url=vllm_service_manager.get_base_urls(),
                    model_name=str(assistant_model_path),
                    tokenizer_name=str(assistant_model_path),
                    parameters=cfg.llm.parameters,
                    use_cache=False,
                    collect_logprobs=True,
                    observe_llm_calls=False,
                )

                test_llm = TrainableLLM(
                    base_url=vllm_service_manager.get_base_urls(),
                    model_name=str(assistant_model_path),
                    tokenizer_name=str(assistant_model_path),
                    parameters=cfg.test_llm.parameters,
                    use_cache=False,
                    observe_llm_calls=False
                )

                train_agent = CoTMathAgent.create(llm=llm)

                splits = [("train", train_agent, train_tapes)]
                if state["iteration"] % cfg.test_every_n_iterations == 0 and cfg.test_every_n_iterations > 0:
                    test_tapes = convert_problems_to_tapes(test_samples, cfg, split_name="test")
                    test_agent = CoTMathAgent.create(llm=test_llm)
                    splits.append(("test", test_agent, test_tapes))

                # Process splits and gather results
                all_results = {}
                for split_name, agent, tapes in splits:
                    start_make_data = time.time()
                    tapes_dir = exp_path / "tapes" / split_name / str(state["iteration"])
                    new_tapes, training_samples, stats = generate_training_data(
                        agent, tapes, cfg, env, tapes_dir, split_name
                    )
                    make_data_took = time.time() - start_make_data

                    llm_stats = agent.llm.get_stats()

                    stats.update({
                        f"execution_time/{split_name}_make_data": make_data_took,
                        f"llm/{split_name}_make_data_output_tokens/s": llm_stats["total_prompt_tokens"] / make_data_took,
                        f"llm/{split_name}_make_data_prompt_tokens/s": llm_stats["total_output_tokens"] / make_data_took,
                        f"llm/{split_name}_make_data_tokens/s": (llm_stats["total_output_tokens"] + llm_stats["total_prompt_tokens"]) / make_data_took,
                    })

                    for k, v in llm_stats.items():
                        if "/s" in k:
                            stats[f"llm/{split_name}_{k}_per_gpu"] = v / torch.cuda.device_count()
                        else:
                            stats[f"llm/{split_name}_{k}"] = v

                    all_results[split_name] = {
                        "new_tapes": new_tapes,
                        "training_samples": training_samples,
                        "stats": stats,
                    }

                    # Log results for current node
                    logger.info(f"Rank {rank + 1}/{world_size} {split_name} stats:")
                    for stat_name, stat_value in stats.items():
                        logger.info(f"{stat_name}: {stat_value}")

                assistant_vllm_stats = vllm_service_manager.get_stats()

        except Exception as e:
            logger.error(colored(f"Failed on rank {rank + 1}/{world_size}: {e}", "red"))
            raise e

        # Ensure all nodes have completed their processing
        if not dist_manager.sync_nodes(message="after processing splits", sync_dir=sync_dir, rank=rank, world_size=world_size):
            raise RuntimeError("Failed sync after processing splits")

        logger.info(f"Rank {rank + 1}/{world_size} collected {len(training_samples)} training samples")

        stats = all_results["train"]["stats"]
        if "test" in all_results:  # test is only present every cfg.test_every_n_iterations
            stats.update(all_results["test"]["stats"])
            time_evaluation = stats["execution_time/test_make_data"]
        else:
            time_evaluation = 0

        safe_wandb_log(
            stats,
            step=state["iteration"],
            dist_manager=dist_manager
        )

        try:
            with VLLMServiceManager(
                model_name_or_path=cfg.model_path,
                stdout_file_path=exp_path / "basemodel_vllm_stdout.log",
                stderr_file_path=exp_path / "basemodel_vllm_stderr.log",
                port=8180,
                verbose=True,
                gpus_per_model_instance=cfg.gpus_per_model_instance,
                cuda_device=",".join([str(i) for i in range(torch.cuda.device_count())]),
                **cfg.vllm_config.vllm_kwargs,
            ) as vllm_service_manager:
                basemodel_llm = TrainableLLM(
                    base_url=vllm_service_manager.get_base_urls(),
                    model_name=cfg.model_path,
                    tokenizer_name=cfg.model_path,
                    parameters=dict(temperature=0.7),
                )

                basemodel_agent = CoTMathAgent.create(llm=basemodel_llm)

                start_basemodel_logprobs = time.time()
                
                node_samples = all_results["train"]["training_samples"]
                logger.info(f"Rank {rank + 1}/{world_size}: Assigned {len(node_samples)} samples to process")
                
                with ThreadPoolExecutor(
                    max_workers=cfg.get_logprobs_workers_per_gpu * torch.cuda.device_count()
                ) as executor:
                    futures = [
                        executor.submit(annotate_traces_with_ref_logprobs, basemodel_agent, trace, strict=False)
                        for trace in node_samples
                    ]
                    
                    failed_samples = 0
                    local_training_samples = []
                    for future in tqdm(
                        as_completed(futures), 
                        total=len(futures), 
                        desc=f"Node {rank + 1}/{world_size}: Adding logprobs"
                    ):
                        result = future.result()
                        if result is not None:
                            local_training_samples.append(result)
                        else:
                            failed_samples += 1

                    logger.info(
                        f"Rank {rank + 1}/{world_size}: Processed {len(local_training_samples)} samples successfully, "
                        f"failed {failed_samples} samples"
                    )

                tmp_dir = exp_path / "tmp" / str(state["iteration"])
                os.makedirs(tmp_dir, exist_ok=True)

                logger.info(f"Rank {rank + 1}/{world_size}: Saving {len(local_training_samples)} samples to temporary file")
                tmp_path = tmp_dir / f"rank_{rank}_samples.pkl"
                with open(tmp_path, 'wb') as f:
                    torch.save(local_training_samples, f)

                # Ensure all ranks have saved their data
                if not dist_manager.sync_nodes(message="after saving local samples", sync_dir=sync_dir, rank=rank, world_size=world_size):
                    raise RuntimeError("Failed sync after saving local samples")

                if dist_manager.is_main_process():
                    logger.info("Main process: Gathering samples from all ranks")
                    all_samples = []
                    total_samples = 0
                    
                    # Gather data generated by each rank
                    for r in range(world_size):
                        rank_path = tmp_dir / f"rank_{r}_samples.pkl"
                        try:
                            with open(rank_path, 'rb') as f:
                                rank_samples = torch.load(f)
                            samples_from_rank = len(rank_samples)
                            total_samples += samples_from_rank
                            all_samples.extend(rank_samples)
                            logger.info(f"Main process: Loaded {samples_from_rank} samples from rank {r}/{world_size-1}")
                        except Exception as e:
                            logger.error(f"Failed to load samples from rank {r}: {e}")
                            raise

                    logger.info(f"Main process: Successfully gathered {len(all_samples)} total samples "
                               f"(sum across {world_size} ranks: {total_samples})")

                    rollout_dir = exp_path / "rollouts" / str(state["iteration"])
                    os.makedirs(rollout_dir, exist_ok=True)
                    
                    samples_written = 0
                    samples_rejected = 0
                    with open(rollout_dir / "data.jsonl", "w") as f:
                        for trace in all_samples:
                            if cfg.use_rejection_sampling and trace.reward <= 0:
                                samples_rejected += 1
                                continue
                            f.write(trace.model_dump_json() + "\n")
                            f.flush()
                            samples_written += 1
                    
                    logger.info(f"Main process: Wrote {samples_written} samples to disk, "
                                f"rejected {samples_rejected} samples due to negative reward")

                    # clean up
                    try:
                        import shutil
                        shutil.rmtree(tmp_dir)
                        logger.info(f"Main process: Cleaned up temporary directory {tmp_dir}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up temporary directory: {e}")

                else:
                    # Non-main processes just wait for completion
                    logger.info(f"Rank {rank + 1}/{world_size}: Completed data saving")

                # Ensure main process has finished processing before continuing
                if not dist_manager.sync_nodes(message="after gathering samples", sync_dir=sync_dir, rank=rank, world_size=world_size):
                    raise RuntimeError("Failed sync after gathering samples")

                refmodel_vllm_stats = vllm_service_manager.get_stats()
                refmodel_starting_time = refmodel_vllm_stats["starting_time"]
                time_populating_ref_logprobs = time.time() - start_basemodel_logprobs
                
                logger.info(
                    f"Rank {rank + 1}/{world_size}: Completed logprob annotation in "
                    f"{time_populating_ref_logprobs:.2f} seconds"
                )

        except Exception as e:
            logger.error(
                colored(
                    f"Rank {rank + 1}/{world_size}: Failed to get ref log probs with error: {str(e)}\n"
                    f"Traceback: {traceback.format_exc()}", 
                    "red"
                )
            )
            raise e

        logprob_stats = {
            "execution_time/populating_ref_logprobs": time_populating_ref_logprobs,
            "execution_time/starting_assistantmodel_vllm": assistant_vllm_stats["starting_time"],
            "execution_time/starting_refmodel_vllm": refmodel_starting_time,
        }
        logger.info(f"Logprob population stats:")
        for stat_name, stat_value in logprob_stats.items():
            logger.info(f"{stat_name}: {stat_value}")
        safe_wandb_log(logprob_stats, step=state["iteration"], dist_manager=dist_manager)

        # Generate config only on main process
        if dist_manager.is_main_process():
            finetune_cfg = cfg.copy()
            checkpoint_steps = finetune_cfg.finetune.save_checkpoint_steps
            interrupt_train_steps = int((state["iteration"] + 1) * checkpoint_steps - 1)
            finetune_cfg.finetune.interrupt_train_steps = interrupt_train_steps
            finetune_cfg.output_dir = str(finetune_path)
            finetune_cfg.finetune.data = {"data_parts_train": [{"path": str(rollout_dir)}]}
            finetune_cfg.finetune.wandb_id = run.id + "_finetune"
            finetune_cfg.finetune.wandb_name = run.name + "_finetune"
            finetune_cfg.finetune.wandb_resume = "always"
            config_path = conf_dir / f"{state['iteration']}.yaml"
            OmegaConf.save(finetune_cfg, config_path)

        # Ensure config file is written before training
        if not dist_manager.sync_nodes(message="before training", sync_dir=sync_dir, rank=rank, world_size=world_size, timeout_mins=10):
            raise RuntimeError("Failed sync after config save")

        # Now all nodes have the same config
        start_finetune = time.time()
        launch_training(
            str(conf_dir), 
            str(state["iteration"]),
            cfg.accelerate_cfg_path,
            use_deepspeed=cfg.use_deepspeed,
            dist_manager=dist_manager
        )
        
        # Sync after training completion
        if not dist_manager.sync_nodes(message="after training completion", sync_dir=sync_dir, rank=rank, world_size=world_size, timeout_mins=60):
            raise RuntimeError("Failed sync after training")

        time_finetune = time.time() - start_finetune
        time_iteration = time.time() - start_iteration
        safe_wandb_log(
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
            dist_manager=dist_manager
        )
        state["iteration"] += 1
        save_state(state, state_path, dist_manager)

        dist_manager.cleanup_gpu_resources()
        time.sleep(5)

    logger.info(f'Finished training after {state["iteration"]} iterations')


if __name__ == "__main__":
    main()