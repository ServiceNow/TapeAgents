import copy
import json
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
import numpy as np
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from termcolor import colored
from tqdm import tqdm

import wandb
from examples.rl_gsm8k.cot_math_agent import (
    CoTMathAgent,
    MathEnvironment,
    RLMathTape,
    Task,
)
from examples.rl_gsm8k.deepseek_math_eval.answer_extraction import extract_last_single_answer, extract_math_answer
from examples.rl_gsm8k.deepseek_math_eval.eval_script import eval_last_single_answer, eval_math
from examples.rl_gsm8k.deepseek_math_eval.process_utils import process_gsm8k_test, process_math_test
from examples.rl_gsm8k.utils import (
    VLLMServiceManager,
    calculate_stats,
    clean_up,
    get_tokens_from_hf_tokenizer,
    launch_training,
    load_state,
    save_state,
    setup_logging,
)
from tapeagents.batch import batch_main_loop
from tapeagents.core import LLMOutputParsingFailureAction, StepMetadata, TrainingText
from tapeagents.finetune.logging_ import flatten_dict_config, init_wandb
from tapeagents.llms import TrainableLLM
from tapeagents.observe import LLMCall, SQLiteWriterThread, retrieve_all_llm_calls
from tapeagents.orchestrator import main_loop

logger = logging.getLogger(__name__)


def annotate_trace_with_ref_logprobs(agent: CoTMathAgent, trace: TrainingText, strict: bool) -> TrainingText | None:
    try:
        ref_logprobs = agent.llm.get_logprobs(trace.prompt_text, trace.output_text)  # type: ignore
        trace.ref_logprobs = [c["logprob"] for c in ref_logprobs["content"]]
        return trace
    except Exception as e:
        logger.error(f"Failed to get ref logprobs: {e}")
        if strict:
            raise e
        return None


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
    compute_logprobs = []
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

            hf_tokens = get_tokens_from_hf_tokenizer(agent.llm.tokenizer, llm_call.prompt, llm_call.output)

            logprobs = []
            vllm_tokens = []
            if hasattr(llm_call.output, "logprobs"):
                logprobs_dict = llm_call.output.logprobs
                logprobs = [c["logprob"] for c in logprobs_dict["content"]]
                vllm_tokens = [c["token"] for c in logprobs_dict["content"]]

            # Note: tokens produced during generation are not always the same as the tokens produced on the full sequence
            if vllm_tokens != hf_tokens:
                # the online vLLM tokenizer does not agree with the HF tokenizer
                try:
                    logprobs_dict = agent.llm.get_logprobs(trace.prompt_text, trace.output_text)  # type: ignore
                    logprobs = [c["logprob"] for c in logprobs_dict["content"]]
                    new_vllm_tokens = [c["token"] for c in logprobs_dict["content"]]
                    assert len(new_vllm_tokens) == len(hf_tokens), "Token mismatch"
                    compute_logprobs.append(1)
                except Exception as e:
                    logger.error(f"Failed to get logprobs: {e}")
                    discarded.append(1)
                    continue
            else:
                compute_logprobs.append(0)

            trace.reward = reward
            trace.logprobs = logprobs
            trace.group_id = new_tape.metadata.parent_id
            tape_prompt_tokens += llm_call.prompt_length_tokens
            tape_output_tokens += llm_call.output_length_tokens
            if (
                len(trace.logprobs) == llm_call.output_length_tokens
                and (llm_call.prompt_length_tokens + llm_call.output_length_tokens) < cfg.finetune.seq_length
            ):
                training_samples.append(trace)
                discarded.append(0)
            else:
                discarded.append(1)
    tape_stats = {
        "reward": reward,
        "steps": len(new_tape.steps),
        "success": success,
        "no_error": no_error,
        "discarded": np.mean(discarded) if discarded else 0,
        "prompt_tokens": tape_prompt_tokens,
        "output_tokens": tape_output_tokens,
        "compute_logprobs": np.mean(compute_logprobs) if compute_logprobs else 0,
    }
    return new_tape, training_samples, tape_stats


def faster_generate_training_data(
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
    compute_logprobs_stats = defaultdict(list)
    training_samples: List[TrainingText] = []

    logger.info(f"Starting {cfg.dataset_name} {split_name} main loop")

    logger.info("Starting data creation")
    start_annotate_tape = time.time()
    prompt_tokens = 0
    output_tokens = 0

    def generate_and_extract_tape_training_samples(
        tape: RLMathTape, agent: CoTMathAgent, env, split_name: str, cfg: DictConfig
    ):
        new_tape = main_loop(agent, tape, env, max_loops=cfg.max_loops).get_final_tape()
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
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tapes", unit="tape"):
            new_tape, tape_training_samples, tape_stats = future.result()
            new_tapes.append(new_tape)
            training_samples.extend(tape_training_samples)
            reward_stats[new_tape.metadata.parent_id].append(tape_stats["reward"])
            step_stats[new_tape.metadata.parent_id].append(tape_stats["steps"])
            success_stats[new_tape.metadata.parent_id].append(tape_stats["success"])
            no_errors_stats[new_tape.metadata.parent_id].append(tape_stats["no_error"])
            discarded_stats[new_tape.metadata.parent_id].append(tape_stats["discarded"])
            compute_logprobs_stats[new_tape.metadata.parent_id].append(tape_stats["compute_logprobs"])
            prompt_tokens += tape_stats["prompt_tokens"]
            output_tokens += tape_stats["output_tokens"]

    end_annotate_tape = time.time()
    with open(tapes_dir / "tapes.json", "w") as f:
        json.dump([tape.model_dump() for tape in new_tapes], f, indent=4)

    end_make_data = time.time()

    stats = {
        **{f"{split_name}_{k}_reward": v for k, v in calculate_stats(reward_stats).items()},
        **{f"{split_name}_{k}_steps": v for k, v in calculate_stats(step_stats).items()},
        **{f"{split_name}_{k}_success": v for k, v in calculate_stats(success_stats).items()},
        **{f"{split_name}_{k}_no_errors": v for k, v in calculate_stats(no_errors_stats).items()},
        **{
            f"execution_time/{split_name}_annotate_tapes": end_annotate_tape - start_annotate_tape,
            f"execution_time/{split_name}_make_data": end_make_data - start_make_data,
            f"execution_time/{split_name}_tapes_made_per_second": len(new_tapes) / (end_make_data - start_make_data),
            f"{split_name}_discarded": np.mean([np.mean(v) for v in discarded_stats.values()]),
            f"{split_name}_compute_logprobs": np.mean([np.mean(v) for v in compute_logprobs_stats.values()]),
            f"{split_name}_prompt_tokens": prompt_tokens,
            f"{split_name}_output_tokens": output_tokens,
        },
    }
    return new_tapes, training_samples, stats


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
    compute_logprobs_stats = defaultdict(list)
    training_samples: List[TrainingText] = []

    logger.info(f"Starting {cfg.dataset_name} {split_name} main loop")
    start_sampling_from_llm = time.time()
    main_loops = batch_main_loop(
        agent, tapes, env, max_loops=cfg.max_loops, n_workers=cfg.n_workers_per_gpu * torch.cuda.device_count()
    )
    new_tapes = list(tqdm(main_loops, total=len(tapes), desc="Run the agent", unit="tape"))
    end_sampling_from_llm = time.time()

    start_dumping_tapes = time.time()
    with open(tapes_dir / "tapes.json", "w") as f:
        json.dump([tape.model_dump() for tape in new_tapes], f, indent=4)
    time_dumping_tapes = time.time() - start_dumping_tapes

    logger.info("Starting data creation")
    start_annotate_tape = time.time()
    prompt_tokens = 0
    output_tokens = 0

    with ThreadPoolExecutor(max_workers=cfg.n_workers_per_gpu * torch.cuda.device_count()) as executor:
        extract_tape_training_samples_partial = partial(
            extract_tape_training_samples,
            agent=agent,
            split_name=split_name,
            cfg=cfg,
        )
        futures = [executor.submit(extract_tape_training_samples_partial, tape) for tape in tapes]
        # Wrap futures with tqdm for progress tracking
        new_tapes = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tapes", unit="tape"):
            new_tape, tape_training_samples, tape_stats = future.result()
            new_tapes.append(new_tape)
            training_samples.extend(tape_training_samples)
            reward_stats[new_tape.metadata.parent_id].append(tape_stats["reward"])
            step_stats[new_tape.metadata.parent_id].append(tape_stats["steps"])
            success_stats[new_tape.metadata.parent_id].append(tape_stats["success"])
            no_errors_stats[new_tape.metadata.parent_id].append(tape_stats["no_error"])
            discarded_stats[new_tape.metadata.parent_id].append(tape_stats["discarded"])
            compute_logprobs_stats[new_tape.metadata.parent_id].append(tape_stats["compute_logprobs"])
            prompt_tokens += tape_stats["prompt_tokens"]
            output_tokens += tape_stats["output_tokens"]

    end_annotate_tape = time.time()
    with open(tapes_dir / "tapes.json", "w") as f:
        json.dump([tape.model_dump() for tape in new_tapes], f, indent=4)

    end_make_data = time.time()

    stats = {
        **{f"{split_name}_{k}_reward": v for k, v in calculate_stats(reward_stats).items()},
        **{f"{split_name}_{k}_steps": v for k, v in calculate_stats(step_stats).items()},
        **{f"{split_name}_{k}_success": v for k, v in calculate_stats(success_stats).items()},
        **{f"{split_name}_{k}_no_errors": v for k, v in calculate_stats(no_errors_stats).items()},
        **{
            f"execution_time/{split_name}_sampling_from_llm": end_sampling_from_llm - start_sampling_from_llm,
            f"execution_time/{split_name}_annotate_tapes": end_annotate_tape - start_annotate_tape,
            f"execution_time/{split_name}_make_data": end_make_data - start_make_data,
            f"execution_time/{split_name}_tapes_made_per_second": len(new_tapes) / (end_make_data - start_make_data),
            f"execution_time/{split_name}_output_tokens_per_second": output_tokens
            / (end_sampling_from_llm - start_sampling_from_llm),
            f"execution_time/{split_name}_prompt_tokens_per_second": prompt_tokens
            / (end_sampling_from_llm - start_sampling_from_llm),
            f"execution_time/{split_name}_dumping_tapes": time_dumping_tapes,
            f"{split_name}_discarded": np.mean([np.mean(v) for v in discarded_stats.values()]),
            f"{split_name}_compute_logprobs": np.mean([np.mean(v) for v in compute_logprobs_stats.values()]),
            f"{split_name}_prompt_tokens": prompt_tokens,
            f"{split_name}_output_tokens": output_tokens,
        },
    }
    return new_tapes, training_samples, stats


@hydra.main(config_path="../../conf/", config_name="rl_gsm8k", version_base="1.3.2")
def main(cfg: DictConfig):
    multiprocessing.set_start_method("spawn")  # necessary to use gpus in subprocesses
    random.seed(42)
    exp_path = Path(cfg.output_dir)
    setup_logging(exp_path)
    logger.info(f"Current dir: {os.getcwd()}, output dir: {cfg.output_dir}")
    cfg.finetune.wandb_id = exp_path.name
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
            dataset_long_name = "hendrycks/competition_math"
            process_fn = process_math_test
        case "gsm8k":
            dataset_long_name = "openai/gsm8k"
            process_fn = process_gsm8k_test
        case _:
            raise ValueError(f"Unknown dataset: {cfg.dataset_name}")

    train_dataset = load_dataset(dataset_long_name, "main", split="train")
    train_samples = [process_fn(s) for s in train_dataset]
    test_dataset = load_dataset(dataset_long_name, "main", split="test")
    test_samples = [process_fn(s) for s in test_dataset]
    logging.info(f"Loaded {len(train_samples)} training samples")
    logging.info(f"Loaded {len(test_samples)} test samples")

    env = MathEnvironment()

    os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(exp_path, "llm_calls.sqlite")
    conf_dir = exp_path / "conf"
    os.makedirs(conf_dir, exist_ok=True)
    finetune_path = exp_path / "finetune"
    remove_leading_white_space = True if "deepseek" in cfg.model_path else False
    if remove_leading_white_space:
        # vLLM sometimes generate a leading white space https://github.com/vllm-project/vllm/issues/3935
        logging.info("Removing leading white space from the model. This is necessary for DeepSeek models")

    while state["iteration"] < cfg.max_iterations:
        start_iteration = time.time()
        if os.path.exists(finetune_path / "current"):
            assistant_model_path = str(finetune_path / "current")
        else:
            assistant_model_path = cfg.model_path

        try:
            all_results = {}
            with VLLMServiceManager(
                model_name_or_path=assistant_model_path,
                stdout_file_path=exp_path / "assistant_vllm_stdout.log",
                stderr_file_path=exp_path / "assistant_vllm_stderr.log",
                port=8080,
                gpus_per_model_instance=cfg.gpus_per_model_instance,
                verbose=True,
                cuda_device=",".join([str(i) for i in range(torch.cuda.device_count())]),
                **cfg.vllm_config.vllm_kwargs,
            ) as vllm_service_manager:
                sub_samples = random.sample(train_samples, cfg.max_agent_forks // cfg.attempts)
                train_tapes = convert_problems_to_tapes(sub_samples, cfg)
                train_tapes = [copy.deepcopy(tape) for tape in train_tapes for _ in range(cfg.attempts)]
                llm = TrainableLLM(
                    base_url=vllm_service_manager.get_base_urls(),
                    model_name=str(assistant_model_path),
                    tokenizer_name=str(assistant_model_path),
                    parameters=cfg.llm.parameters,
                    use_cache=False,
                    collect_logprobs=True,
                    remove_leading_white_space=remove_leading_white_space,
                    log_llm_call_to_sqlite=False,
                    log_llm_call_to_tape=True,
                )

                test_llm = TrainableLLM(
                    base_url=vllm_service_manager.get_base_urls(),
                    model_name=str(assistant_model_path),
                    tokenizer_name=str(assistant_model_path),
                    parameters=cfg.test_llm.parameters,
                    use_cache=False,
                    remove_leading_white_space=remove_leading_white_space,
                    log_llm_call_to_sqlite=False,
                    log_llm_call_to_tape=True,
                )

                train_agent = CoTMathAgent.create(llm=llm)

                splits = [("train", train_agent, train_tapes)]
                if state["iteration"] % cfg.test_every_n_iterations == 0 and cfg.test_every_n_iterations > 0:
                    test_tapes = convert_problems_to_tapes(test_samples, cfg)
                    test_agent = CoTMathAgent.create(llm=test_llm)
                    splits.append(("test", test_agent, test_tapes))
                for split_name, agent, tapes in splits:
                    tapes_dir = exp_path / "tapes" / split_name / str(state["iteration"])
                    if cfg.faster_generate_training_data:
                        new_tapes, training_samples, stats = faster_generate_training_data(
                            agent, tapes, cfg, env, tapes_dir, split_name
                        )
                    else:
                        new_tapes, training_samples, stats = generate_training_data(
                            agent, tapes, cfg, env, tapes_dir, split_name
                        )

                    llm_stats = agent.llm.get_stats()
                    more_llm_stats = {}
                    for k, v in llm_stats.items():
                        if "per_second" in k:
                            more_llm_stats.update({f"{k}_per_gpu": v / torch.cuda.device_count()})
                    llm_stats.update(more_llm_stats)
                    llm_stats = {f"llm/{split_name}_{k}": v for k, v in llm_stats.items()}
                    stats.update(llm_stats)

                    all_results[split_name] = {
                        "new_tapes": new_tapes,
                        "training_samples": training_samples,
                        "stats": stats,
                    }

                    # Log results
                    logger.info(f"{cfg.dataset_name} {split_name.capitalize()} Results:")
                    for stat_name, stat_value in stats.items():
                        logger.info(f"{stat_name}: {stat_value}")
                assistant_vllm_stats = vllm_service_manager.get_stats()

        except Exception as e:
            logger.error(colored(f"Failed to solve task: {e}", "red"))
            raise e

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

        if state["iteration"] + 1 == cfg.max_iterations:
            # No need to continue if we are at the final iteration
            break

        start_basemodel_logprobs = time.time()
        try:
            with VLLMServiceManager(
                model_name_or_path=cfg.model_path,
                stdout_file_path=exp_path / "basemodel_vllm_stdout.log",
                stderr_file_path=exp_path / "basemodel_vllm_stderr.log",
                port=8081,
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

                with ThreadPoolExecutor(
                    max_workers=cfg.get_logprobs_workers_per_gpu * torch.cuda.device_count()
                ) as executor:
                    futures = [
                        executor.submit(annotate_trace_with_ref_logprobs, basemodel_agent, trace, strict=False)
                        for trace in all_results["train"]["training_samples"]
                    ]
                    training_samples: List[TrainingText] = [  # type: ignore
                        future.result()
                        for future in tqdm(as_completed(futures), total=len(futures), desc="Annotating traces")
                        if future.result() is not None
                    ]
                refmodel_vllm_stats = vllm_service_manager.get_stats()
                refmodel_starting_time = refmodel_vllm_stats["starting_time"]

        except Exception as e:
            logger.error(colored(f"Failed to get ref log probs: {e}", "red"))
            raise e

        time_populating_ref_logprobs = time.time() - start_basemodel_logprobs
        wandb.log(
            {
                "execution_time/populating_ref_logprobs": time_populating_ref_logprobs,
                "execution_time/starting_assistantmodel_vllm": assistant_vllm_stats["starting_time"],
                "execution_time/starting_refmodel_vllm": refmodel_starting_time,
            },
            step=state["iteration"],
        )
        rollout_dir = exp_path / "rollouts" / str(state["iteration"])
        os.makedirs(rollout_dir, exist_ok=True)
        with open(rollout_dir / "data.jsonl", "w") as f:
            for trace in training_samples:
                if cfg.use_rejection_sampling and trace.reward <= 0:
                    continue
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
        launch_training(str(conf_dir), str(state["iteration"]), cfg.accelerate_cfg_path)
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


if __name__ == "__main__":
    main()
