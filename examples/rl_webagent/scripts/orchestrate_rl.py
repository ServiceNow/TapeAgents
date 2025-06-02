import copy
import json
import logging
import multiprocessing
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
import torch
import wandb
from browsergym.miniwob import ALL_MINIWOB_TASKS
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from termcolor import colored
from tqdm import tqdm

from tapeagents.core import LLMCall, LLMOutputParsingFailureAction, TrainingText
from tapeagents.finetune.data import MASKED_TOKEN_ID
from tapeagents.finetune.logging_ import flatten_dict_config, init_wandb
from tapeagents.llms import TrainableLLM
from tapeagents.llms.trainable import trainable_llm_make_training_text
from tapeagents.orchestrator import main_loop
from tapeagents.tools.simple_browser import PageObservation

from ..agent import WebAgent, WebTape
from ..environment import WebEnvironment
from ..steps import WebTapeMetadata
from ..utils import (
    VLLMServiceManager,
    calculate_stats,
    clean_up,
    launch_training,
    load_state,
    save_state,
    setup_logging,
    tqdm_joblib,
)

logger = logging.getLogger(__name__)


def load_webtasks_debug():
    logger.info(f"Loading {len(ALL_MINIWOB_TASKS)} MiniWoB tasks")

    # load tasks where we don't always have 100% or 0% success rate
    DEBUG_SPLIT = [
        "miniwob.buy-ticket",
        "miniwob.bisect-angle",
        "miniwob.choose-list",
        "miniwob.click-checkboxes-large",
        "miniwob.click-checkboxes-soft",
        ### Massimo Easy Split ###
        # "miniwob.click-color",
        # "miniwob.click-test-2",
        # "miniwob.click-test-transfer",
        # "miniwob.enter-password",
        # "miniwob.focus-text-2",
        # "miniwob.identify-shape",
        # "miniwob.navigate-tree",
        # "miniwob.phone-book",
        # "miniwob.read-table",
        # "miniwob.use-autocomplete",
        # "miniwob.use-autocomplete",
        # "miniwob.buy-ticket",
        # "miniwob.click-checkboxes-soft",
        # "miniwob.click-collapsible-2",
        # "miniwob.click-collapsible-2-nodelay",
        # "miniwob.click-collapsible-nodelay",
        # "miniwob.click-dialog-2",
        # "miniwob.click-tab-2",
        # "miniwob.click-tab-2-medium",
        # "miniwob.form-sequence-3",
        # "miniwob.hot-cold",
        # "miniwob.multi-orderings",
        # "miniwob.tic-tac-toe",
        # "miniwob.use-autocomplete-nodelay"
    ]
    train_tasks = [t for t in ALL_MINIWOB_TASKS if t.get_task_id() in DEBUG_SPLIT]
    test_tasks = [t for t in ALL_MINIWOB_TASKS if t.get_task_id() in DEBUG_SPLIT]

    train_samples = [
        {"dataset": "miniwob", "task": task, "seed": 0}
        for task in train_tasks
        ### massimo setup ###
        # {"dataset": "miniwob", "task": task, "seed": np.random.randint(0, 1000)}
        # for task in train_tasks
        # for _ in range(2)
    ]

    test_samples = [
        {"dataset": "miniwob", "task": task, "seed": 0}
        for task in test_tasks
        ### massimo setup ###
        # {"dataset": "miniwob", "task": task, "seed": s}
        # for task in test_tasks
        # for s in range(12)
    ]

    logger.info(f"Loaded {len(train_samples)} training samples")
    logger.info(f"Loaded {len(test_samples)} test samples")
    return train_samples, test_samples


def load_webtasks(train_split: float = 0.6, seeds: list[int] = [0, 1, 2, 3, 4]) -> Tuple[list[dict], list[dict]]:
    """
    Load web tasks from the MiniWoB dataset.

    Returns:
        Tuple[list[dict], list[dict]]: Train and test task lists
    """
    logger.info(f"Loading {len(ALL_MINIWOB_TASKS)} MiniWoB tasks")
    n_train_tasks = int(len(ALL_MINIWOB_TASKS) * train_split)
    n_test_tasks = len(ALL_MINIWOB_TASKS) - n_train_tasks
    logger.info(f"Splitting {len(ALL_MINIWOB_TASKS)} tasks into {n_train_tasks} train and {n_test_tasks} test")

    train_tasks = ALL_MINIWOB_TASKS[:n_train_tasks]
    test_tasks = ALL_MINIWOB_TASKS[n_train_tasks:]

    train_samples = [
        {"dataset": "miniwob", "task": task, "seed": seed, "max_loops": 10} for task in train_tasks for seed in seeds
    ]

    test_samples = [
        {"dataset": "miniwob", "task": task, "seed": seed, "max_loops": 10} for task in test_tasks for seed in seeds
    ]

    logger.info(f"Loaded {len(train_samples)} training samples")
    logger.info(f"Loaded {len(test_samples)} test samples")
    return train_samples, test_samples


def run_agent(agent: WebAgent, env: WebEnvironment, task: dict) -> WebTape:
    """
    Run agent on a single web task with its environment.

    Args:
        agent: The web agent
        env: The web environment for this task
        task: Task dict containing 'task' (MiniWoB task) and 'seed'

    Returns:
        WebTape: Completed tape
    """
    # Initialize task in environment - this creates the initial tape with web observation and task
    tape, metadata = env.start_task(task)
    # metadata is a dict with: str:name, str:goal, dict:task_info as a result of running gym_env.reset(seed=seed)
    try:
        # Run agent-environment loop
        tape = main_loop(agent, tape, env, max_loops=task.get("max_loops", 5)).get_final_tape()

        # result is a dict of reward, stop, message, info
        success, result = env.validate_task(tape)
        final_reward = 1.0 if success else 0.0
        tape.metadata.result = {"success": success, **result, "final_reward": final_reward}
    except Exception as e:
        logger.error(f"Failed to run task: {e}")
        tape.metadata.result = {"success": False, "reward": 0.0, "final_reward": 0.0, "error": str(e)}

    # Close the environment
    env.finish_task()

    # Convert tape metadata to WebTapeMetadata if needed (not needed when no new step is added)
    if not isinstance(tape.metadata, WebTapeMetadata):
        tape.metadata = WebTapeMetadata(
            **tape.metadata.model_dump(),
            seed=task["seed"],
            task_name=task["task"].get_task_id(),
        )
    return tape


def tape_contains_an_error(tape: WebTape) -> bool:
    """
    Returns true if the tape ends with an error, ie if one of the following is true:
    - the last step is an LLMOutputParsingFailureAction
    - the tape metadata has an error
    - the last step is a PageObservation with an error
    """
    return (
        isinstance(tape.steps[-1], LLMOutputParsingFailureAction)
        or tape.metadata.result.get("error") is not None
        or (isinstance(tape.steps[-1], PageObservation) and tape.steps[-1].error)
    )


def extract_tape_training_samples_and_stats(
    new_tape: WebTape, split_name: str, tokenizer: Any
) -> Tuple[List[TrainingText], Dict[str, int]]:
    """
    Process a single tape to extract training samples and statistics.

    Args:
        new_tape: The tape to process containing web task steps
        split_name: Name of split ('train' or 'test')
        tokenizer: Tokenizer to use for generating training traces

    Returns:
        Tuple[List[TrainingText], Dict[str, int]]:
            - List of training samples (`TrainingText`) extracted from the tape.
            - Dictionary with tape-level statistics, including:
                - 'reward': Reward for the tape (-1, 0, or 1).
                - 'success': 1 if the answer is correct, 0 otherwise.
                - 'no_error': 1 if the LLM output is parsable, 0 otherwise.
                - 'prompt_tokens': Total prompt tokens used in the tape.
                - 'output_tokens': Total output tokens generated in the tape.
                - 'overflows': Number of times the output overflowed token limit.
    """
    # Get final reward from tape metadata
    success: bool = new_tape.metadata.result["success"]  # bool(reward > 0.5)
    final_reward: float = new_tape.metadata.result["final_reward"]  # 1.0 if success else 0.0
    raw_reward: float = new_tape.metadata.result["reward"]  # RAW_REWARD_GLOBAL from environment
    no_error = not tape_contains_an_error(new_tape)

    # get the number of LLM calls in the tape
    n_llm_calls = len(
        [
            step
            for step in new_tape.steps
            if "llm_call" in step.metadata.other and step.metadata.other["llm_call"] is not None
        ]
    )
    if n_llm_calls == 0:
        logger.warning(
            colored(
                f"tape {new_tape.metadata.id} has no LLM calls. n_steps:{len(new_tape.steps)}. final_reward:{final_reward}. raw_reward:{raw_reward}.",
                "red",
            )
        )

    # get the number of LLMOutputParsingFailureAction in the tape
    n_step_errors = len([step for step in new_tape.steps if isinstance(step, LLMOutputParsingFailureAction)])
    # get the number of PageObservation steps in the tape
    n_page_observations = len([step for step in new_tape.steps if isinstance(step, PageObservation)])

    # get the raw reward returned by the environment for each step
    raw_step_rewards = [
        (i, step.metadata.other.get("info", {}).get("task_info", {}).get("RAW_REWARD_GLOBAL", None))
        for i, step in enumerate(new_tape.steps)
    ]
    raw_step_rewards = [(i, reward) for i, reward in raw_step_rewards if reward is not None]  # filter out nones
    n_positive_rewards = len(
        [reward for _, reward in raw_step_rewards if reward > 0]
    )  # count number of positive rewards
    if n_positive_rewards > 1:
        logger.warning(
            colored(
                f"tape {new_tape.metadata.id} has {n_positive_rewards} positive rewards, but using final reward. Consider using step rewards instead.",
                "red",
            )
        )

    # design the reward. a few options to explore:
    # - use final_reward (0, 1, -1) --vs-- use the raw_reward (-1, 0-1) for all steps
    # - divide by n_llm_calls for all steps
    # - discount by number of errors for all steps
    # - use step rewards at all steps (likely 0 everywhere except the last step)
    reward_to_use = raw_reward * 0.99**n_step_errors if no_error and raw_reward >= 0 else -1.0
    # reward_to_use = None  # use step rewards instead

    # MAYBE update reward to penalize repeated and/or actions that do not change the state of the environment (MOVE_MOUSE, HOVER, SCROLL, ...) or filter them out?
    # MAYBE update reward to penalize intermediate steps that caused an error (LLMOutputParsingFailureAction) -> -1
    # depends on grouping: if we group by task_id, it's better to use the same reward for all steps.
    # if we group by task_id + step number, it's better to use individual rewards for each step (and penalize step errors).
    # TODO: reward_to_use = success or 0.1 or 0
    # config.finetune.rl.use_advantages (set to False to use reward instead)
    # config.finetune.rl.relu_log_p_weights (set to True to do SFT and use REINFORCE)

    ### MASSIMO setup: success_rate(1 | -1) * 0.95 ** n_steps ###
    # reward_to_use = 1 if success else -1
    # reward_to_use *= 0.95 ** len(new_tape.steps)  # discount by number of steps

    training_samples: list[TrainingText] = []
    tape_prompt_tokens = 0
    tape_output_tokens = 0
    overflows = []
    # For each LLM interaction in the tape, make a training example.
    for i, step in enumerate(new_tape.steps):
        if "llm_call" not in step.metadata.other or step.metadata.other["llm_call"] is None:
            continue

        llm_call = step.metadata.other["llm_call"]
        if isinstance(llm_call, dict):
            llm_call = LLMCall(**llm_call)

        tape_prompt_tokens += llm_call.prompt_length_tokens
        tape_output_tokens += llm_call.output_length_tokens

        if split_name == "train":
            # Create training sample
            trace = trainable_llm_make_training_text(llm_call.prompt, llm_call.output, tokenizer)

            input_ids = [lp.token_id for lp in llm_call.logprobs]
            labels = [lp.token_id for lp in llm_call.logprobs if lp.generated]
            labels = [MASKED_TOKEN_ID] * (len(input_ids) - len(labels)) + labels

            trace.input_ids = input_ids
            trace.labels = labels

            # check if the last produced token is the end of sequence token
            overflow = input_ids[-1] != tokenizer.eos_token_id
            overflows.append(overflow)

            if reward_to_use is not None:
                trace.reward = reward_to_use
            else:
                # Step specific reward are in the **next** PageObservation step, so the first step with index > i.
                step_reward = [r for j, r in raw_step_rewards if j > i][0]
                trace.reward = step_reward

            trace.logprobs = [lp.logprob for lp in llm_call.logprobs if lp.generated]
            trace.group_id = f"{new_tape.metadata.task_name}_{new_tape.metadata.seed}"  # Group by task_id + seed
            training_samples.append(trace)

    tape_stats = {
        "reward": reward_to_use if reward_to_use else final_reward,
        "success": success,
        "prompt_tokens": tape_prompt_tokens,
        "output_tokens": tape_output_tokens,
        "no_error": no_error,
        "overflows": sum(overflows),
        "n_llm_calls": n_llm_calls,
        "n_step_errors": n_step_errors,
        "n_page_observations": n_page_observations,
        "n_steps": len(new_tape.steps),
    }

    return training_samples, tape_stats


def _debug(tapes):
    steps_with_llm_call = [
        step
        for tape in tapes
        for step in tape.steps
        if "llm_call" in step.metadata.other and step.metadata.other["llm_call"] is not None
    ]
    if steps_with_llm_call:
        steps_with_llm_call[0].metadata.other["llm_call"].model_dump()


def generate_data(
    cfg: DictConfig,  # ExpArgs in AgentLab
    task: dict,
    url: str,
    split_name: str,
    iteration: int,
) -> Tuple[WebTape, List[TrainingText], Dict[str, int]]:
    # TODO: we could write a parralel version of this with 1 agent running multiple tasks in parallel
    # by using agent.run_batch & environment.react_batch in some sort of main_loop_batch() function.
    """
    Worker function for generating tapes and training samples for a single task.
    This function will create the agent, env, and llm inside it.
    It will then run the main_loop() to generate the tape.
    It will then extract the training samples from the tape.
    Finally the new tapes and training samples will be returned.
    """
    ### Set up paths
    exp_path = Path(cfg.output_dir)
    task_folder = exp_path / split_name / f"it{iteration}" / f"{task['task'].get_task_id()}_{task['seed']}"
    os.makedirs(task_folder, exist_ok=True)

    if os.path.exists(exp_path / "finetune/current"):
        model_path = str(exp_path / "finetune/current")
    else:
        model_path = cfg.model_path
    env_path = task_folder / "env"
    os.makedirs(env_path, exist_ok=True)
    log_file = exp_path / split_name / f"it{iteration}" / "logs" / f"{os.getpid()}.log"
    os.makedirs(log_file.parent, exist_ok=True)
    tapes_file = task_folder / f"tapes_{os.getpid()}.json"

    ### Set up logging
    log_handler = logging.FileHandler(str(log_file))
    log_handler.setLevel(logging.INFO)
    logging.basicConfig(
        format="%(asctime)s - PID_%(process)d - Thread_%(threadName)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[log_handler, logging.StreamHandler()],
        force=True,  # forget previous handlers
    )

    timers = {}
    ### STEP 0: create llm, env, agent ###
    t = time.perf_counter()
    llm = TrainableLLM(
        base_url=url,
        model_name=str(model_path),
        tokenizer_name=str(model_path),
        parameters=cfg.llm.parameters,
        use_cache=False,
        collect_logprobs=split_name == "train",
        observe_llm_calls=False,
    )
    timers["instantiated_llm"] = time.perf_counter() - t

    t = time.perf_counter()
    env = WebEnvironment(
        exp_path=str(env_path),
        headless=cfg.env.headless,
        observation_format=cfg.env.observation_format,
    )
    timers["instantiated_env"] = time.perf_counter() - t

    t = time.perf_counter()
    agent = WebAgent.create(llm)
    timers["instantiated_agent"] = time.perf_counter() - t

    ### STEP 1: run the agent on its task ###
    logger.info(
        f"[it{iteration}.{split_name}] ======== RUNNING THE AGENT ON {task['task'].get_task_id()} WITH SEED {task['seed']} ========"
    )
    t = time.perf_counter()
    new_tape = run_agent(agent, env, task)
    timers["generated_new_tape"] = time.perf_counter() - t
    logger.info(
        f"[it{iteration}.{split_name}] ======== GENERATED TAPE IN {timers['generated_new_tape']} s. NOW SAVING TAPE ========"
    )
    # some tapes end with PageObservation because the agent did not yield a stop step before the end of main_loop

    ### SAVE THE TAPE ###
    t = time.perf_counter()
    _debug([new_tape])  # TODO: debug why can't we save tapes anymore? https://github.com/pydantic/pydantic/issues/7713
    if os.path.exists(tapes_file):
        # load previous tapes and append the new tape
        with open(tapes_file, "r") as f:
            to_save = json.load(f)
        to_save.append(new_tape.model_dump())
    else:
        to_save = [new_tape.model_dump()]
    with open(tapes_file, "w") as f:
        json.dump(to_save, f, indent=4)
    timers["saved_new_tape"] = time.perf_counter() - t
    logger.info(
        f"[it{iteration}.{split_name}] ======== SAVED TAPE IN {timers['saved_new_tape']} s. NOW EXTRACT TRAINING TRACES ========"
    )

    ### STEP 2: get LLM & ENV timers ###
    llm_stats = {
        "mean_time_send_request": np.mean(llm._stats["time_send_request"]) if llm._stats["time_send_request"] else 0,
        "sum_time_send_request": sum(llm._stats["time_send_request"]) if llm._stats["time_send_request"] else 0,
        "mean_time_log_output": np.mean(llm._stats["time_log_output"]) if llm._stats["time_log_output"] else 0,
        "sum_time_log_output": sum(llm._stats["time_log_output"]) if llm._stats["time_log_output"] else 0,
        "mean_time_postprocess_llm_response": np.mean(llm._stats["time_postprocess_llm_response"])
        if llm._stats["time_postprocess_llm_response"]
        else 0,
        "sum_time_postprocess_llm_response": sum(llm._stats["time_postprocess_llm_response"])
        if llm._stats["time_postprocess_llm_response"]
        else 0,
    }
    # compute total llm time
    total_llm_time = (
        llm_stats["sum_time_send_request"]
        + llm_stats["sum_time_log_output"]
        + llm_stats["sum_time_postprocess_llm_response"]
    )
    timers["llm_total_time"] = total_llm_time

    # env.timers contains: start_task, finish_task, validate_task, and all the react times for each action
    if "react" in env.timers:
        timers["env_mean_react"] = np.mean(env.timers["react"])
        timers["env_var_react"] = np.var(env.timers["react"])
        timers["env_min_react"] = min(env.timers["react"])
        timers["env_max_react"] = max(env.timers["react"])
        timers["env_sum_react"] = sum(env.timers["react"])
        del env.timers["react"]  # remove react from timers to avoid confusion
    # add all other env timers
    timers.update({f"env_{key}": value for key, value in env.timers.items()})
    # compute total env time
    total_env_time = (
        env.timers.get("start_task", 0)
        + env.timers.get("finish_task", 0)
        + env.timers.get("validate_task", 0)
        + timers["env_sum_react"]
    )
    timers["env_total_time"] = total_env_time
    # compute action execution time
    action_execution_times = [
        step.metadata.other.get("info", {}).get("action_exec_stop", np.inf)
        - step.metadata.other.get("info", {}).get("action_exec_start", 0)
        for step in new_tape.steps
        if isinstance(step, PageObservation)
    ]
    timers["env_mean_action_execution"] = np.mean(action_execution_times) if action_execution_times else 0.0
    timers["env_var_action_execution"] = np.var(action_execution_times) if action_execution_times else 0.0
    timers["env_min_action_execution"] = min(action_execution_times) if action_execution_times else 0.0
    timers["env_max_action_execution"] = max(action_execution_times) if action_execution_times else 0.0
    timers["env_sum_action_execution"] = sum(action_execution_times) if action_execution_times else 0.0

    ### STEP 3: extract training samples from the newly generated tape ###
    t = time.perf_counter()
    llm.load_tokenizer()  # make sure llm.tokenizer is loaded
    # 1 tape -> multiple training samples because of multiple LLM calls
    tape_training_samples, tape_stats = extract_tape_training_samples_and_stats(new_tape, split_name, llm.tokenizer)
    timers["extracted_training_samples"] = time.perf_counter() - t
    logger.info(
        f"[it{iteration}.{split_name}] ======== EXTRACTED {len(tape_training_samples)} TRACES IN {timers['extracted_training_samples']} s. ========"
    )
    # tape_stats contains: reward, success, no_error, prompt_tokens, output_tokens, overflows, n_llm_calls, n_step_errors, n_page_observations, n_steps
    timers["llm_total_time_per_llm_call"] = (
        total_llm_time / tape_stats["n_llm_calls"] if tape_stats["n_llm_calls"] > 0 else 0
    )
    timers["env_total_time_per_observation"] = (
        total_env_time / tape_stats["n_page_observations"] if tape_stats["n_page_observations"] > 0 else 0
    )
    # compute average prompt and output tokens per LLM call
    if tape_stats["n_llm_calls"] > 0:
        tape_stats["prompt_tokens_per_llm_call"] = tape_stats["prompt_tokens"] / tape_stats["n_llm_calls"]
        tape_stats["output_tokens_per_llm_call"] = tape_stats["output_tokens"] / tape_stats["n_llm_calls"]

    new_tape.metadata.other["timers"] = timers
    logger.info(
        f"[it{iteration}.{split_name}] ======== TAPE {new_tape.metadata.id} TOOK {json.dumps(timers, indent=4)} ========"
    )
    return new_tape, tape_training_samples, tape_stats, llm_stats


def batch_generate_data(
    cfg: DictConfig,
    tasks: list[dict],
    urls: list[str],
    split_name: str,
    iteration: int,
) -> Tuple[List[WebTape], List[TrainingText], Dict[str, float]]:
    """
    Generate complete tapes and training samples from a list of tasks.

    Args:
        cfg: Config
        tasks: List of task dicts containing 'task' (MiniWoB task) and 'seed'
        urls: List of urls to use for the llm
        split_name: Name of split ('train' or other)
        iteration: Current iteration number
    Returns:
        Tuple containing:
        - List of completed WebTapes
        - List of training samples with rewards and logprobs
        - Dictionary of performance statistics and execution times
    """
    assert len(urls) == len(tasks), f"Number of urls ({len(urls)}), and tasks ({len(tasks)}) must match"
    start_make_data = time.time()

    ### STEP 1: run the agents on their tasks in parallel ###
    logger.info(f"[it{iteration}] Run the agent on {split_name} with {cfg.n_processes_for_data_generation} processes")
    with tqdm_joblib(tqdm(desc="generating tapes...", total=len(tasks))) as _:
        results = Parallel(n_jobs=cfg.n_processes_for_data_generation, prefer="processes")(
            [delayed(generate_data)(cfg, task, url, split_name, iteration) for task, url in zip(tasks, urls)]
        )  # will return a list when all tasks are finished in the same order as the tasks
    logger.info(f"[it{iteration}] Making tapes took {time.time() - start_make_data}")
    assert len(results) == len(tasks), f"Number of results ({len(results)}) and tasks ({len(tasks)}) must match"

    ### STEP 2: aggregate training samples and stats ###
    final_tapes: list[WebTape] = []  # list of final tapes
    training_samples: List[TrainingText] = []  # list of training samples
    # tape_stats aggregators
    reward_stats = defaultdict(list)  # map from group id to list of rewards
    success_stats = defaultdict(list)  # map from group id to list of successes
    no_errors_stats = defaultdict(list)  # map from group id to list of no_errors
    prompt_tokens_stats = defaultdict(list)  # map from group id to list of prompt tokens length
    output_tokens_stats = defaultdict(list)  # map from group id to list of output tokens length

    all_llm_stats = defaultdict(list)  # map from stat name to list of llm stats
    all_tape_timers = defaultdict(list)  # map from stat name to list of timers stats
    all_tape_stats = defaultdict(list)  # map from stat name to list of tape stats
    for new_tape, samples, tape_stats, llm_stats in results:
        # timers contains instantiated_llm, instantiated_env, instantiated_agent, generated_new_tape, saved_new_tape, extracted_training_samples, llm_total_time, llm_total_time_per_llm_call, env_... timers
        # tape_stats contains: reward, success, no_error, prompt_tokens, output_tokens, overflows, n_llm_calls, n_step_errors, n_page_observations, n_steps
        # llm_stats contains: {mean|sum}_time_send_request, {mean|sum}_time_log_output, {mean|sum}_time_postprocess_llm_response
        final_tapes.append(new_tape)
        training_samples.extend(samples)
        group_id = f"{new_tape.metadata.task_name}_{new_tape.metadata.seed}"
        if samples:
            assert all(
                [group_id == s.group_id for s in samples]
            ), f"Group id mismatch in samples: {group_id}, {[s.group_id for s in samples]}"
        # special treatment for reward, success, no_error, prompt_tokens, output_tokens: we will compute the mean of the min/max of each group
        reward_stats[group_id].append(tape_stats["reward"])
        success_stats[group_id].append(tape_stats["success"])
        no_errors_stats[group_id].append(tape_stats["no_error"])
        prompt_tokens_stats[group_id].append(tape_stats["prompt_tokens"])
        output_tokens_stats[group_id].append(tape_stats["output_tokens"])
        # the rest of the stats we don't need to group by group_id as we don't care about their mean of min/max
        for key, value in tape_stats.items():
            if key not in ["reward", "success", "no_error", "prompt_tokens", "output_tokens"]:
                all_tape_stats[key].append(value)
        for key, value in llm_stats.items():
            all_llm_stats[key].append(value)
        for key, value in new_tape.metadata.other.get("timers", {}).items():
            all_tape_timers[key].append(value)

    ### STEP 3: compute stats ###
    end_make_data = time.time()
    sum_prompt_tokens = sum([sum(pt) for pt in prompt_tokens_stats.values()])
    sum_output_tokens = sum([sum(ot) for ot in output_tokens_stats.values()])
    stats = {
        **{f"tape_stats/{split_name}_{k}_reward": v for k, v in calculate_stats(reward_stats).items()},
        **{f"tape_stats/{split_name}_{k}_success": v for k, v in calculate_stats(success_stats).items()},
        **{f"tape_stats/{split_name}_{k}_no_errors": v for k, v in calculate_stats(no_errors_stats).items()},
        **{f"tape_stats/{split_name}_{k}_prompt_tokens": v for k, v in calculate_stats(prompt_tokens_stats).items()},
        **{f"tape_stats/{split_name}_{k}_output_tokens": v for k, v in calculate_stats(output_tokens_stats).items()},
        **{
            # f"execution_time/{split_name}_dumping_tapes": end_dump - start_dump,
            f"execution_time/{split_name}_make_data": end_make_data - start_make_data,
            f"execution_time/{split_name}_tapes_made_per_second": len(final_tapes) / (end_make_data - start_make_data),
            f"execution_time/{split_name}_prompt_tokens_per_sec": sum_prompt_tokens / (end_make_data - start_make_data),
            f"execution_time/{split_name}_output_tokens_per_sec": sum_output_tokens / (end_make_data - start_make_data),
            f"execution_time/{split_name}_total_tokens_per_sec": (sum_prompt_tokens + sum_output_tokens)
            / (end_make_data - start_make_data),
            f"tape_stats/{split_name}_sum_prompt_tokens": sum_prompt_tokens,
            f"tape_stats/{split_name}_sum_output_tokens": sum_output_tokens,
            f"tape_stats/{split_name}_all_success": np.mean([all(s) for s in success_stats.values()]),
            f"tape_stats/{split_name}_any_success": np.mean([any(s) for s in success_stats.values()]),
            f"tape_stats/{split_name}_no_success": np.mean([not any(s) for s in success_stats.values()]),
        },
        **{f"tape_stats/{split_name}_mean_{k}": np.mean(v) for k, v in all_tape_stats.items()},
        **{f"llm/{split_name}_mean_{k}": np.mean(v) for k, v in all_llm_stats.items()},
        **{f"tape_timers/{split_name}_mean_{k}": np.mean(v) for k, v in all_tape_timers.items()},
    }

    return final_tapes, training_samples, stats


def batch_annotate_traces_with_ref_logprobs(llm: TrainableLLM, traces: List[TrainingText]):
    """
    Annotates training traces with reference model log probabilities.

    Args:
        llm (TrainableLLM): The reference language model to get log probabilities from
        traces (List[TrainingText]): List of training traces to annotate

    Returns:
        None: The traces are modified in-place by adding ref_logprobs
    """
    prompt_token_ids = []
    completion_token_ids = []
    for trace in traces:
        assert trace.input_ids, f"Input IDs are empty for trace: {trace}"
        assert trace.logprobs, f"Logprobs are empty for trace: {trace}"
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


@hydra.main(config_path="../../../conf/", config_name="rl_webagent", version_base="1.3.2")
def main(cfg: DictConfig):
    os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(cfg.output_dir, "tapedata.sqlite")
    os.environ["MINIWOB_URL"] = cfg.environment_variables.miniwob_url
    # os.environ["SNOW_INSTANCE_URL"] = cfg.environment_variables.snow_instance_url
    # os.environ["SNOW_INSTANCE_UNAME"] = cfg.environment_variables.snow_instance_uname
    # os.environ["SNOW_INSTANCE_PWD"] = cfg.environment_variables.snow_instance_pwd

    ### Step 0: init logging, wandb, rl state, paths ###
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

    conf_dir = exp_path / "conf"
    os.makedirs(conf_dir, exist_ok=True)
    finetune_path = exp_path / "finetune"

    ### Step 1: load datasets ###
    # train_samples, test_samples = load_webtasks(train_split=cfg.train_split, seeds=cfg.seeds)
    train_samples, test_samples = load_webtasks_debug()  # TODO: load all tasks when ready

    ############## PROFILE ENVIRONMENT CREATION AND TASK INIT ##############
    # start_tape_creation_times = []
    # for _ in range(10):
    #     for task in train_samples:
    #         env_path = exp_path / "train" / "it1" / f"{task['task'].get_task_id()}_{task['seed']}" / "env"
    #         _zero = time.perf_counter()
    #         env = WebEnvironment(
    #             exp_path=str(env_path),
    #             headless=cfg.env.headless,
    #             observation_format=cfg.env.observation_format,
    #         )
    #         zero = time.perf_counter() - _zero
    #         logger.info(f"WebEnvironment class instance created in {zero:.2f} seconds")
    #         # init the tape
    #         _one = time.perf_counter()
    #         _ = env.start_task(task)
    #         one = time.perf_counter() - _one
    #         logger.info(f"Start_task done in {one:.2f} seconds.")
    #         _two = time.perf_counter()
    #         env.finish_task()
    #         two = time.perf_counter() - _two
    #         logger.info(f"Finish_task done in {two:.2f} seconds.")
    #         start_tape_creation_times.append((zero, one, two))
    # logger.info("============= PROFILE SUMMARY ============")
    # logger.info(start_tape_creation_times)
    # logger.info(
    #     f"Average time to create an env, start, and terminate a task: {np.sum(start_tape_creation_times) / len(start_tape_creation_times):.2f} over {len(start_tape_creation_times)} tasks"
    # )
    # return
    ############## END PROFILE ##############

    ### repeat until we have reached the max number of iterations ###
    ### each iteration is a forward pass (agent making predictions on tapes), a reference pass (reference model populating ref_logprobs), and a finetuning run on the generated training samples ###
    while state["iteration"] < cfg.max_iterations:
        ### Massimo's setup: sample new tasks for each iterations ###
        # train_samples, test_samples = load_webtasks_debug()

        logger.info(f"Starting iteration {state['iteration']}")
        start_iteration = time.time()
        if os.path.exists(finetune_path / "current"):
            assistant_model_path = str(finetune_path / "current")
        else:
            assistant_model_path = cfg.model_path

        ### Step 2: collect tapes (and training samples) using the assistant model ###
        ### We might also evaluate the assistant model on the test set ###
        try:
            all_results = {}  # map from split name to dict with new_tapes, training_samples, stats
            with VLLMServiceManager(
                exp_path=exp_path,
                service_name="actor",
                model_name_or_path=assistant_model_path,
                port=8080,
                verbose=True,
                cuda_device=",".join([str(i) for i in range(torch.cuda.device_count())]),
                **(dict(cfg.vllm_config.vllm_kwargs) | dict(cfg.vllm_config.actor_vllm_kwargs)),
            ) as vllm_service_manager:
                ### Step 2.1: create train & test splits ###
                if cfg.max_agent_forks // cfg.attempts > len(train_samples):
                    # over sample tasks if needed
                    sub_samples = []
                    while len(sub_samples) < cfg.max_agent_forks // cfg.attempts:
                        # add tasks 1 by 1 in a random order
                        for task in random.sample(train_samples, len(train_samples)):
                            sub_samples.append(copy.deepcopy(task))
                            if len(sub_samples) >= cfg.max_agent_forks // cfg.attempts:
                                break
                else:
                    sub_samples = random.sample(train_samples, cfg.max_agent_forks // cfg.attempts)
                logger.info(
                    f"[it{state['iteration']}] Sampling {len(sub_samples)} train tasks out of {len(train_samples)}. Each task will be repeated {cfg.attempts} times"
                )
                # Repeat each task cfg.attempts times
                train_tasks = [copy.deepcopy(task) for task in sub_samples for _ in range(cfg.attempts)]
                n_train_tasks = len(train_tasks)
                logger.info(
                    f"[it{state['iteration']}] Total number of train tasks: {n_train_tasks} (max target was {cfg.max_agent_forks})"
                )

                # Get number of VLLM services available
                vllm_services = vllm_service_manager.get_base_urls()
                n_services = len(vllm_services)

                # alternate between services (URL) for each task
                train_urls = [vllm_services[task_idx % n_services] for task_idx in range(n_train_tasks)]
                splits = [("train", train_tasks, train_urls)]

                # Create test split if needed
                if state["iteration"] % cfg.test_every_n_iterations == 0 and cfg.test_every_n_iterations > 0:
                    logger.info(f"[it{state['iteration']}] Creating test split with {len(test_samples)} tasks")
                    test_urls = [vllm_services[task_idx % n_services] for task_idx in range(len(test_samples))]
                    splits.append(("test", test_samples, test_urls))

                ### Step 2.2: generate continuations for each split ###
                for split_name, tasks, urls in splits:
                    assert len(tasks) == len(urls)
                    new_tapes, split_training_samples, stats = batch_generate_data(
                        cfg, tasks, urls, split_name, state["iteration"]
                    )

                    all_results[split_name] = {
                        "new_tapes": new_tapes,
                        "training_samples": split_training_samples,
                        "stats": stats,
                    }

                    # Log results
                    logger.info(f"[it{state['iteration']}] {split_name} stats:")
                    for stat_name, stat_value in stats.items():
                        logger.info(f"  {stat_name}: {stat_value}")
                # keep track of the starting time of the assistant model
                assistant_model_starting_time = vllm_service_manager.get_stats()["starting_time"]
            ### end with vllm_service_manager("actor")
        except Exception as e:
            logger.exception(e, stack_info=True)
            logger.error(colored(f"Failed to solve task: {e}", "red"))
            raise e

        ### Step 3: log all stats from forward pass###
        training_samples: list[TrainingText] = all_results["train"]["training_samples"]
        logger.info(f"[it{state['iteration']}] Collected {len(training_samples)} training samples")
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

        ### Step 4: populate reference logprobs with the reference model ###
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
                ### Step 4.1: create reference models ###
                ref_llms = [
                    TrainableLLM(
                        base_url=url,
                        model_name=cfg.model_path,
                        tokenizer_name=cfg.model_path,
                        parameters=dict(temperature=0.7),
                    )
                    for url in vllm_service_manager.get_base_urls()
                ]

                ### Step 4.2: populate reference logprobs ###
                start_basemodel_logprobs = time.time()
                with ThreadPoolExecutor(
                    max_workers=cfg.get_logprobs_workers_per_gpu * torch.cuda.device_count()
                ) as executor:
                    chunk_size = 64
                    futures = []
                    for chunk_id, chunk_offset in enumerate(range(0, len(training_samples), chunk_size)):
                        # one reference model per chunk
                        ref_llm = ref_llms[chunk_id % len(ref_llms)]
                        # get a chunk of training samples
                        chunk = training_samples[chunk_offset : chunk_offset + chunk_size]
                        # submit the chunk to the executor
                        futures.append(executor.submit(batch_annotate_traces_with_ref_logprobs, ref_llm, chunk))
                    # Reference logprobs are added in-place
                    # wait for all futures to complete
                    futures = tqdm(as_completed(futures), total=len(futures), desc="Adding logprobs")
                    # get the results
                    _ = [future.result() for future in futures]

                # keep track of the starting time of the reference model
                refmodel_starting_time = vllm_service_manager.get_stats()["starting_time"]
                time_populating_ref_logprobs = time.time() - start_basemodel_logprobs
            ### end with vllm_service_manager("reference")
        except Exception as e:
            logger.error(colored(f"Failed to get ref log probs: {e}", "red"))
            raise e

        ### Step 5: log all stats from reference model pass ###
        logprob_stats = {
            "execution_time/populating_ref_logprobs": time_populating_ref_logprobs,
            "execution_time/starting_assistantmodel_vllm": assistant_model_starting_time,
            "execution_time/starting_refmodel_vllm": refmodel_starting_time,
        }
        logger.info(f"[it{state['iteration']}] Logprob population stats:")
        for stat_name, stat_value in logprob_stats.items():
            logger.info(f"  {stat_name}: {stat_value}")
        wandb.log(logprob_stats, step=state["iteration"])

        ### Step 6: save the training samples ###
        rollout_dir = exp_path / "rollouts" / str(state["iteration"])
        os.makedirs(rollout_dir, exist_ok=True)
        with open(rollout_dir / "data.jsonl", "w") as f:
            for trace in training_samples:
                f.write(trace.model_dump_json() + "\n")
                f.flush()

        ### Step 7: launch the finetuning iteration ###

        # Create a config for this finetuning iteration
        finetune_cfg = cfg.copy()

        # we increment the number of steps to interrupt the training because each finetuning run will continue from the last one
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
        # launch the finetuning in a subprocess
        launch_training(
            str(conf_dir),
            str(state["iteration"]),
            cfg.accelerate_cfg_path,
            use_deepspeed=cfg.use_deepspeed,  # defaults to False
        )
        # log finetuning stats
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
    ### end of the while loop ###
    logger.info(f'Finished training after {state["iteration"]} iterations')


if __name__ == "__main__":
    main()
