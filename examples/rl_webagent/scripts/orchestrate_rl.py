import copy
import json
import logging
import multiprocessing
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
from joblib import Parallel, delayed
import numpy as np
import torch
from browsergym.miniwob import ALL_MINIWOB_TASKS
from omegaconf import DictConfig, OmegaConf
from termcolor import colored
from tqdm import tqdm

import wandb
from tapeagents.core import LLMCall, LLMOutputParsingFailureAction, TrainingText
from tapeagents.finetune.data import MASKED_TOKEN_ID
from tapeagents.finetune.logging_ import flatten_dict_config, init_wandb
from tapeagents.llms import TrainableLLM, trainable_llm_make_training_text
from tapeagents.observe import retrieve_all_llm_calls
from tapeagents.orchestrator import main_loop
from tapeagents.tools.simple_browser import PageObservation

from ..agent import WebAgent, WebTape
from ..environment import WebEnvironment
from ..steps import WebAction
from ..utils import (
    VLLMServiceManager,
    calculate_stats,
    clean_up,
    launch_training,
    load_state,
    save_state,
    setup_logging,
)

logger = logging.getLogger(__name__)


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
        {"dataset": "miniwob", "task": task, "seed": seed}
        for task in train_tasks
        for seed in seeds
    ]

    test_samples = [
        {"dataset": "miniwob", "task": task, "seed": seed}
        for task in test_tasks
        for seed in seeds
    ]

    logger.info(f"Loaded {len(train_samples)} training samples")
    logger.info(f"Loaded {len(test_samples)} test samples")
    return train_samples, test_samples


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
    try:
        # Initialize task in environment - this creates the initial tape with web observation and task
        tape, metadata = env.start_task(task["task"], task["seed"])

        # Run agent-environment loop
        # tape = main_loop(agent, tape, env, max_loops=20).get_final_tape()
        last_action = None
        repeated_action_cnt = 0
        for event in main_loop(agent, tape, env, max_loops=20):
            if event.agent_event and event.agent_event.step:
                step = event.agent_event.step
                # Check for repeated actions
                if isinstance(step, WebAction):
                    step_view = step.llm_view()
                    if step_view == last_action:
                        repeated_action_cnt += 1
                        if repeated_action_cnt > 4:
                            break
                    else:
                        repeated_action_cnt = 0
                    last_action = step_view
                tape = tape.append(step)
            if event.observation:
                # PageObservation have the reward in metadata.other["reward"]
                tape = tape.append(event.observation)

        # Get final reward (1.0 if success, where success is defined as result["reward"] > 0)
        # result is a dict of reward, stop, message, info
        success, result = env.validate_task(tape)
        final_reward = 1.0 if success else 0.0
        tape.metadata.result = {"success": success, **result, "final_reward": final_reward}

    except Exception as e:
        logger.error(f"Failed to run task: {e}")
        tape.metadata.result = {"success": False, "final_reward": 0.0, "error": str(e)}

    finally:
        # Always close the environment
        env.finish_task()

    return tape


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
    final_reward = new_tape.metadata.result["final_reward"]
    success = new_tape.metadata.result["success"]
    no_error = not isinstance(new_tape.steps[-1], LLMOutputParsingFailureAction) and new_tape.metadata.result.get("error") is None

    training_samples: list[TrainingText] = []
    tape_prompt_tokens = 0
    tape_output_tokens = 0
    overflows = []
    # For each LLM interaction in the tape:
    for step in new_tape.steps:
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

            # Step specific reward are in the next PageObservation step
            # TODO: change code to fetch these.
            step_reward = step.metadata.other.get("reward", final_reward)
            trace.reward = step_reward

            trace.logprobs = [lp.logprob for lp in llm_call.logprobs if lp.generated]
            trace.group_id = new_tape.metadata.parent_id
            training_samples.append(trace)

    tape_stats = {
        "reward": final_reward,
        "success": success,
        "prompt_tokens": tape_prompt_tokens,
        "output_tokens": tape_output_tokens,
        "no_error": no_error,
        "overflows": sum(overflows),
    }

    return training_samples, tape_stats


def generate_data(
    cfg: DictConfig,
    task: dict,
    url: str,
    split_name: str,
    iteration: int,
) -> Tuple[WebTape, List[TrainingText], Dict[str, int]]:
    """
    Worker function for generating tapes and training samples for a single task.
    This function will create the agent, env, and llm inside it.
    It will then run the main_loop() to generate the tape.
    It will then extract the training samples from the tape.
    Finally the new tapes and training samples will be returned.
    """
    exp_path = Path(cfg.output_dir)
    if os.path.exists(exp_path / "finetune/current"):
        model_path = str(exp_path / "finetune/current")
    else:
        model_path = cfg.model_path

    tapes_dir = exp_path / "tapes" / split_name / str(iteration)
    os.makedirs(tapes_dir, exist_ok=True)

    timers = {}
    ### STEP 0: create llm, env, agent ###
    t = time.perf_counter()
    llm = TrainableLLM(
        base_url=url,
        model_name=str(model_path),
        tokenizer_name=str(model_path),
        parameters=cfg.llm.parameters,
        use_cache=False,
        collect_logprobs=True,
        observe_llm_calls=False,
    )
    timers["instantiated_llm"] = time.perf_counter() - t

    t = time.perf_counter()
    env_path = f"{exp_path}/envs/{split_name}/it{iteration}/{task['task'].get_task_id()}"
    env = WebEnvironment(
        exp_path=env_path,
        headless=cfg.env.headless,
        ax_tree=cfg.env.ax_tree,
        html=cfg.env.html,
        markdown_html=cfg.env.markdown_html,
    )
    timers["instantiated_env"] = time.perf_counter() - t

    t = time.perf_counter()
    agent = WebAgent.create(llm)
    timers["instantiated_agent"] = time.perf_counter() - t

    ### STEP 1: run the agent on its task ###
    t = time.perf_counter()
    new_tape = run_agent(agent, env, task)
    timers["generated_new_tape"] = time.perf_counter() - t

    ### STEP 2: extract training samples from the newly generated tape ###
    t = time.perf_counter()
    llm.load_tokenizer()  # make sure llm.tokenizer is loaded
    # 1 tape -> multiple training samples because of multiple LLM calls
    tape_training_samples, tape_stats = extract_tape_training_samples_and_stats(new_tape, split_name, llm.tokenizer)
    timers["extract_training_samples"] = time.perf_counter() - t

    new_tape.metadata.other["timers"] |= timers
    return new_tape, tape_training_samples, tape_stats


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
    assert len(urls) == len(tasks), (
        f"Number of urls ({len(urls)}), and tasks ({len(tasks)}) must match"
    )
    start_make_data = time.time()

    reward_stats = defaultdict(list)  # map from parent tape id to list of rewards
    success_stats = defaultdict(list)  # map from parent tape id to list of successes
    no_errors_stats = defaultdict(list)  # map from parent tape id to list of no_errors
    prompt_tokens_stats = defaultdict(list)  # map from parent tape id to list of prompt tokens length
    output_tokens_stats = defaultdict(list)  # map from parent tape id to list of output tokens length
    overflow_stats = defaultdict(list)  # map from parent tape id to list of overflows

    training_samples: List[TrainingText] = []

    ### STEP 1: run the agents on their tasks in parallel ###
    logger.info(f"[it{iteration}] Run the agent on {split_name} with {cfg.n_processes_for_data_generation} processes.")
    final_tapes, tape_training_samples, tape_stats = Parallel(n_jobs=cfg.n_processes_for_data_generation, prefer="processes")(
        [delayed(generate_data)(cfg, task, url, split_name, iteration) for task, url in zip(tasks, urls)]
    )  # will return lists only when all tasks are finished in the same order as the tasks
    logger.info(f"[it{iteration}] Making tapes took {time.time() - start_make_data}")
    # final_tapes is a list of tapes
    # training_samples is a list of list of training samples
    # tape_stats is a list of dict of stats
    assert len(final_tapes) == len(tape_training_samples) == len(tape_stats) == len(tasks), (
        f"Number of tapes ({len(final_tapes)}), "
        f"training samples ({len(tape_training_samples)}), "
        f"stats ({len(tape_stats)}) and tasks ({len(tasks)}) must match"
    )

    ### STEP 2: aggregate training samples and stats ###
    for new_tape, samples, stats in zip(final_tapes, tape_training_samples, tape_stats):
        training_samples.extend(samples)
        reward_stats[new_tape.metadata.parent_id].append(stats["reward"])
        success_stats[new_tape.metadata.parent_id].append(stats["success"])
        no_errors_stats[new_tape.metadata.parent_id].append(stats["no_error"])
        prompt_tokens_stats[new_tape.metadata.parent_id].append(stats["prompt_tokens"])
        output_tokens_stats[new_tape.metadata.parent_id].append(stats["output_tokens"])
        overflow_stats[new_tape.metadata.parent_id].append(stats["overflows"])

    ### STEP 3: save the tapes ###
    exp_path = Path(cfg.output_dir)
    tapes_dir = exp_path / "tapes" / split_name / str(iteration)
    start_dump = time.time()
    with open(tapes_dir / "tapes.json", "w") as f:
        json.dump([tape.model_dump() for tape in final_tapes], f, indent=4)
    end_dump = time.time()

    ### STEP 4: compute stats ###
    end_make_data = time.time()
    stats = {
        **{f"{split_name}_{k}_reward": v for k, v in calculate_stats(reward_stats).items()},
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
            f"{split_name}_overflows": np.mean([np.mean(ov) for ov in overflow_stats.values()]),
        },
    }
    return final_tapes, training_samples, stats


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
    train_samples, test_samples = load_webtasks(train_split=cfg.train_split, seeds=cfg.seeds)

    ### repeat until we have reached the max number of iterations ###
    ### each iteration is a forward pass (agent making predictions on tapes), a reference pass (reference model populating ref_logprobs), and a finetuning run on the generated training samples ###
    while state["iteration"] < cfg.max_iterations:
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
                sub_samples = random.sample(train_samples, min(len(train_samples), cfg.max_agent_forks // cfg.attempts))
                logger.info(f"[it{state['iteration']}] Subsampling {len(sub_samples)} train tasks out of {len(train_samples)}. Each task will be repeated {cfg.attempts} times")
                # Repeat each task cfg.attempts times
                train_tasks = [copy.deepcopy(task) for task in sub_samples for _ in range(cfg.attempts)]
                n_train_tasks = len(train_tasks)
                logger.info(f"[it{state['iteration']}] Total number of train tasks: {n_train_tasks} (max target was {cfg.max_agent_forks})")

                # Get number of VLLM services available
                vllm_services = vllm_service_manager.get_base_urls()
                n_services = len(vllm_services)

                train_urls = [vllm_services[task_idx % n_services] for task_idx in range(n_train_tasks)]
                splits = [("train", train_tasks, train_urls)]

                # Create test split if needed
                if state["iteration"] % cfg.test_every_n_iterations == 0 and cfg.test_every_n_iterations > 0:
                    # Calculate how many agents to create for this service
                    logger.info(f"[it{state['iteration']}] Creating test split with {len(test_samples)} tasks")
                    test_urls = [vllm_services[task_idx % n_services] for task_idx in range(len(test_samples))]
                    splits.append(("test", test_samples, test_urls))

                ### Step 2.2: generate continuations for each split ###
                for split_name, tasks, urls in splits:
                    assert len(tasks) == len(urls)
                    new_tapes, split_training_samples, stats = batch_generate_data(cfg, tasks, urls, split_name, state["iteration"])

                    # SKIP llm stats for now because each subprocess has its own llm and we will not be able to aggregate them.

                    # old way of doing it was to return the stats from the agent's llm and then do this:
                    # llm_stats = agent_replicas_with_stats[0].llm.get_stats()
                    # llm_stats = {f"llm/{split_name}_{k}": v for k, v in llm_stats.items()}
                    # but returning the agent or the llm from a subprocess can be tricky as it is a complex datastructure that must be transferred to the main process
                    # also, since each process has its own llm, it will have its own stats and we will not be able to aggregate them

                    # One way to try to get some stats is to load the sqlite file and then do this (copied from gaia_agent/scripts/tape_browser.py):
                    # sqlite_fpath = os.path.join(cfg.output_dir, "tapedata.sqlite")
                    # llm_calls: dict[str, LLMCall] = {llm_call.prompt.id: llm_call for llm_call in retrieve_all_llm_calls(sqlite_fpath)}
                    # for llm_call in llm_calls.values():
                    #     prompt_tokens_num += llm_call.prompt_length_tokens
                    #     output_tokens_num += llm_call.output_length_tokens
                    #     total_cost += llm_call.cost

                    # stats.update(llm_stats)
                    # this would add the following to the stats:
                    # llm/{split_name}_time_send_request
                    # llm/{split_name}_time_log_output
                    # llm/{split_name}_total_prompt_tokens
                    # llm/{split_name}_total_output_tokens
                    # llm/{split_name}_time_postprocess_llm_response

                    make_data_took = stats[f"execution_time/{split_name}_make_data"]
                    throughput_stats = {
                        f"{split_name}_prompt_tokens_per_sec": stats[f"{split_name}_prompt_tokens"] / make_data_took,
                        f"{split_name}_output_tokens_per_sec": stats[f"{split_name}_output_tokens"] / make_data_took,
                        f"{split_name}_total_tokens_per_sec": (
                            stats[f"{split_name}_prompt_tokens"] + stats[f"{split_name}_output_tokens"]
                        )
                        / make_data_took,
                    }
                    stats.update(throughput_stats)

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
