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
from typing import Dict, List, Tuple

import hydra
import numpy as np
import torch
from browsergym.miniwob import ALL_MINIWOB_TASKS
from omegaconf import DictConfig, OmegaConf
from termcolor import colored
from tqdm import tqdm

import wandb
from tapeagents.core import LLMCall, TrainingText
from tapeagents.finetune.data import MASKED_TOKEN_ID
from tapeagents.finetune.logging_ import flatten_dict_config, init_wandb
from tapeagents.llms import TrainableLLM
from tapeagents.orchestrator import main_loop

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


def batch_run_agent_replica(agent: WebAgent, env: WebEnvironment, task: dict) -> tuple[WebAgent, WebTape]:
    """
    Run agent on a single web task with its environment.

    Args:
        agent: The web agent
        env: The web environment for this task
        task: Task dict containing 'task' (MiniWoB task) and 'seed'

    Returns:
        tuple[WebAgent, WebTape]: Updated agent and completed tape
    """
    try:
        # Initialize task in environment - this creates the initial tape with web observation and task
        tape, metadata = env.start_task(task["task"], task["seed"])

        # Run agent-environment loop
        tape = main_loop(agent, tape, env, max_loops=20).get_final_tape()
        # last_action = None
        # repeated_action_cnt = 0
        # for event in main_loop(agent, tape, env, max_loops=20):
        #     if event.agent_event and event.agent_event.step:
        #         step = event.agent_event.step
        #         # Get immediate reward for action
        #         immediate_reward = env.get_step_reward(step) if hasattr(env, "get_step_reward") else None
        #         if hasattr(step, "metadata"):
        #             step.metadata.other["reward"] = immediate_reward
        #         # Check for repeated actions
        #         if isinstance(step, WebAction):
        #             step_view = step.llm_view()
        #             if step_view == last_action:
        #                 repeated_action_cnt += 1
        #                 if repeated_action_cnt > 4:
        #                     break
        #             else:
        #                 repeated_action_cnt = 0
        #             last_action = step_view
        #         tape = tape.append(step)
        #     if event.observation:
        #         tape = tape.append(event.observation)

        # Get final reward
        success, result = env.validate_task(tape)
        final_reward = 1.0 if success else 0.0
        tape.metadata.result = {"success": success, **result, "final_reward": final_reward}

    except Exception as e:
        logger.error(f"Failed to run task: {e}")
        tape.metadata.result = {"success": False, "final_reward": 0.0, "error": str(e)}

    finally:
        # Always close the environment
        env.finish_task()

    return agent, tape


def extract_tape_training_samples_and_stats(
    new_tape: WebTape, agent: WebAgent, split_name: str
) -> Tuple[List[TrainingText], Dict[str, int]]:
    """
    Extract training samples with per-action rewards and final reward.
    """
    # Get final reward from tape metadata
    final_reward = new_tape.metadata.result["final_reward"]
    success = new_tape.metadata.result["success"]

    training_samples: list[TrainingText] = []
    tape_prompt_tokens = 0
    tape_output_tokens = 0

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
            trace = agent.llm.make_training_text(llm_call.prompt, llm_call.output)

            input_ids = [lp.token_id for lp in llm_call.logprobs]
            labels = [lp.token_id for lp in llm_call.logprobs if lp.generated]
            labels = [MASKED_TOKEN_ID] * (len(input_ids) - len(labels)) + labels

            trace.input_ids = input_ids
            trace.labels = labels

            # Get step-specific reward if available, otherwise use final reward
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
    }

    return training_samples, tape_stats


def generate_training_data(
    agent_replicas: list[WebAgent],
    envs: list[WebEnvironment],
    tasks: list[dict],
    tapes_dir: Path,
    split_name: str,
    n_processes: int,
) -> Tuple[list[WebAgent], List[WebTape], List[TrainingText], Dict[str, float]]:
    """
    Generate complete tapes and training samples from a list of tasks.

    Args:
        agent_replicas: List of WebAgents that interact with their own web environment
        envs: List of WebEnvironments (one per agent/task)
        tasks: List of task dicts containing 'task' (MiniWoB task) and 'seed'
        tapes_dir: Directory to save processed episodes
        split_name: Name of split ('train' or other)

    Returns:
        Tuple containing:
        - List of WebAgents (agent state can be modified during batch run)
        - List of completed WebTapes
        - List of training samples with rewards and logprobs
        - Dictionary of performance statistics and execution times
    """
    assert len(agent_replicas) == len(envs) == len(tasks), (
        f"Number of agents ({len(agent_replicas)}), environments ({len(envs)}), " f"and tasks ({len(tasks)}) must match"
    )

    start_make_data = time.time()
    os.makedirs(tapes_dir, exist_ok=True)
    reward_stats = defaultdict(list)  # map from parent tape id to list of rewards
    success_stats = defaultdict(list)  # map from parent tape id to list of successes
    no_errors_stats = defaultdict(list)  # map from parent tape id to list of no_errors
    prompt_tokens_stats = defaultdict(list)  # map from parent tape id to list of prompt tokens length
    output_tokens_stats = defaultdict(list)  # map from parent tape id to list of output tokens length
    overflow_stats = defaultdict(list)  # map from parent tape id to list of overflows
    training_samples: List[TrainingText] = []

    logger.info(f"Run the agent on {split_name}")

    ### STEP 1: run the agents on their tasks in parallel ###
    start_making_tapes = time.time()
    final_tapes = []
    updated_agents = []

    for batch_start in range(0, len(agent_replicas), n_processes):
        batch_end = min(batch_start + n_processes, len(agent_replicas))
        batch_agents = agent_replicas[batch_start:batch_end]
        batch_envs = envs[batch_start:batch_end]
        batch_tasks = tasks[batch_start:batch_end]

        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            # futures = [
            #     executor.submit(batch_run_agent_replica, agent, env, task)
            #     for agent, env, task in zip(batch_agents, batch_envs, batch_tasks)
            # ]
            # results = [future.result() for future in futures]
            # other way of doing this:
            results = executor.map(batch_run_agent_replica, batch_agents, batch_envs, batch_tasks)
            batch_updated_agents, batch_tapes = zip(*[(r[0], r[1]) for r in results])

            updated_agents.extend(batch_updated_agents)
            final_tapes.extend(batch_tapes)

        logger.info(f"Completed batch {batch_start//n_processes + 1} of {(len(agent_replicas) + n_processes - 1)//n_processes}")

    agent_replicas = updated_agents  # Update the original agent_replicas list

    logger.info(f"Making tapes took {time.time() - start_making_tapes}")

    ### STEP 2: extract training samples and stats from the tapes ###
    for new_tape in tqdm(final_tapes, total=len(final_tapes), desc="Extracting training data from tapes", unit="tape"):
        tape_training_samples, tape_stats = extract_tape_training_samples_and_stats(
            new_tape, agent_replicas[0], split_name
        )
        # 1 tape -> multiple training samples because of multiple LLM calls
        training_samples.extend(tape_training_samples)
        reward_stats[new_tape.metadata.parent_id].append(tape_stats["reward"])
        success_stats[new_tape.metadata.parent_id].append(tape_stats["success"])
        no_errors_stats[new_tape.metadata.parent_id].append(tape_stats["no_error"])
        prompt_tokens_stats[new_tape.metadata.parent_id].append(tape_stats["prompt_tokens"])
        output_tokens_stats[new_tape.metadata.parent_id].append(tape_stats["output_tokens"])
        overflow_stats[new_tape.metadata.parent_id].append(tape_stats["overflows"])

    ### STEP 3: save the tapes ###
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
    return agent_replicas, final_tapes, training_samples, stats


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
                logger.info(f"[it{state['iteration']}] Splitting tasks, envs, and agents across {n_services} VLLM services")

                # Calculate how many agents to create per service (distribute evenly)
                agents_per_service = (n_train_tasks + n_services - 1) // n_services  # Ceiling division
                logger.info(f"[it{state['iteration']}] Each service will be used for {agents_per_service} agents, environment, and task")
                # Create LLMs, agents, and environments - one of each per task, agents distributed across services
                train_llms = [
                    TrainableLLM(
                        base_url=base_url,
                        model_name=str(assistant_model_path),
                        tokenizer_name=str(assistant_model_path),
                        parameters=cfg.llm.parameters,
                        use_cache=False,
                        collect_logprobs=True,
                        observe_llm_calls=False,
                    )
                    for base_url in vllm_services
                ]
                train_agent_replicas = []
                train_envs = []
                for llm_idx, llm in enumerate(train_llms):
                    # Calculate how many agents to create for this service
                    start_idx = llm_idx * agents_per_service
                    end_idx = min((llm_idx + 1) * agents_per_service, n_train_tasks)
                    n_agents_this_service = end_idx - start_idx
                    train_agent_replicas.extend([WebAgent.create(llm) for _ in range(n_agents_this_service)])
                    train_env_path = f"{exp_path}/envs/train/it{state['iteration']}/llm{llm_idx}"
                    train_envs.extend([WebEnvironment(
                        exp_path=f"{train_env_path}/agent{i}",
                        headless=cfg.env.headless,
                        ax_tree=cfg.env.ax_tree,
                        html=cfg.env.html,
                        markdown_html=cfg.env.markdown_html,
                    ) for i in range(n_agents_this_service)])
                assert len(train_agent_replicas) == len(train_envs) == len(train_tasks), (
                    f"Number of train agents ({len(train_agent_replicas)}) and environments ({len(train_envs)}) "
                    f"!= number of train tasks ({len(train_tasks)})"
                )
                # Create train split
                logger.info(
                    f"[it{state['iteration']}] Created train split with {len(train_agent_replicas)} agents & environments "
                    f"(distributed across {n_services} VLLM services) "
                    f"for {len(train_tasks)} tasks"
                )
                splits = [("train", train_agent_replicas, train_envs, train_tasks)]

                # Create test split if needed
                if state["iteration"] % cfg.test_every_n_iterations == 0 and cfg.test_every_n_iterations > 0:
                    # Calculate how many agents to create for this service
                    logger.info(f"[it{state['iteration']}] Creating test split with {len(test_samples)} tasks, llms, envs, and agents")
                    n_test_tasks = len(test_samples)
                    agents_per_service = (n_test_tasks + n_services - 1) // n_services
                    logger.info(f"[it{state['iteration']}] Each service will be used for {agents_per_service} agents, environment, and task")
                    # Create test LLMs, agents, and environments - one of each per task
                    test_llms = [
                        TrainableLLM(
                            base_url=base_url,
                            model_name=str(assistant_model_path),
                            tokenizer_name=str(assistant_model_path),
                            parameters=cfg.test_llm.parameters,
                            use_cache=False,
                            observe_llm_calls=False,
                        )
                        for base_url in vllm_services
                    ]
                    test_agent_replicas = []
                    test_envs = []
                    for llm_idx, llm in enumerate(test_llms):
                        start_idx = llm_idx * agents_per_service
                        end_idx = min((llm_idx + 1) * agents_per_service, n_test_tasks)
                        n_agents_this_service = end_idx - start_idx
                        test_agent_replicas.extend([WebAgent.create(llm) for _ in range(n_agents_this_service)])
                        test_env_path = f"{exp_path}/envs/test/it{state['iteration']}/llm{llm_idx}"
                        test_envs.extend([WebEnvironment(
                            exp_path=f"{test_env_path}/agent{i}",
                            headless=cfg.env.headless,
                            ax_tree=cfg.env.ax_tree,
                            html=cfg.env.html,
                            markdown_html=cfg.env.markdown_html,
                        ) for i in range(n_agents_this_service)])
                    assert len(test_agent_replicas) == len(test_envs) == len(test_samples), (
                        f"Number of test agents ({len(test_agent_replicas)}) and environments ({len(test_envs)}) "
                        f"!= number of test tasks ({len(test_samples)})"
                    )
                    logger.info(
                        f"[it{state['iteration']}] Created test split with {len(test_agent_replicas)} agents & environments "
                        f"(distributed across {n_services} VLLM services) "
                        f"for {len(test_samples)} tasks"
                    )
                    splits.append(("test", test_agent_replicas, test_envs, test_samples))

                ### Step 2.2: generate continuations for each split ###
                for split_name, agent_replicas, envs, tasks in splits:
                    tapes_dir = exp_path / "tapes" / split_name / str(state["iteration"])
                    agent_replicas_with_stats, new_tapes, split_training_samples, stats = generate_training_data(
                        agent_replicas, envs, tasks, tapes_dir, split_name, cfg.n_processes_for_data_generation
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
                    logger.info(f"{split_name} stats:")
                    for stat_name, stat_value in stats.items():
                        logger.info(f"{stat_name}: {stat_value}")
                # keep track of the starting time of the assistant model
                assistant_model_starting_time = vllm_service_manager.get_stats()["starting_time"]
            ### end with vllm_service_manager("actor")
        except Exception as e:
            logger.error(colored(f"Failed to solve task: {e}", "red"))
            raise e

        ### Step 3: log all stats from forward pass###
        training_samples: list[TrainingText] = all_results["train"]["training_samples"]
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
        logger.info("Logprob population stats:")
        for stat_name, stat_value in logprob_stats.items():
            logger.info(f"{stat_name}: {stat_value}")
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
