import json
import logging
import os
import subprocess
from typing import Any, Counter, Generator

import yaml
from huggingface_hub import snapshot_download
from termcolor import colored

from tapeagents.io import load_tapes, save_json_tape
from tapeagents.orchestrator import main_loop
from tapeagents.renderers import step_view

from .agent import GaiaAgent
from .environment import GaiaEnvironment
from .scorer import question_scorer
from .steps import GaiaAnswer, SearchAction
from .tape import GaiaMetadata, GaiaTape

logger = logging.getLogger(__name__)

DATASET_DIR = "data/gaia/"


def get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def tape_correct(tape: GaiaTape) -> bool:
    if not tape.metadata.result:
        return False
    predicted = str(tape.metadata.result)
    golden = tape.metadata.task["Final answer"]
    if golden == "?":  # placeholder for hidden answer in test set
        return False
    return question_scorer(predicted, golden)


def calculate_accuracy(tapes: list[GaiaTape], show_intermediate=False, show_wrong=False):
    accs = []
    accuracy = 0.0
    for tape in tapes:
        correct = tape_correct(tape)
        if show_wrong and not correct:
            for step in tape:
                print("-" * 80)
                print(step_view(step))
            print("=" * 120)

        accs.append(int(correct))
        accuracy = sum(accs) * 100 / len(accs)
        if show_intermediate:
            print(
                tape.metadata.task["Final answer"],
                "|",
                colored(str(tape.metadata.result), "green" if correct else "red"),
            )
            print(f"{len(accs)}: Accuracy {accuracy:.2f}")
    return accuracy, sum(accs)


def majority_vote(results: list[Any]) -> int:
    result_counts = Counter(results)
    top = result_counts.most_common()
    best, _ = top.pop(0)
    while len(top) and best in ["", "None", None]:
        best, _ = top.pop(0)
    best_idx = [i for i, result in enumerate(results) if result == best][0]
    return best_idx


def download_dataset():
    logger.info("Downloading GAIA dataset...")
    repo = "gaia-benchmark/GAIA"
    os.makedirs(DATASET_DIR, exist_ok=True)
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=DATASET_DIR, local_dir_use_symlinks=False)


def load_dataset(split: str):
    tasks = {1: [], 2: [], 3: []}
    fname = os.path.join(DATASET_DIR, "2023", split, "metadata.jsonl")
    if not os.path.exists(fname):
        download_dataset()
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Dataset not found: {fname}")
    with open(fname) as f:
        for line in f:
            task = json.loads(line)
            if task["file_name"]:
                task["file_name"] = os.path.join(DATASET_DIR, "2023", split, task["file_name"])
            tasks[task["Level"]].append(task)

    logger.info(f"GAIA Tasks: Level 1: {len(tasks[1])}, Level 2: {len(tasks[2])}, Level 3: {len(tasks[3])}")
    return tasks


def solve_task(
    task: dict,
    agent: GaiaAgent,
    env: GaiaEnvironment,
    level: int,
    retries: int = 3,
    max_loops: int = 50,
) -> Generator[GaiaTape, None, None]:
    """Solve GAIA task. 
    
    This function is a generator that yields intermediate tapes during the solving process.
    The last tape will contain the agent's response.
    
    """
    start_steps = env.task_to_observations(task)
    solved = None
    result = None
    while not solved and retries:
        tape = GaiaTape(steps=start_steps)
        try:
            for event in main_loop(agent, tape, env, max_loops=max_loops):
                if partial_tape := (event.agent_tape or event.env_tape):
                    tape = partial_tape
                    tape.metadata = GaiaMetadata.model_validate(
                            tape.metadata.model_dump() | {"task": task, "level": level}
                        )                    
                    yield tape
                if n_search_repetitions(tape) >= 3:
                    break
        except Exception as e:
            tape.metadata.error = str(e)
            logger.exception(f"Failed to solve task: {e}")
            break
        result = tape[-1].answer if isinstance(tape[-1], GaiaAnswer) else None
        result = str(result) if result is not None else ""
        solved = result != ""
        retries -= 1
    logger.info(f"Expected: {task['Final answer']}, Agent produced: {result}")
    tape.metadata = GaiaMetadata.model_validate(
        tape.metadata.model_dump() | {"task": task, "result": result, "level": level}
    )
    yield tape


def n_search_repetitions(tape: GaiaTape) -> int:
    steps_by_query = {}
    for step in tape:
        if isinstance(step, SearchAction):
            steps_by_query[step.query] = steps_by_query.get(step.query, 0) + 1
    return max(steps_by_query.values(), default=0)


def ensemble_results(all_tapes: list[list[GaiaTape]], oracle: bool = False) -> list[GaiaTape]:
    ensemble = []
    improved = 0
    degraded = 0
    for i, tapes in enumerate(zip(*all_tapes)):
        tapes: list[GaiaTape] = tapes
        results = [tape.metadata.result for tape in tapes]
        most_common_idx = majority_vote(results)
        if oracle:
            for j, tape in enumerate(tapes):
                if tape_correct(tape):
                    most_common_idx = j
                    break
        best_tape = tapes[most_common_idx].copy()

        orig = tapes[0]
        orig_correct = int(tape_correct(orig))
        ensemble_correct = int(tape_correct(best_tape))
        expected = tapes[0].metadata.task["Final answer"]
        log_message = f"{i+1}: {orig_correct} -> {ensemble_correct} | choose {most_common_idx+1} ({best_tape.metadata.result}) of {results}. Expected: {expected}"
        if orig_correct < ensemble_correct:
            logger.info("Improved")
            improved += 1
            logger.info(log_message)
        elif orig_correct > ensemble_correct:
            logger.info("Degraded")
            degraded += 1
            logger.info(log_message)
        ensemble.append(best_tape)
    logger.info(f"Improved {improved}, degraded {degraded}")
    return ensemble


def ensemble_files(tape_dirs: list[str], out_dir: str = ""):
    tapes: list[list[GaiaTape]] = [load_tapes(GaiaTape, tape_dir, file_extension=".json") for tape_dir in tape_dirs]  # type: ignore
    ensembled_tapes = ensemble_results(tapes)
    acc, num = calculate_accuracy(ensembled_tapes, show_intermediate=False)
    logger.info(f"Ensembled {len(tape_dirs)} accuracy: {acc:.2f} ({num} of {len(ensembled_tapes)})")
    if out_dir:
        out_tapes_dir = os.path.join(out_dir, "tapes")
        os.makedirs(out_tapes_dir)
        for i, tape in enumerate(ensembled_tapes):
            save_json_tape(tape, out_tapes_dir, f"tape{i+1:03d}")
        with open(os.path.join(out_dir, "ensemble_config.json"), "w") as f:
            json.dump({"sources": tape_dirs, "mode": "majority"}, f, indent=2)
        logger.info(f"Saved to {out_dir}")


def get_exp_config_dict(exp_path):
    config_path = os.path.join(exp_path, ".hydra", "config.yaml")
    assert os.path.exists(config_path), f"Config file {config_path} not found"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg
