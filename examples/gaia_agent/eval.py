import datetime
import json
import logging
import os
import shutil
import subprocess
from typing import Any, Counter, Iterable
from uuid import uuid4

import yaml
from pydantic import BaseModel, Field
from termcolor import colored

from tapeagents.io import load_tapes, save_json_tape
from tapeagents.orchestrator import main_loop
from tapeagents.rendering import step_view

from .agent import GaiaAgent
from .environment import GaiaEnvironment
from .scorer import question_scorer
from .steps import GaiaAnswer, GaiaQuestion, PlanThought
from .tape import GaiaMetadata, GaiaTape

logger = logging.getLogger(__name__)


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


def load_dataset(data_dir):
    tasks = {1: [], 2: [], 3: []}
    with open(f"{data_dir}/metadata.jsonl") as f:
        for line in f:
            task = json.loads(line)
            if task["file_name"]:
                task["file_name"] = f"{data_dir}/{task['file_name']}"
            tasks[task["Level"]].append(task)

    logger.info(f"GAIA Tasks: Level 1: {len(tasks[1])}, Level 2: {len(tasks[2])}, Level 3: {len(tasks[3])}")
    return tasks


def solve_task(task: dict, agent: GaiaAgent, env: GaiaEnvironment, n_attempts: int = 1) -> GaiaTape:
    question = task_to_question_step(task, env)
    tapes: list[GaiaTape] = []
    results: list[Any] = []
    previous_plans: list[str] = []
    while len(tapes) < n_attempts:
        predicted = None
        tries = 3
        while not predicted and tries:
            tape = GaiaTape(steps=[question])
            logger.info(colored(f"Attempt {len(tapes)+1}", "green"))
            discard_attempt = False
            planned = False
            step = None
            try:
                for event in main_loop(agent, tape, env, max_loops=30):
                    if event.agent_event and event.agent_event.step:
                        step = event.agent_event.step
                        tape = tape.append(step)  # type: ignore
                        if isinstance(step, PlanThought) and not planned:
                            plan_dump = "\n".join(step.plan)
                            if plan_dump in previous_plans:
                                logger.info("Plan already been used, discard attempt")
                                discard_attempt = True
                                break
                            else:
                                planned = True
                                previous_plans.append(plan_dump)
                    if event.observation:
                        tape = tape.append(event.observation)  # type: ignore
                if discard_attempt:
                    continue
            except Exception as e:
                tape.metadata.error = str(e)
                logger.exception(f"Failed to solve task: {e}")
                break
            predicted = step.answer if isinstance(step, GaiaAnswer) else None
            tries -= 1
        predicted = str(predicted)
        tapes.append(tape)
        results.append(predicted)
        logger.info(f"Expected: {task['Final answer']}, Agent produced: {predicted}")
    logger.info(f"Produced {len(tapes)} tapes, vote")
    best = majority_vote(results)
    logger.info(f"Majority vote best non-empty result: {best}, out of {results}")
    best_tape = tapes[best]
    best_tape.metadata = GaiaMetadata.model_validate(
        best_tape.metadata.model_dump() | {"task": task, "result": results[best]}
    )
    return best_tape


def task_to_question_step(task: dict, env: GaiaEnvironment, max_doc_length: int = 8000) -> GaiaQuestion:
    question = GaiaQuestion.from_task(task)
    if question.filename:
        name, ext = question.filename.rsplit(".", maxsplit=1)
        if ext == "zip":
            folder_name = name
            os.makedirs(folder_name, exist_ok=True)
            shutil.unpack_archive(question.filename, folder_name)
            document_text = "\n\nArchive contains the following files:\n"
            for i, file in enumerate(os.listdir(folder_name)):
                file_path = os.path.join(folder_name, file)
                content = env.browser.get_whole_document(file_path)
                file_text = f"{i+1}. {file}. Content:\n{content}\n\n"
                if len(file_text) > max_doc_length:
                    file_text = ""
                file_text += f"{i+1}. Path to the '{file}': {file_path}"
                document_text += file_text
        else:
            content = env.browser.get_whole_document(question.filename)
            document_text = f"\n\n{ext.upper()} document content:\n{content}\n"
            if len(document_text) > max_doc_length:
                document_text = ""
            document_text += f"\nPath to the mentioned document: {question.filename}"
        question.content += document_text
    question.filename = None
    logger.info(f"Question: {question.content}")
    return question


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
        assert not os.path.exists(out_dir), f"Directory {out_dir} already exists"
        for i, tape in enumerate(ensembled_tapes):
            save_json_tape(tape, out_dir, f"tape{i+1}")
        logger.info(f"Saved to {out_dir}")


def get_exp_config_dict(exp_path):
    config_path = os.path.join(exp_path, ".hydra", "config.yaml")
    assert os.path.exists(config_path), f"Config file {config_path} not found"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg
