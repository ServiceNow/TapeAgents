import json
import logging
import os
import shutil
import subprocess
import time
from typing import Any, Counter

import yaml
from huggingface_hub import snapshot_download
from pdf2image import convert_from_path
from termcolor import colored

from tapeagents.agent import Agent
from tapeagents.environment import ToolCollectionEnvironment
from tapeagents.io import load_tapes, save_json_tape
from tapeagents.orchestrator import main_loop
from tapeagents.renderers import step_view
from tapeagents.tools.code_executor import PythonCodeAction
from tapeagents.tools.simple_browser import SimpleTextBrowser
from tapeagents.tools.web_search import SearchAction

from .scorer import question_scorer
from .steps import GaiaAnswer, GaiaMetadata, GaiaQuestion, GaiaTape, ImageObservation

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


def tape_without_result(tape: GaiaTape) -> bool:
    return tape.metadata.result in ["", "None", None]


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
    agent: Agent,
    env: ToolCollectionEnvironment,
    level: int,
    task_num: int,
    tapes_dir: str,
    max_loops: int = 50,
    max_action_repetitions: int = 3,
) -> GaiaTape:
    start_steps = task_to_observations(task)
    t = time.perf_counter()
    tape = GaiaTape(steps=start_steps)
    loop_timout_sec = 30 * 60
    tmp_file = os.path.join(tapes_dir, f"l{level}_task{task_num:03d}.json.tmp")
    try:
        start_time = time.perf_counter()
        for event in main_loop(agent, tape, env, max_loops=max_loops):
            if time.perf_counter() - start_time > loop_timout_sec:
                tape.metadata.error = "Timeout, task took too long"
                logger.warning("Timeout, task took too long")
                break
            if partial_tape := (event.agent_tape or event.env_tape):
                tape = partial_tape
                tape.metadata = GaiaMetadata.model_validate(tape.metadata.model_dump() | {"task": task, "level": level})
                save_json_tape(tape, tmp_file)
            if action_repetitions(tape) >= max_action_repetitions:
                break
    except Exception as e:
        tape.metadata.error = str(e)
        logger.exception(f"Failed to solve task: {e}")
    result = tape[-1].answer if isinstance(tape[-1], GaiaAnswer) else None  # type: ignore
    result = str(result) if result is not None else ""
    logger.info(f"Expected: {task['Final answer']}, Agent produced: {result}")
    tape.metadata = GaiaMetadata.model_validate(
        tape.metadata.model_dump() | {"task": task, "result": result, "level": level}
    )
    tape.metadata.other["timers"] = {"solve_task": time.perf_counter() - t}
    if os.path.exists(tmp_file):
        os.unlink(tmp_file)
    return tape


def action_repetitions(tape: GaiaTape) -> int:
    unique_actions = {}
    for step in tape:
        if isinstance(step, (SearchAction, PythonCodeAction)):
            key = step.llm_view()
            unique_actions[key] = unique_actions.get(key, 0) + 1
    return max(unique_actions.values(), default=0)


def ensemble_results(all_tapes: list[list[GaiaTape]], oracle: bool = False) -> list[GaiaTape]:
    ensemble = []
    improved = 0
    degraded = 0
    added_result = 0
    no_result = 0
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

        tape0 = tapes[0]
        tape0_correct = int(tape_correct(tape0))
        ensemble_correct = int(tape_correct(best_tape))
        expected = tapes[0].metadata.task["Final answer"]
        change = "switched" if best_tape.metadata.result != results[0] else "same"
        log_message = f"{i+1}: {tape0_correct} -> {ensemble_correct} | {change} {most_common_idx+1} ({best_tape.metadata.result}) of {results}. Expected: {expected}"
        if tape0_correct < ensemble_correct:
            logger.info("Improved")
            improved += 1
        elif tape0_correct > ensemble_correct:
            logger.info("Degraded")
            degraded += 1
        if tape_without_result(best_tape):
            no_result += 1
        if tape_without_result(tape0) and not tape_without_result(best_tape):
            added_result += 1
            logger.info("Added result")
        logger.info(log_message)
        ensemble.append(best_tape)
    logger.info(f"Improved {improved}, degraded {degraded}, no result {no_result}, added result {added_result}")
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


def pdf_to_images(filename: str, n_pages: int = 3):
    images = []
    for i, image in enumerate(convert_from_path(filename)):
        page_index = i + 1
        page_fname = filename[:-4] + f"_{page_index}.png"
        if os.path.exists(page_fname):
            images.append(page_fname)
            continue
        image.save(page_fname)
        images.append(page_fname)
    return images[:n_pages], len(images)


def task_to_observations(task: dict, max_doc_length: int = 8000) -> list[GaiaQuestion | ImageObservation]:
    logger.info(f"Question: {task['Question']}")
    browser = SimpleTextBrowser()
    steps: list[GaiaQuestion | ImageObservation] = [GaiaQuestion.from_task(task)]
    filename: str | None = steps[0].filename
    steps[0].filename = None
    if filename:
        name, ext = filename.rsplit(".", maxsplit=1)
        ext = ext.lower()
        if ext == "zip":
            folder_name = name
            os.makedirs(folder_name, exist_ok=True)
            shutil.unpack_archive(filename, folder_name)
            document_text = "\n\nArchive contains the following files:\n"
            for i, file in enumerate(os.listdir(folder_name)):
                file_path = os.path.join(folder_name, file)
                content = browser.get_whole_document(file_path)
                file_text = f"{i+1}. {file}. Content:\n{content}\n\n"
                if len(file_text) > max_doc_length:
                    file_text = ""
                file_text += f"{i+1}. Path to the '{file}': {file_path}"
                document_text += file_text
        elif ext in ("png", "jpg", "jpeg"):
            steps.append(ImageObservation(image_path=filename, image_caption="Attached image"))
            document_text = ""
        else:
            attach_doc_text = True
            if ext == "pdf":
                images, total_pages = pdf_to_images(filename)
                if total_pages <= 3:
                    attach_doc_text = False
                for i, img_path in enumerate(images):
                    steps.append(ImageObservation(image_path=img_path, image_caption=f"PDF page {i+1}"))
            if attach_doc_text:
                try:
                    content = browser.get_whole_document(filename)
                except Exception as e:
                    logger.exception(f"Failed to read document: {e}")
                    content = ""
                document_text = f"\n\nAttached {ext.upper()} file content:\n{content}\n"
                if not len(content) or len(document_text) > max_doc_length:
                    document_text = ""
            else:
                document_text = "\nDocument pages attached as images below"
            steps[0].filename = filename
        steps[0].content += document_text  # type: ignore
    return steps
