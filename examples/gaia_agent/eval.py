import json
import logging
import os
import shutil
import subprocess
from typing import Any, Counter, Generator

import yaml
from huggingface_hub import snapshot_download
from pdf2image import convert_from_path
from termcolor import colored

from tapeagents.environment import ToolCollectionEnvironment
from tapeagents.io import load_tapes, save_json_tape, save_tape_images
from tapeagents.orchestrator import main_loop
from tapeagents.renderers import step_view
from tapeagents.tools.search import SearchAction
from tapeagents.tools.simple_browser import SimpleTextBrowser

from .agent import GaiaAgent
from .scorer import question_scorer
from .steps import GaiaAnswer, GaiaQuestion, ImageObservation
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
    env: ToolCollectionEnvironment,
    level: int,
    task_number: int,
    exp_path: str,
    retries: int,
    aggregate_majority: int,
    max_iterations: int,
) -> GaiaTape:
    """Solve GAIA task.

    This function is a generator that yields intermediate tapes during the solving process.
    The last tape will contain the agent's response.

    """
    assert aggregate_majority % 2 > 0, "Aggregate majority should be an odd number"
    assert retries < 10, "Too many retries, should not be more than 10"
    assert aggregate_majority <= retries, "Aggregate majority should be less than or equal to retries"

    tapes_dir = os.path.join(exp_path, "tapes")
    majority = (aggregate_majority // 2 + 1) if aggregate_majority > 1 else 1
    start_steps = task_to_observations(task)
    results = []
    attempt = 0
    top_result = ""
    while len(results) < aggregate_majority and attempt < (retries * aggregate_majority):
        tape = GaiaTape(steps=start_steps)
        filename = f"l{level}_task{task_number:03d}_attempt{attempt}"
        if tape_exist_with_result(filename, tapes_dir):
            logger.info(f"Tape already exist with result: {filename}, skip")
            continue
        metadata = GaiaMetadata(
            task=task,
            level=level,
            task_number=task_number,
            filename=filename,
            attempt_number=attempt,
        )
        try:
            for event in main_loop(agent, tape, env, max_loops=max_iterations):
                if partial_tape := (event.agent_tape or event.env_tape):
                    tape = partial_tape
                    tape.metadata = metadata
                    save_json_tape(tape, tapes_dir, f"{filename}_unfinished")
                if n_search_repetitions(tape) >= 3:
                    break
        except Exception as e:
            logger.error(f"Failed to solve task: {e}")
        if isinstance(tape.steps[-1], GaiaAnswer) and tape.steps[-1].answer not in ["None", "none"]:
            metadata.result = tape.steps[-1].answer
        attempt += 1
        tape.metadata = metadata
        tmp_file = os.path.join(tapes_dir, f"{filename}_unfinished.json")
        if os.path.exists(tmp_file):
            os.unlink(tmp_file)
        save_json_tape(tape, tapes_dir, filename)
        save_tape_images(tape, os.path.join(exp_path, "images"))
        if metadata.result:
            results.append(metadata.result)
            counts = Counter(results)
            top_result, num_occured = counts.most_common(1)[0]
            if num_occured >= majority:
                logger.info(f"Majority reached: {top_result} ({num_occured} of {len(results)}), break")
                break
    logger.info(f"Expected: {task['Final answer']}, Agent produced: {top_result}")
    return tape  # type: ignore


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
    filename: str | None = steps[0].filename  # type: ignore
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
                content = browser.get_whole_document(filename)
                document_text = f"\n\n{ext.upper()} document content:\n{content}\n"
                if len(document_text) > max_doc_length:
                    document_text = ""
                document_text += f"\nPath to the mentioned document: {filename}"
            else:
                document_text = "\nDocument pages attached as images below"
        steps[0].content += document_text  # type: ignore
    steps[0].filename = None  # type: ignore
    return steps


def tape_exist_with_result(tape_name: str, tapes_dir: str) -> bool:
    result = ""
    tape_path = os.path.join(tapes_dir, f"{tape_name}.json")
    if os.path.exists(tape_path):
        with open(tape_path) as f:
            tape_dict = json.load(f)
        result = tape_dict["metadata"]["result"]
    return os.path.exists(tape_path) and result not in ["", None, "None", "none"]
