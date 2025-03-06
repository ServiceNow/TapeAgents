import json
import logging
import os
import sys

from tapeagents.io import load_tapes

from ..eval import get_exp_config_dict, load_dataset, tape_correct
from ..steps import GaiaTape

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def main(exp_path: str):
    assert os.path.isdir(exp_path), f"Directory {exp_path} does not exist or is not a directory"
    cfg = get_exp_config_dict(exp_path)
    tasks = load_dataset(cfg["split"])
    submission_file = os.path.join(exp_path, "submission.jsonl")
    submission = []
    expected_tasks_num = sum(len(level_tasks) for level_tasks in tasks.values())
    total_unsolved = 0
    all_tapes = load_tapes(GaiaTape, os.path.join(exp_path, "tapes"), file_extension=".json")  # type: ignore
    tapes_by_task_id = {tape.metadata.task["task_id"]: tape for tape in all_tapes}
    print(f"Loaded {len(all_tapes)} tapes")
    correct = 0
    for level, level_tasks in tasks.items():
        unsolved = 0
        tapes = []
        for task in level_tasks:
            tape = tapes_by_task_id.get(task["task_id"])
            last_step = tape.steps[-1]
            model_answer = last_step.answer if last_step.kind == "gaia_answer_action" else None
            if model_answer != tape.metadata.result:
                print(f"Model answer '{model_answer}' does not match tape result '{tape.metadata.result}'")
            if not model_answer:
                unsolved += 1
                model_answer = "0"
            if tape_correct(tape):
                correct += 1
            line = {"task_id": tape.metadata.task["task_id"], "model_answer": str(model_answer)}
            submission.append(line)
            tapes.append(tape)
        logger.info(f"Level {level+1}: gathered {len(tapes)} task solutions, {unsolved} unsolved, {correct} correct")
        total_unsolved += unsolved

    if len(submission) != expected_tasks_num:
        logger.info(f"Submission is not ready, expected {expected_tasks_num} task tapes, got {len(submission)}")
        return None
    else:
        logger.info(f"Submission is ready, has {total_unsolved} unsolved tasks out of {expected_tasks_num}")
        logger.info(f"Writing submission with {len(submission)} tasks to {submission_file}")
        with open(submission_file, "w") as f:
            for line in submission:
                f.write(f"{json.dumps(line)}\n")
        return submission_file


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: examples.gaia_agent.scripts.prepare_submission <exp_dir>"
    submission_file = main(sys.argv[1])
