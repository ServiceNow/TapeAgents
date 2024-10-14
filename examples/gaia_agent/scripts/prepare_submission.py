import json
import os
import sys

import yaml

from examples.gaia_agent.eval import load_dataset
from examples.gaia_agent.tape import GaiaTape
from tapeagents.io import load_tapes


def main(exp_path: str):
    assert os.path.isdir(exp_path), f"Directory {exp_path} does not exist or is not a directory"
    config_path = os.path.join(exp_path, ".hydra", "config.yaml")
    assert os.path.exists(config_path), f"Config file {config_path} not found"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    data_dir = cfg["data_dir"]

    submission_file = os.path.join(exp_path, "submission.jsonl")
    submission = []
    tasks = load_dataset(data_dir)
    expected_tasks_num = sum(len(level_tasks) for level_tasks in tasks.values())
    total_unsolved = 0
    for level, level_tasks in tasks.items():
        unsolved = 0
        tapes = []
        for i in range(len(level_tasks)):
            fname = f"l{level}_task{i}"
            tape_path = os.path.join(exp_path, "tapes", fname + ".json")
            try:
                tape: GaiaTape = load_tapes(GaiaTape, tape_path)[0]  # type: ignore
            except FileNotFoundError:
                print(f"Skipping {fname} as tape file not found")
                continue
            last_step = tape.steps[-1]
            model_answer = last_step.answer if last_step.kind == "gaia_answer_action" else None
            if not model_answer:
                unsolved += 1
            line = {"task_id": tape.metadata.task["task_id"], "model_answer": str(model_answer)}
            submission.append(line)
            tapes.append(tape)
        print(f"Level {level+1}: gathered {len(tapes)} task solutions, {unsolved} unsolved")
        total_unsolved += unsolved

    if len(submission) != expected_tasks_num:
        print(f"Submission is not ready, expected {expected_tasks_num} task tapes, got {len(submission)}")
    else:
        print(f"Submission is ready, has {total_unsolved} unsolved tasks out of {expected_tasks_num}")
        print(f"Writing submission with {len(submission)} tasks to {submission_file}")
        with open(submission_file, "w") as f:
            for line in submission:
                f.write(f"{json.dumps(line)}\n")


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: examples.gaia_agent.scripts.prepare_submission <exp_dir>"
    main(sys.argv[1])
