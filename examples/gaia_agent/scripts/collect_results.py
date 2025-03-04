import logging
import os
from collections import defaultdict

from tapeagents.io import load_tapes

from ..eval import ensemble_files, load_dataset, majority_vote, tape_correct
from ..steps import GaiaTape

logging.basicConfig(level=logging.INFO)


def main(root: str, runs: list[str], split: str):
    tasks_by_level = load_dataset(split)
    tasks = [task for level in tasks_by_level.values() for task in level]
    print(f"Tasks: {len(tasks)}")
    task_solutions = defaultdict(list)
    for run in runs:
        tapes_dir = os.path.join(root, run, "tapes")
        tapes: list[GaiaTape] = load_tapes(GaiaTape, tapes_dir, file_extension=".json")
        print(f"Run {run}: {len(tapes)} tapes")
        for tape in tapes:
            try:
                task_solutions[tape.metadata.task["task_id"]].append(tape)
            except KeyError as e:
                print(f"Missing task id of task {tape.metadata.task} in tape {tape.metadata.id}")
                raise e
    maj_name = f"maj@{len(runs)}"
    results = {run: [] for run in runs} | {maj_name: []}
    for task in tasks:
        tapes = task_solutions[task["task_id"]]
        task_results = [tape.metadata.result for tape in tapes]
        if len(task_results) < len(runs):
            print(f"Missing tapes for task {task['task_id']}, add {len(runs) - len(task_results)} Nones")
            task_results += [None] * (len(runs) - len(task_results))
        best_idx = majority_vote(task_results)
        results[maj_name].append(tapes[best_idx])
        for i in range(len(runs)):
            results[runs[i]].append(tapes[i] if i < len(tapes) else None)
    for run, results in results.items():
        assert len(results) == len(tasks), "Results length mismatch"
        acc = [int(tape_correct(tape)) if tape else 0 for tape in results]
        empty_results = [tape.metadata.result in ["", "None", None] for tape in results]
        print(f"Avg. {run}: {sum(acc) / len(acc):.3f} ({sum(acc)} of {len(acc)}), {sum(empty_results)} no results")


if __name__ == "__main__":
    root = "outputs/gaia/runs/"
    runs = [
        "sonnet37_test1",
        "sonnet37_test2",
        "sonnet37_test3",
    ]
    ensemble = "sonnet37_test_maj3"
    split = "test"
    main(root, runs, split)
    if ensemble:
        tape_dirs = [os.path.join(root, run, "tapes") for run in runs]
        out = os.path.join(root, ensemble)
        ensemble_files(tape_dirs, out)


# gp4o-mini, 3 runs
# L1 gpt4o_mini_val_image_pdf2: 0.415 (22 of 53)
# L1 gpt4o_mini_val_image_pdf3: 0.396 (21 of 53)
# L1 gpt4o_mini_val_image_pdf4: 0.415 (22 of 53)
# L1 maj@3: 0.472 (25 of 53)

# L2 gpt4o_mini_val_image_pdf2: 0.302 (26 of 86)
# L2 gpt4o_mini_val_image_pdf3: 0.244 (21 of 86)
# L2 gpt4o_mini_val_image_pdf4: 0.186 (16 of 86)
# L2 maj@3: 0.291 (25 of 86)

# L3 gpt4o_mini_val_image_pdf2: 0.077 (2 of 26)
# L3 gpt4o_mini_val_image_pdf3: 0.115 (3 of 26)
# L3 gpt4o_mini_val_image_pdf4: 0.077 (2 of 26)
# L3 maj@3: 0.077 (2 of 26)

# Avg. gpt4o_mini_val_image_pdf2: 0.303 (50 of 165)
# Avg. gpt4o_mini_val_image_pdf3: 0.273 (45 of 165)
# Avg. gpt4o_mini_val_image_pdf4: 0.242 (40 of 165)
# Avg. maj@3: 0.315 (52 of 165)


# gpt4o, 3 runs
# L1 gpt4o_1: 0.377 (20 of 53)
# L1 gpt4o_latest_1: 0.396 (21 of 53)
# L1 gpt4o_t02_1: 0.472 (25 of 53)
# L1 maj@3: 0.491 (26 of 53)

# L2 gpt4o_1: 0.314 (27 of 86)
# L2 gpt4o_latest_1: 0.267 (23 of 86)
# L2 gpt4o_t02_1: 0.349 (30 of 86)
# L2 maj@3: 0.360 (31 of 86)

# L3 gpt4o_1: 0.000 (0 of 26)
# L3 gpt4o_latest_1: 0.077 (2 of 26)
# L3 gpt4o_t02_1: 0.038 (1 of 26)
# L3 maj@3: 0.000 (0 of 26)

# Avg. gpt4o_1: 0.285 (47 of 165)
# Avg. gpt4o_latest_1: 0.279 (46 of 165)
# Avg. gpt4o_t02_1: 0.339 (56 of 165)
# Avg. maj@3: 0.345 (57 of 165)
