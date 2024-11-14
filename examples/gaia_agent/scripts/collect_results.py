import os
from collections import defaultdict

from examples.gaia_agent.tape import GaiaTape
from tapeagents.io import load_tapes

from ..eval import majority_vote, tape_correct


def main(root: str, runs: list[str]):
    by_level_by_run = defaultdict(lambda: defaultdict(list))
    for run in runs:
        tapes_dir = os.path.join(root, run, "tapes")
        tapes: list[GaiaTape] = load_tapes(GaiaTape, tapes_dir, file_extension=".json")  # type: ignore
        for tape in tapes:
            by_level_by_run[tape.metadata.level][run].append(tape)

    maj_name = f"maj@{len(runs)}"
    avg = {run: [] for run in runs} | {maj_name: []}
    print("Accuracy")
    for lvl_name, lvl_runs in by_level_by_run.items():
        acc_by_run = defaultdict(list)
        avg_acc = []
        run_names = []
        run_tapes = []
        for run_name, tapes in lvl_runs.items():
            run_names.append(run_name)
            run_tapes.append(tapes)
            acc_by_run[run_name] = [int(tape_correct(tape)) for tape in tapes]
            avg[run_name] += acc_by_run[run_name]
        for tapes in zip(*run_tapes):
            best_idx = majority_vote([tape.metadata.result for tape in tapes])
            best_tape = tapes[best_idx]
            avg_acc.append(int(tape_correct(best_tape)))
        avg[maj_name] += avg_acc
        for run_name, acc in acc_by_run.items():
            print(f"L{lvl_name} {run_name}: {sum(acc) / len(acc):.3f} ({sum(acc)} of {len(acc)})")
        print(f"L{lvl_name} {maj_name}: {sum(avg_acc) / len(avg_acc):.3f} ({sum(avg_acc)} of {len(avg_acc)})")
        print()

    for run, acc in avg.items():
        print(f"Avg. {run}: {sum(acc) / len(acc):.3f} ({sum(acc)} of {len(acc)})")


if __name__ == "__main__":
    runs = [
        "gpt4o_mini_val_batch32_5",
        "gpt4o_mini_val_batch32_6",
        "gpt4o_mini_val_batch32_7",
        "gpt4o_mini_val_batch32_t0_2",
        "gpt4o_mini_val_batch32_t05",
    ]
    main(root="../gaia/runs/", runs=runs)


# gp4o-mini, 3 runs
# L1 gpt4o_mini_t02: 0.302 (16 of 53)
# L1 gpt4o_mini_t02_3: 0.453 (24 of 53)
# L1 gpt4o_mini_t05: 0.377 (20 of 53)
# L1 maj@3: 0.453 (24 of 53)

# L2 gpt4o_mini_t02: 0.244 (21 of 86)
# L2 gpt4o_mini_t02_3: 0.209 (18 of 86)
# L2 gpt4o_mini_t05: 0.151 (13 of 86)
# L2 maj@3: 0.267 (23 of 86)

# L3 gpt4o_mini_t02: 0.077 (2 of 26)
# L3 gpt4o_mini_t02_3: 0.000 (0 of 26)
# L3 gpt4o_mini_t05: 0.038 (1 of 26)
# L3 maj@3: 0.038 (1 of 26)

# Avg. gpt4o_mini_t02: 0.236 (39 of 165)
# Avg. gpt4o_mini_t02_3: 0.255 (42 of 165)
# Avg. gpt4o_mini_t05: 0.206 (34 of 165)
# Avg. maj@3: 0.291 (48 of 165)


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
