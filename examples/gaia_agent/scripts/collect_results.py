from tapeagents.io import load_tapes

from ..eval import GaiaResults, majority_vote, tape_correct


def main(root: str, model: str, runs: list[str]):
    assert len(runs) == 3
    template = "{root}{run}/l{level}_{model}_run.json"

    lvl1 = []
    lvl2 = []
    lvl3 = []
    for run in runs:
        fname1 = template.format(root=root, run=run, level="1", model=model)
        fname2 = template.format(root=root, run=run, level="2", model=model)
        fname3 = template.format(root=root, run=run, level="3", model=model)
        lvl1.append(load_tapes(GaiaResults, fname1, file_extension=".json")[0])
        lvl2.append(load_tapes(GaiaResults, fname2, file_extension=".json")[0])
        lvl3.append(load_tapes(GaiaResults, fname3, file_extension=".json")[0])

    avg = [[]] + [[] for _ in runs]
    print("Accuracy")
    for lvl_name, lvl in enumerate([lvl1, lvl2, lvl3]):
        acc1 = []
        acc2 = []
        acc3 = []
        avg_acc = []
        for i, tape1 in enumerate(lvl[0].tapes):
            tape2 = lvl[1].tapes[i]
            tape3 = lvl[2].tapes[i]
            result1 = tape1["metadata"]["result"]
            result2 = tape2["metadata"]["result"]
            result3 = tape3["metadata"]["result"]
            tapes = [tape1, tape2, tape3]
            best_idx = majority_vote([result1, result2, result3])
            best_tape = tapes[best_idx]
            acc1.append(int(tape_correct(tape1)))
            acc2.append(int(tape_correct(tape2)))
            acc3.append(int(tape_correct(tape3)))
            avg_acc.append(int(tape_correct(best_tape)))
        print(f"L{lvl_name+1} {runs[0]}: {sum(acc1) / len(acc1):.3f} ({sum(acc1)} of {len(acc1)})")
        print(f"L{lvl_name+1} {runs[1]}: {sum(acc2) / len(acc2):.3f} ({sum(acc2)} of {len(acc2)})")
        print(f"L{lvl_name+1} {runs[2]}: {sum(acc3) / len(acc3):.3f} ({sum(acc3)} of {len(acc3)})")
        print(f"L{lvl_name+1} maj@3: {sum(avg_acc) / len(avg_acc):.3f} ({sum(avg_acc)} of {len(avg_acc)})")
        print()
        avg[0] += acc1
        avg[1] += acc2
        avg[2] += acc3
        avg[3] += avg_acc

    print(f"Avg. {runs[0]}: {sum(avg[0]) / len(avg[0]):.3f} ({sum(avg[0])} of {len(avg[0])})")
    print(f"Avg. {runs[1]}: {sum(avg[1]) / len(avg[1]):.3f} ({sum(avg[1])} of {len(avg[1])})")
    print(f"Avg. {runs[2]}: {sum(avg[2]) / len(avg[2]):.3f} ({sum(avg[2])} of {len(avg[2])})")
    print(f"Avg. maj@3: {sum(avg[3]) / len(avg[3]):.3f} ({sum(avg[3])} of {len(avg[3])})")


if __name__ == "__main__":
    main(
        root="../gaia/runs/",
        model="gpt-4o-mini-2024-07-18",
        runs=["gpt4o_mini_t02", "gpt4o_mini_t02_3", "gpt4o_mini_t05"],
    )


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
