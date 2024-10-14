import json
import os
import sys

from examples.gaia_agent.eval import load_results

TEST_SET_DIR = "../gaia/dataset/test/"


def main(exp_dir: str):
    assert os.path.isdir(exp_dir), f"Directory {exp_dir} does not exist or is not a directory"
    submission_file = os.path.join(exp_dir, "submission.jsonl")
    submission = []
    for level in range(3):
        level_file = None
        for fname in os.listdir(exp_dir):
            if fname.startswith(f"l{level+1}_") and fname.endswith("_run.json"):
                level_file = os.path.join(exp_dir, fname)
                break
        assert level_file is not None, f"Missing results for level {level+1}"
        level_results = load_results(level_file)
        unsolved = 0
        for tape in level_results.tapes:
            task = tape["metadata"]["task"]
            last_step = tape["steps"][-1]
            model_answer = last_step["answer"] if last_step["kind"] == "gaia_answer_action" else None
            if not model_answer:
                unsolved += 1
            line = {"task_id": task["task_id"], "model_answer": str(model_answer)}
            submission.append(line)
        print(f"Level {level+1}: gathered {len(level_results.tapes)} task solutions, {unsolved} unsolved")

    print(f"Writing submission with {len(submission)} tasks to {submission_file}")
    with open(submission_file, "w") as f:
        for line in submission:
            f.write(f"{json.dumps(line)}\n")


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: examples.gaia_agent.scripts.prepare_submission <exp_dir>"
    main(sys.argv[1])
