import json
import logging
import sys

from examples.gaia_agent.scorer import question_scorer

from ..eval import load_dataset

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def validate_submission(submission_file: str):
    tasks = load_dataset("validation")
    answers = {task["task_id"]: task["Final answer"] for level_tasks in tasks.values() for task in level_tasks}
    model_answers = {}
    with open(submission_file) as f:
        for line in f:
            task = json.loads(line)
            model_answers[task["task_id"]] = task["model_answer"]
    accs = []
    for task_id, answer in answers.items():
        model_answer = model_answers[task_id]
        acc = int(question_scorer(model_answer, answer))
        accs.append(acc)
    print(f"\nSubmission accuracy: {sum(accs) / len(accs):.3f} ({sum(accs)} of {len(accs)})\n")


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: examples.gaia_agent.scripts.prepare_submission <exp_dir>"
    validate_submission(sys.argv[1])
