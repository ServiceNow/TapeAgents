import json
import logging
import sys

import numpy as np
from datasets import load_dataset
from termcolor import colored
from tqdm import tqdm

from tapeagents.llms import TrainableLLM

from .math_agent import (
    MathAgent,
    MathEnvironment,
    extract_result_value,
    solve_task,
)

logger = logging.getLogger(__name__)

env = MathEnvironment()


def eval(tested_agent, test_set, name="") -> float:
    test_solved = []
    n = 0
    for sample in tqdm(test_set):
        sample = extract_result_value(sample)
        try:
            tape = solve_task(tested_agent, env, sample)
            test_solved.append(int(tape.metadata.result["solved"]))
        except Exception as e:
            logger.error(colored("Failed to solve task: {e}", "red"))
            test_solved.append(0)
            raise e
        acc = np.mean(test_solved).item()
        n = len(test_solved)
        if n % 10 == 0 and n > 0:
            logger.info(f"{n}: Current accuracy: {acc:.3f}")
            with open("results.jsonl", "a") as f:
                f.write(json.dumps({name: acc, "n": n}) + "\n")
    acc = np.mean(test_solved).item()
    with open("results.jsonl", "a") as f:
        f.write(json.dumps({name: acc, "n": n}) + "\n")
    return acc


def main(student_path: str):
    test_split = list(load_dataset("openai/gsm8k", "main", split="test"))
    np.random.seed(42)
    np.random.shuffle(test_split)  # type: ignore
    test_set = test_split[:200]

    # to run inference: vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct
    untuned_agent = MathAgent.create(
        TrainableLLM(base_url="http://localhost:8000", model_name="meta-llama/Meta-Llama-3.1-8B-Instruct")
    )

    acc = eval(untuned_agent, test_set)
    logger.info(f"Untuned test accuracy: {acc:.3f}")

    # to run inference: vllm serve <student_path> --port 8001
    tuned_agent = MathAgent.create(TrainableLLM(base_url="http://localhost:8001", model_name=student_path))

    tuned_acc = eval(tuned_agent, test_set, "tuned_acc")
    logger.info(f"Tuned test accuracy: {tuned_acc:.3f}")

    # check teacher model
    big_agent = MathAgent.create(
        TrainableLLM(
            base_url="https://api.together.xyz",
            model_name="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            tokenizer_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
        )
    )

    big_acc = eval(big_agent, test_set, "big_acc")
    logger.info(f"Teacher test accuracy: {big_acc:.3f}")


if __name__ == "__main__":
    student_path = sys.argv[1] if len(sys.argv) > 1 else "gsm8k/tuning/llama31_70b_train_t02/tune1/intermediate/800/"
    main(student_path)
