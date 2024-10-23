import logging
import os
import sys

import numpy as np
from datasets import load_dataset
from termcolor import colored
from tqdm import tqdm

from tapeagents.io import save_json_tape
from tapeagents.llms import TrainableLLM

from .math_agent import (
    MathAgent,
    MathEnvironment,
    extract_result_value,
    solve_task,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(exp_path: str, attempts: int = 1):
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    samples = [s for s in dataset]
    np.random.seed(42)
    np.random.shuffle(samples)  # type: ignore
    logging.info(f"Loaded {len(samples)} samples")

    llm = TrainableLLM(
        base_url="https://api.together.xyz",
        model_name="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        tokenizer_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
        parameters=dict(temperature=0.2),
    )
    agent = MathAgent.create(llm)
    env = MathEnvironment()

    tapes_dir = os.path.join(exp_path, "tapes")
    logger.info(f"Saving tapes to {tapes_dir}")
    os.makedirs(tapes_dir, exist_ok=True)
    os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(exp_path, "tapedata.sqlite")

    solved = []
    for i, sample in enumerate(tqdm(samples)):
        sample = extract_result_value(sample)
        for j in range(attempts):
            tape_file = os.path.join(tapes_dir, f"task{i}_attempt{j+1}.json")
            if os.path.exists(tape_file):
                logger.info(f"Task {i} attempt {j+1} already solved, skipping")
                continue
            try:
                tape = solve_task(agent, env, sample, tape_file)
                solved.append(int(tape.metadata.result["solved"]))
                save_json_tape(tape, tape_file)
            except Exception as e:
                logger.error(colored(f"Failed to solve task, attempt {j+1}: {e}", "red"))
                solved.append(0)
        if i % 10 == 0 and i > 0:
            logger.info(f"{i}: Current accuracy: {np.mean(solved):.3f}, prompt tokens used: {agent.llm.token_count}")
    logger.info(f"Accuracy: {np.mean(solved):.3f}, prompt tokens used: {agent.llm.token_count}")
    logger.info(f"{len(solved)} tapes saved to {tapes_dir}")


if __name__ == "__main__":
    exp_path = sys.argv[1] if len(sys.argv) > 1 else "gsm8k/tuning/llama31_70b_train"
    main(exp_path)
