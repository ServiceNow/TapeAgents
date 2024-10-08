import logging
import os
import sys

import numpy as np
from datasets import load_dataset
from termcolor import colored
from tqdm import tqdm

from tapeagents.llms import TrainableLLM

from examples.rl_gsm8k.math_agent import (
    MathAgent,
    MathEnvironment,
    extract_result_value,
    save_tape,
    solve_task,
    MathAgentStep,
    Task,
    MathNodes,
    ALLOWED_STEPS,
    HINTS,
    START_TASK_GUIDANCE,
    SYSTEM_PROMPT,
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
        base_url="http://127.0.0.1:8080",
        model_name="/mnt/llmd/base_models/Meta-Llama-3.1-8B-Instruct",
        tokenizer_name="/mnt/llmd/base_models/Meta-Llama-3.1-8B-Instruct",
        parameters=dict(temperature=0.7),
    )

    agent = MathAgent.create(llm=llm,
                             nodes=MathNodes,
                             system_prompt=SYSTEM_PROMPT,
                             steps_prompt=ALLOWED_STEPS,
                             start_step_cls=Task,
                             agent_step_cls=MathAgentStep,
                             )
    env = MathEnvironment()

    tapes_dir = os.path.join(exp_path, "tapes")
    logger.info(f"Saving tapes to {tapes_dir}")
    os.makedirs(tapes_dir, exist_ok=True)
    os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(exp_path, "llm_calls.sqlite")

    solved = []
    for i, sample in enumerate(tqdm(samples)):
        sample = extract_result_value(sample)
        for j in range(attempts):
            tape_file = os.path.join(tapes_dir, f"task{i}_attempt{j+1}.json")
            tape = solve_task(agent, env, sample, tape_file)
            solved.append(int(tape.metadata.result["solved"]))
            save_tape(tape_file, tape)
        if i % 10 == 0 and i > 0:
            logger.info(f"{i}: Current accuracy: {np.mean(solved):.3f}, prompt tokens used: {agent.llm.token_count}")
    logger.info(f"Accuracy: {np.mean(solved):.3f}, prompt tokens used: {agent.llm.token_count}")
    logger.info(f"{len(solved)} tapes saved to {tapes_dir}")


if __name__ == "__main__":
    exp_path = sys.argv[1] if len(sys.argv) > 1 else "gsm8k/tuning/llama31_8b_train"
    main(exp_path)
