import json
import logging
import os
import sys

from tqdm import tqdm

from tapeagents.core import TrainingText
from tapeagents.finetune.data import load_samples, save_samples
from tapeagents.finetune.finetune import load_config, run_finetuning_loop
from tapeagents.llms import TrainableLLM

from .math_agent import ActionExecutionFailure, MathAgent, MathTape

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_training_samples_from_tapes(tapes_path: str) -> list[TrainingText]:
    """
    Make training samples from tapes that were solved successfully,
    does not contain action execution failures, and do not have repeated steps.
    """

    # We need the agent to cut tapes into training samples
    agent = MathAgent.create(
        TrainableLLM(
            base_url="https://api.together.xyz",
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            tokenizer_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        )
    )

    training_samples: list[TrainingText] = []
    failures = 0
    not_solved = 0
    with_duplicates = 0
    tape_files = sorted([os.path.join(tapes_path, f) for f in os.listdir(tapes_path) if f.endswith(".json")])
    for tape_file in tqdm(tape_files):
        with open(tape_file) as f:
            tape_dict = json.load(f)
        tape = MathTape.model_validate(tape_dict)
        step_types = set(type(step) for step in tape)

        # detect repeated steps
        last_step_view = None
        duplicate = False
        for step in tape:
            view = step.llm_view()
            if view == last_step_view:
                duplicate = True
                break
            last_step_view = view

        if duplicate:
            with_duplicates += 1
            continue
        if ActionExecutionFailure in step_types:
            failures += 1
            continue
        if not tape.metadata.result["solved"]:
            not_solved += 1
            continue

        for sample in agent.make_training_data(tape):
            training_samples.append(sample)
    logger.info(f"Skipped failures {failures}, not solved {not_solved}, with duplicates {with_duplicates}")
    logger.info(f"Created positive training samples: {len(training_samples)}")
    return training_samples


def main(exp_path: str):
    train_samples_file = "gsm8k/tuning/llama31_70b_train_t02/training_samples_3k.jsonl"
    if os.path.exists(train_samples_file):
        training_samples = load_samples(train_samples_file)
    else:
        tapes_path = "gsm8k/tuning/llama31_70b_train_t02/tapes"
        training_samples = get_training_samples_from_tapes(tapes_path)
        save_samples(training_samples, train_samples_file)

    cfg = load_config("llama31_8b", output_dir=exp_path)
    run_finetuning_loop(cfg=cfg, training_samples=training_samples)


if __name__ == "__main__":
    exp_path = sys.argv[1] if len(sys.argv) > 1 else "gsm8k/tuning/llama31_70b_train_t02/tune_llama31_8b_1"
    main(exp_path)
