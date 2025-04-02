import logging
import os
import sys
from pathlib import Path

import transformers
from omegaconf import DictConfig

from tapeagents.finetune.data import load_samples
from tapeagents.io import load_tapes
from tapeagents.llms import LLM, LLMCall, ReplayLLM, TrainableLLM
from tapeagents.orchestrator import replay_tapes

sys.path.append(str(Path(__file__).parent.parent.resolve()))  # allow to import from examples

from examples.gsm8k_tuning.finetune_student import get_training_samples_from_tapes
from examples.gsm8k_tuning.math_agent import MathAgent, MathTape
from examples.rl_gsm8k.orchestrate_rl import (
    CoTMathAgent,
    RLMathTape,
    extract_tape_training_samples,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

res_path = Path(__file__).parent.resolve() / "res"


def mock_llm(run_dir: str) -> LLM:
    llama = TrainableLLM(
        base_url="https://api.together.xyz",
        model_name="meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
        tokenizer_name="meta-llama/Meta-Llama-3-70B-Instruct",
        parameters=dict(temperature=0.7, max_tokens=512),
    )
    return ReplayLLM.from_llm(llama, run_dir)


def test_gsm8k_tuning_tapes_generation():
    run_dir = f"{res_path}/gsm8k_tuning"
    llm = mock_llm(run_dir)
    agent = MathAgent.create(llm)
    tapes = load_tapes(MathTape, os.path.join(run_dir, "tapes"), file_extension=".json")
    logger.info(f"Validate {len(tapes)} tapes")
    fails = replay_tapes(agent, tapes, reuse_observations=True)
    assert fails == 0, f"{fails} failed tapes"


def test_gsm8k_tuning_samples_prep():
    run_dir = f"{res_path}/gsm8k_tuning"
    training_samples = load_samples(f"{run_dir}/training_samples.jsonl")
    new_training_samples = get_training_samples_from_tapes(f"{run_dir}/tapes/")
    assert training_samples == new_training_samples


def test_rl_gsm8k_data():
    run_dir = f"{res_path}/rl_gsm8k"
    tapes = load_tapes(RLMathTape, run_dir, file_extension=".json")
    llm = mock_llm(run_dir)
    llm.tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    agent = CoTMathAgent.create(system_prompt="", llm=llm, max_prompt_length=1024)
    cfg = DictConfig(
        {"dataset_name": "math", "llm": {"parameters": {"max_tokens": 2048}}, "finetune": {"seq_length": 2048}}
    )
    training_samples = []
    for tape in tapes:
        for step in tape:
            if llm_call_data := step.metadata.other.get("llm_call"):
                step.metadata.other["llm_call"] = LLMCall(**llm_call_data)
        training_sample, _ = extract_tape_training_samples(tape, agent, "train", cfg)
        training_samples.append(training_sample[0])
    new_training_samples = load_samples(f"{run_dir}/training_samples.jsonl")
    assert training_samples == new_training_samples


if __name__ == "__main__":
    test_gsm8k_tuning_tapes_generation()
    test_gsm8k_tuning_samples_prep()
    test_rl_gsm8k_data()
