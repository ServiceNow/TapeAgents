from collections import Counter
import json
from pathlib import Path
import logging
from typing import Tuple

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from tapeagents.io import load_tapes, save_tapes
from tapeagents.llms import LiteLLM
from tapeagents.parallel_processing import lazy_thread_pool_processor

from examples.form_filler.critic import (
    Critic, CriticExpert, CriticStep, CriticTape,
    IsGrounded,
    IsHelpful,
    IsResponsive,
    IsAccurate,
    IsTransparent1,
    IsTransparent2,
)
from examples.form_filler.tape import FormFillerTape

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def get_labels(steps: list[CriticStep]) -> dict:
    labels = {}
    for step in steps:
        if isinstance(step, IsGrounded):
            labels["grounded"] = step.grounded
        elif isinstance(step, IsHelpful):
            labels["helpful"] = step.helpful
        elif isinstance(step, IsResponsive):
            labels["responsive"] = step.responsive
        elif isinstance(step, IsAccurate):
            labels["accurate"] = step.accurate
        elif isinstance(step, IsTransparent1):
            labels["transparent1"] = step.transparent1
        elif isinstance(step, IsTransparent2):
            labels["transparent2"] = step.transparent2
        # else:
        #     logger.info(f"Unknown step type: {step}")
    return labels


def annotate_formfiller_tape(tape: FormFillerTape, agent: Critic) -> CriticTape:
    predicted_tape = CriticTape(context=tape)
    for event in agent.run(predicted_tape):
        if event.step:
            predicted_tape = predicted_tape.append(event.step)
    return predicted_tape


def annotate_groundtruth_tape(groundtruth_tape: CriticTape, agent: Critic) -> Tuple[CriticTape, CriticTape]:
    assert isinstance(groundtruth_tape.context, FormFillerTape)
    predicted_tape = annotate_formfiller_tape(groundtruth_tape.context, agent)
    return groundtruth_tape, predicted_tape


@hydra.main(config_path="../conf", config_name="critique")
def main(cfg: DictConfig):
    llm = LiteLLM(
        # base_url="https://api.together.xyz",
        # model_name="meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
        # base_url="https://snow_infiniband-pyllmd-meta_llama_3_1_70b_instruct.job.console.elementai.com",
        # model_name="/mnt/llmd/base_models/Meta-Llama-3.1-70B-Instruct",
        base_url="https://snow-research-tapes-vllm_llama405b.job.toolkit-sp.yul201.service-now.com",
        model_name="meta-llama/Meta-Llama-3.1-405B-Instruct-FP8",
        tokenizer_name="meta-llama/Meta-Llama-3.1-405B-Instruct-FP8",
        parameters=dict(
            temperature=0,
            max_tokens=3
        )
    )
    src = cfg.groundtruth_tapes_path  # "/mnt/llmd/data/gontiern/tapes/tmp_annotations.yaml"
    dst = cfg.predicted_tapes_path  # "/mnt/llmd/data/gontiern/tapes/tmp_annotations_predicted.yaml"
    # src = "/mnt/llmd/data/gontiern/tapes/CoffeeCorpGoldV2_annotations.yaml"
    # dst = "/mnt/llmd/data/gontiern/tapes/CoffeeCorpGoldV2_annotations_predicted.yaml"

    tapes = load_tapes(CriticTape, src)
    agent = CriticExpert.create(llm, templates=cfg.critique)

    def tape_gen():
        for tape in tapes:
            yield tape

    counts = Counter()
    totals = Counter()
    predicted_tapes: list[CriticTape] = []
    for result in lazy_thread_pool_processor(
            stream=tape_gen(),
            worker_func=lambda tape: annotate_groundtruth_tape(tape, agent),
            n_workers=cfg.n_workers,
    ):
        if isinstance(result, Exception):
            logger.error(f"Error: {result}")
            continue
        else:
            original_tape, predicted_tape = result

        gold_labels = get_labels(original_tape.steps)

        predicted_tapes.append(predicted_tape)
        predicted_labels = get_labels(predicted_tape.steps)
        # logger.info("PREDICTED:", colored(predicted_labels, "red"))
        # logger.info("TARGET:", colored(gold_labels, "red"))

        # make sure the gold labels are included in the predited labels
        for key in gold_labels:
            totals[key] += 1
            if gold_labels[key] == predicted_labels.get(key, ""):
                counts[key] += 1

    for key in counts:
        logger.info(f"{key}: {counts[key]} / {totals[key]} = {counts[key] / totals[key] * 100:.2f}%")

    metrics = {
        key: {
            "total": totals[key],
            "counts": counts[key],
            "score": counts[key] / totals[key]
        }
        for key in counts
    }
    with open(dst.replace(".yaml", "_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with save_tapes(Path(dst)) as saver: 
        for tape in tqdm(predicted_tapes, desc="Saving tapes"):
            saver.save(tape)

if __name__ == "__main__":
    main()    