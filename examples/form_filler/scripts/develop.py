from pathlib import Path
import sys
import logging

import hydra
from omegaconf import DictConfig

from tapeagents.io import load_tapes
from tapeagents.studio import Studio
from tapeagents.llms import LLAMA, LLM
from tapeagents.rendering import PrettyRenderer

from examples.form_filler.tape import FormFillerTape
from examples.form_filler.critic import (
    Critic,
    CriticExpert,
    CriticTape,
    HelpfulnessExpert,
    TransparencyExpert,
    GroundednessExpert,
    AccuracyExpert,
    ResponsivenessExpert
)
from tapeagents.chain import Chain

logging.basicConfig(level=logging.INFO)

   
def develop_critic(cfg: DictConfig, llm: LLM):
    # path = Path(__file__).parent.parent / "assets/agent_forks.yaml"
    # tapes: list[FormFillerTape] = load_tapes(FormFillerTape, path)
    tapes: list[CriticTape] = load_tapes(CriticTape, "/mnt/llmd/data/gontiern/tapes/tmp_annotations.yaml")

    # select random FormFillerTape
    tape = tapes[0].context

    # agent = Chain.create(
    #     name="critique",
    #     subagents=[
    #         GroundednessExpert.create(llm, templates=cfg.critique.is_grounded_templates),
    #         HelpfulnessExpert.create(llm, templates=cfg.critique.is_helpful_templates),
    #         ResponsivenessExpert.create(llm, templates=cfg.critique.is_responsive_templates),
    #         AccuracyExpert.create(llm, templates=cfg.critique.is_accurate_templates),
    #         TransparencyExpert.create(llm, templates=cfg.critique.is_transparent1_templates, mode="i_should", name="transparency1_expert"),
    #         TransparencyExpert.create(llm, templates=cfg.critique.is_transparent2_templates, mode="i_note", name="transparency2_expert"),
    #     ]
    # )
    agent = CriticExpert.create(llm, templates=cfg.critique)
    Studio(agent, CriticTape(context=tape), PrettyRenderer()).launch()


@hydra.main(config_path="../conf", config_name="base")
def main(cfg: DictConfig):
    llm = LLAMA(
        base_url="https://snow-research-tapes-vllm_llama405b.job.toolkit-sp.yul201.service-now.com",
        model_name="meta-llama/Meta-Llama-3.1-405B-Instruct-FP8",
        parameters=dict(
            temperature=0,
            max_tokens=3
        )
    )
    if cfg.agent == "critic":
        develop_critic(cfg, llm)
    else:
        raise ValueError(f"Unknown agent type: {cfg.agent}")
    
if __name__ == "__main__":
    main()    