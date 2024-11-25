import os
import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from examples.gaia_agent.tape import GaiaTape
from examples.gaia_agent.v2 import GaiaPlanner
from tapeagents.annotation import Annotation
from tapeagents.core import Tape
from tapeagents.io import load_tapes, save_json_tape
from tapeagents.llms import TrainableLLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(
    version_base=None,
    config_path="../../../conf",
    config_name="gaia_openai",
)
def main(cfg: DictConfig) -> None:
    """Run Gaia agent on tape prefixs that humans annotated."""
    os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(cfg.exp_path, "tapedata.sqlite")

    llm: TrainableLLM = instantiate(cfg.llm)
    agent = GaiaPlanner.create(llm)
    annotator_tapes = load_tapes(Tape[GaiaTape, Annotation], cfg.annotations)
    input_tapes = [tape.context[:tape.steps[0].step] for tape in annotator_tapes]
    logger.info(f"Loaded {len(input_tapes)} tapes from {cfg.annotations}")
    output_tapes_dir = os.path.join(cfg.exp_path, "tapes")
    output_tapes = [agent.run(tape).get_final_tape() for tape in input_tapes]

    os.makedirs(output_tapes_dir, exist_ok=True)
    for i, tape in enumerate(output_tapes):
        save_json_tape(tape, output_tapes_dir, str(i))
    logger.info(f"Saved {len(output_tapes)} forks to {output_tapes_dir}")

if __name__ == "__main__":
    main()