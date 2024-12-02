import json
import logging
import os
import sys

from tapeagents.io import save_json_tape

from ..eval import get_exp_config_dict, load_dataset, load_results
from ..tape import GaiaTape

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def main(exp_path: str) -> None:
    cfg = get_exp_config_dict(exp_path)
    tasks = load_dataset(cfg["split"])
    tapes_dir = os.path.join(exp_path, "tapes")
    os.makedirs(tapes_dir, exist_ok=True)
    with open(os.path.join(exp_path, "browser_log.jsonl"), "w") as wf:
        wf.write("")
    for level, level_tasks in tasks.items():
        outfile = os.path.join(exp_path, f"l{level}_{cfg['llm']['model_name']}_run.json")
        logger.info(f"Convert level {level} with {len(level_tasks)} from {outfile}")
        assert os.path.exists(outfile)
        results = load_results(outfile)
        logger.info(f"Loaded solutions for {len(results.tapes)} tasks")
        for i in range(len(level_tasks)):
            old_tape_dict = results.tapes[i] if i < len(results.tapes) else None
            if old_tape_dict is None:
                logger.info(f"Skipping task {i} as tape not found")
                continue
            old_tape = GaiaTape.model_validate(old_tape_dict)
            old_tape.metadata.level = level
            save_json_tape(old_tape, tapes_dir, f"l{level}_task{i}")
        logger.info(f"Converted {len(level_tasks)} tasks to tapes")
        with open(os.path.join(exp_path, "browser_log.jsonl"), "a") as wf:
            for k, v in results.web_cache.items():
                wf.write(json.dumps({"k": k, "v": v}) + "\n")
    logger.info("Done")


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: examples.gaia_agent.scripts.convert_legacy <exp_dir>"
    main(sys.argv[1])
