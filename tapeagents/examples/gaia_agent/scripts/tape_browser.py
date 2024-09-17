import json
import logging
import os
import sys
from pathlib import Path

from tapeagents.examples.gaia_agent.eval import GaiaResults, load_results, tape_correct
from tapeagents.examples.gaia_agent.scripts.demo import GaiaRender
from tapeagents.tape_browser import TapeBrowser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


class GaiaTapeBrowser(TapeBrowser):
    def load_tapes(self, fname: str) -> list:
        fpath = os.path.join(self.tapes_folder, fname)
        assert os.path.exists(fpath), f"File {fpath} does not exist"
        try:
            self.results: GaiaResults = load_results(fpath)
            self.prompts = {p["prompt"]["id"]: p for p in self.results.prompts}
            logger.info(f"Loaded {len(self.results.tapes)} tapes, {len(self.prompts)} prompts from {fpath}")
        except Exception as e:
            logger.error(f"Failed to load tapes from {fpath}: {e}")
            self.results = GaiaResults()
        return self.results.tapes

    def get_steps(self, tape) -> list:
        return tape["steps"]

    def get_context(self, tape) -> list:
        return []

    def get_file_label(self, filename: str, tapes: list) -> str:
        acc, n_solved = self.results.accuracy
        return f"<h2>Accuracy {acc:.2f}%, {n_solved} out of {len(tapes)}</h2>"

    def get_tape_name(self, i: int, tape: dict) -> str:
        mark = ("+" if tape_correct(tape) else "") + ("[f]" if tape["metadata"]["task"]["file_name"] else "")
        return f'{mark}L{tape["metadata"]["task"]["Level"]}{i+1}: {tape["steps"][0]["content"][:32]}'

    def get_tape_label(self, tape: dict) -> str:
        tape_prompts = [s for s in tape["steps"] if s.get("prompt_id") in self.prompts]
        failure_count = len([s for s in tape["steps"] if s["kind"].endswith("failure")])
        label = f"""<h2>Tape Result</h2>
            <div class="result-label expected">Golden Answer: <b>{tape["metadata"]["task"]['Final answer']}</b></div>
            <div class="result-label">Agent Answer: <b>{tape["metadata"]["result"]}</b></div>
            <div class="result-label">Steps: {len(tape["steps"])}</div>
            <div class="result-label">Failures: {failure_count}</div>"""
        success = tape["steps"][-1].get("success", "")
        overview = tape["steps"][-1].get("overview", "")
        label += f"""
            <div class="result-success">Finished successfully: {success}</div>
            <div>LLM Calls: {len(tape_prompts)}</div>
            <div class="result-overview">Overview:<br>{overview}</div>"""
        return label

    def get_tape_files(self) -> list[str]:
        files = sorted(
            [
                f for f in os.listdir(self.tapes_folder) if f.endswith(self.file_extension)
            ]  # for recursive use: Path(self.tapes_folder).glob(f"**/*{self.file_extension}")
        )
        assert files, f"No files found in {self.tapes_folder}"
        logger.info(f"Found {len(files)} files")
        return sorted(files)


def main(dirname: str):
    renderer = GaiaRender()
    browser = GaiaTapeBrowser(load_results, dirname, renderer, file_extension=".json")
    browser.launch()


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python -m tapeagents.examples.gaia_agent.scripts.tape_browser <dirname>"
    main(sys.argv[1])
