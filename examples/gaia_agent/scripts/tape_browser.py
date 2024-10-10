import logging
import os
import sys
from pathlib import Path

from tapeagents.core import LLMCall
from tapeagents.rendering import GuidedAgentRender
from tapeagents.tape_browser import TapeBrowser

from ..eval import GaiaResults, load_results, tape_correct

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class GaiaTapeBrowser(TapeBrowser):
    def __init__(self, tapes_folder: str, renderer):
        super().__init__(tapes_folder=tapes_folder, renderer=renderer, tape_cls=GaiaResults, file_extension=".json")

    def load_tapes(self, fname: str) -> list:
        fpath = os.path.join(self.tapes_folder, fname)
        assert os.path.exists(fpath), f"File {fpath} does not exist"
        try:
            self.results: GaiaResults = load_results(fpath)
            for i, prompt in enumerate(self.results.prompts):
                if "completion" in prompt:
                    # legacy compatibility
                    prompt["output"] = prompt.pop("completion")
                    prompt["output_length_tokens"] = prompt.pop("completion_length_tokens")
                self.results.prompts[i] = prompt
            self.llm_calls = {p["prompt"]["id"]: LLMCall.model_validate(p) for p in self.results.prompts}
            logger.info(f"Loaded {len(self.results.tapes)} tapes, {len(self.llm_calls)} LLM calls from {fpath}")
        except Exception as e:
            logger.error(f"Failed to load tapes from {fpath}: {e}")
            self.results = GaiaResults()
        return self.results.tapes

    def load_llm_calls(self):
        pass

    def update_tape_view(self, tape_id: int) -> tuple[str, str]:
        logger.info(f"Loading tape {tape_id}")
        tape = self.tapes[tape_id]
        label = self.get_tape_label(tape)
        steps = self.get_steps(tape)
        step_views = []
        last_prompt_id = None
        for i, step in enumerate(steps):
            prompt_id = step.get("metadata", {}).get("prompt_id", step.pop("prompt_id", None))
            view = self.renderer.render_step(step, i)  # type: ignore
            if prompt_id in self.llm_calls and prompt_id != last_prompt_id:
                prompt_view = self.renderer.render_llm_call(self.llm_calls[prompt_id])
                view = prompt_view + view
            step_views.append(view)
            last_prompt_id = prompt_id
        steps_html = "".join(step_views)
        html = f"{self.renderer.style}"
        html += f"{self.renderer.steps_header}{steps_html}"
        return html, label

    def get_steps(self, tape) -> list:
        return tape["steps"]  # type: ignore

    def get_context(self, tape) -> list:
        return []

    def get_file_label(self, filename: str, tapes: list) -> str:
        acc, n_solved = self.results.accuracy
        parsing_errors = 0
        page_errors = 0
        tokens_num = 0
        for tape in self.results.tapes:
            for step in tape["steps"]:
                prompt_id = step.get("metadata", {}).get("prompt_id", step.get("prompt_id"))
                if prompt_id and prompt_id in self.llm_calls:
                    tokens_num += (
                        self.llm_calls[prompt_id].prompt_length_tokens + self.llm_calls[prompt_id].output_length_tokens
                    )
                if step.get("kind") == "page_observation":
                    if step.get("error"):
                        page_errors += 1
                if step.get("kind") == "agent_response_parsing_failure_action":
                    parsing_errors += 1
        return f"<h2>Accuracy {acc:.2f}%, {n_solved} out of {len(tapes)}</h2>LLM tokens spent: {tokens_num}<br>Step parsing errors: {parsing_errors}<br>Page loading errors: {page_errors}"

    def get_tape_name(self, i: int, tape: dict) -> str:
        page_error = False
        parsing_error = False
        for step in tape["steps"]:
            if step.get("kind") == "page_observation" and step.get("error"):
                page_error = True
            if step.get("kind") == "agent_response_parsing_failure_action":
                parsing_error = True
        mark = "+" if tape_correct(tape) else ""
        if tape["metadata"]["task"]["file_name"]:
            mark += "f"
        if parsing_error:
            mark += "E"
        if page_error:
            mark += "e"
        if mark:
            mark += "|"
        return f'{mark}{i+1}: {tape["steps"][0]["content"][:32]}'

    def get_tape_label(self, tape: dict) -> str:
        llm_calls_num = 0
        tokens_num = 0

        for step in tape["steps"]:
            prompt_id = step.get("metadata", {}).get("prompt_id", step.get("prompt_id"))
            if prompt_id:
                llm_calls_num += 1
                tokens_num += (
                    self.llm_calls[prompt_id].prompt_length_tokens + self.llm_calls[prompt_id].output_length_tokens
                )
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
            <div>LLM Calls: {llm_calls_num}, tokens: {tokens_num}</div>
            <div class="result-overview">Overview:<br>{overview}</div>"""
        return label

    def get_tape_files(self) -> list[str]:
        files = sorted(
            # for 1-level use: [f for f in os.listdir(self.tapes_folder) if f.endswith(self.file_extension)]
            [str(p.relative_to(self.tapes_folder)) for p in Path(self.tapes_folder).glob(f"**/*{self.file_extension}")]
        )
        assert files, f"No files found in {self.tapes_folder}"
        logger.info(f"Found {len(files)} files")
        return sorted(files)


def main(dirname: str):
    renderer = GuidedAgentRender()
    browser = GaiaTapeBrowser(dirname, renderer)
    browser.launch(port=7861)


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python -m scripts.tape_browser <dirname>"
    main(sys.argv[1])
