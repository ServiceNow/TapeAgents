import logging
import os
import sys
from collections import defaultdict

from tapeagents.core import Action
from tapeagents.io import load_tapes
from tapeagents.observe import retrieve_all_llm_calls
from tapeagents.renderers.camera_ready_renderer import CameraReadyRenderer
from tapeagents.tape_browser import TapeBrowser

from ..eval import calculate_accuracy, get_exp_config_dict, tape_correct
from ..tape import GaiaTape

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class GaiaTapeBrowser(TapeBrowser):
    def __init__(self, tapes_folder: str, renderer):
        super().__init__(tape_cls=GaiaTape, tapes_folder=tapes_folder, renderer=renderer, file_extension=".json")

    def load_tapes(self, name: str) -> list:
        _, fname, postfix = name.split("/", maxsplit=2)
        tapes_path = os.path.join(self.tapes_folder, fname, "tapes")
        try:
            all_tapes: list[GaiaTape] = load_tapes(GaiaTape, tapes_path, file_extension=".json")  # type: ignore
        except Exception as e:
            logger.error(f"Failed to load tapes from {tapes_path}: {e}")
            return []
        tapes = []
        for tape in all_tapes:
            if postfix == "all" or str(tape.metadata.level) == postfix:
                tapes.append(tape)
        self.llm_calls = {}
        sqlite_fpath = os.path.join(self.tapes_folder, fname, "tapedata.sqlite")
        if not os.path.exists(sqlite_fpath):
            sqlite_fpath = os.path.join(self.tapes_folder, fname, "llm_calls.sqlite")
        try:
            llm_calls = retrieve_all_llm_calls(sqlite_fpath)
            self.llm_calls = {llm_call.prompt.id: llm_call for llm_call in llm_calls}
        except Exception as e:
            logger.error(f"Failed to load LLM calls from {sqlite_fpath}: {e}")
        logger.info(f"Loaded {len(tapes)} tapes from {tapes_path}")
        logger.info(f"Loaded {len(self.llm_calls)} prompts from {sqlite_fpath}")
        return tapes

    def load_llm_calls(self):
        pass

    def update_tape_view(self, tape_id: int) -> tuple[str, str]:
        try:
            tape: GaiaTape = self.tapes[tape_id]  # type: ignore
        except IndexError:
            return "", "Tape not found"
        label = self.get_tape_label(tape)
        html = f"{self.renderer.style}<style>.thought {{ background-color: #ffffba !important; }};</style>{self.renderer.render_tape(tape, self.llm_calls)}"
        return html, label

    def get_file_label(self, filename: str, tapes: list[GaiaTape]) -> str:
        acc, n_solved = calculate_accuracy(tapes)
        errors = defaultdict(int)
        tokens_num = 0
        for tape in tapes:
            if tape.metadata.error:
                errors["fatal"] += 1
            last_action = None
            for step in tape:
                if isinstance(step, Action):
                    last_action = step
                prompt_id = step.metadata.prompt_id
                if prompt_id and prompt_id in self.llm_calls:
                    tokens_num += (
                        self.llm_calls[prompt_id].prompt_length_tokens + self.llm_calls[prompt_id].output_length_tokens
                    )
                if step.kind == "page_observation" and step.error:
                    errors["browser"] += 1
                elif step.kind == "llm_output_parsing_failure_action":
                    errors["parsing"] += 1
                elif step.kind == "action_execution_failure":
                    errors[f"{last_action.kind}"] += 1
        html = f"<h2>Accuracy {acc:.2f}%, {n_solved} out of {len(tapes)}</h2>LLM tokens spent: {tokens_num}"
        if errors:
            errors_str = "<br>".join(f"{k}: {v}" for k, v in errors.items())
            html += f"<h2>Errors</h2>{errors_str}"
        return html

    def get_tape_name(self, i: int, tape: GaiaTape) -> str:
        error = "F" if tape.metadata.error else None
        last_action = None
        for step in tape:
            if isinstance(step, Action):
                last_action = step
            if step.kind == "page_observation" and step.error:
                error = "br"
            elif step.kind == "llm_output_parsing_failure_action":
                error = "pa"
            elif step.kind == "action_execution_failure" and last_action:
                error = last_action.kind[:2]
        mark = "+" if tape_correct(tape) else ("" if tape.metadata.result else "∅")
        if tape.metadata.task.get("file_name"):
            mark += "📁"
        if error:
            mark += f"[{error}]"
        if mark:
            mark += " "
        return f"{i+1} {mark}{tape[0].content[:32]}"  # type: ignore

    def get_tape_label(self, tape: GaiaTape) -> str:
        llm_calls_num = 0
        tokens_num = 0

        for step in tape:
            prompt_id = step.metadata.prompt_id
            if prompt_id:
                llm_calls_num += 1
                if prompt_id in self.llm_calls:
                    tokens_num += (
                        self.llm_calls[prompt_id].prompt_length_tokens + self.llm_calls[prompt_id].output_length_tokens
                    )
        failure_count = len(
            [step for step in tape if "failure" in step.kind or (step.kind == "page_observation" and step.error)]
        )
        label = f"""<h2>Tape Result</h2>
            <div class="result-label expected">Golden Answer: <b>{tape.metadata.task.get('Final answer', '')}</b></div>
            <div class="result-label">Agent Answer: <b>{tape.metadata.result}</b></div>
            <div class="result-label">Steps: {len(tape)}</div>
            <div class="result-label">Failures: {failure_count}</div>"""
        success = tape[-1].success if hasattr(tape[-1], "success") else ""  # type: ignore
        overview = tape[-1].overview if hasattr(tape[-1], "overview") else ""  # type: ignore
        label += f"""
            <div class="result-success">Finished successfully: {success}</div>
            <div>LLM Calls: {llm_calls_num}, tokens: {tokens_num}</div>
            <div class="result-overview">Overview:<br>{overview}</div>"""
        if tape.metadata.error:
            label += f"<div class='result-error'><b>Error</b>: {tape.metadata.error}</div>"
        return label

    def get_tape_files(self) -> list[str]:
        raw_exps = [
            d for d in os.listdir(self.tapes_folder) if os.path.isdir(os.path.join(self.tapes_folder, d, "tapes"))
        ]
        assert raw_exps, f"No experiments found in {self.tapes_folder}"
        logger.info(f"Found {len(raw_exps)} experiments")
        exps = []
        for postfix in ["1", "2", "3", "all"]:
            for r in raw_exps:
                exp_dir = os.path.join(self.tapes_folder, r)
                cfg = get_exp_config_dict(exp_dir)
                parts = cfg["data_dir"].split("/")
                set_name = parts[-2] if cfg["data_dir"].endswith("/") else parts[-1]
                exps.append(f"{set_name}/{r}/{postfix}")
        return sorted(exps)


def main(dirname: str):
    browser = GaiaTapeBrowser(dirname, CameraReadyRenderer(fold_length=3000))
    browser.launch(port=7861)


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python -m scripts.tape_browser <dirname>"
    main(sys.argv[1])
