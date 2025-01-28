import json
import logging
import os
import sys

from pydantic import BaseModel

from tapeagents.core import Step
from tapeagents.observe import retrieve_all_llm_calls
from tapeagents.renderers.camera_ready_renderer import CameraReadyRenderer
from tapeagents.tape_browser import TapeBrowser

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class WorkarenaTapeBrowser(TapeBrowser):
    def __init__(self, tapes_folder: str, renderer):
        super().__init__(tapes_folder=tapes_folder, renderer=renderer, tape_cls=BaseModel, file_extension=".json")

    def load_tapes(self, fname: str) -> list:
        fpath = os.path.join(self.tapes_folder, fname, "tapes")
        assert os.path.exists(fpath), f"Path {fpath} does not exist"
        assert os.path.isdir(fpath)
        tapes = []
        paths = [os.path.join(fpath, tape_name) for tape_name in os.listdir(fpath)]
        for tape_path in paths:
            tape_name = os.path.basename(tape_path)
            tape_path = os.path.join(fpath, tape_name)
            with open(tape_path) as f:
                tape_dict = json.load(f)
                tape_dict["metadata"]["tape_dir"] = fname
                for i, step_dict in enumerate(tape_dict["steps"]):
                    screenshot_path = ""
                    if "screenshot_path" in step_dict:  # deprecated format
                        screenshot_path = step_dict["screenshot_path"]
                    if "screenshot_path" in step_dict.get("metadata", {}).get("other", {}):
                        screenshot_path = step_dict["metadata"]["other"]["screenshot_path"]
                    if screenshot_path:
                        if "metadata" not in step_dict:
                            step_dict["metadata"] = {"other": {}}
                        full_path = os.path.join(fname, "screenshots", screenshot_path)
                        step_dict["metadata"]["other"]["screenshot_path"] = full_path
                    tape_dict["steps"][i] = step_dict
            tapes.append(tape_dict)
        self.llm_calls = {}
        base_path = os.path.dirname(fpath)
        sqlite_fpath = os.path.join(base_path, "tapedata.sqlite")
        if not os.path.exists(sqlite_fpath):
            sqlite_fpath = os.path.join(base_path, "llm_calls.sqlite")
        try:
            llm_calls = retrieve_all_llm_calls(sqlite_fpath)
        except Exception as e:
            logger.error(f"Failed to load LLM calls from {sqlite_fpath}: {e}")
            llm_calls = []
        if os.path.exists(sqlite_fpath):
            self.llm_calls = {llm_call.prompt.id: llm_call for llm_call in llm_calls}
        logger.info(f"Loaded {len(tapes)} tapes, {len(self.llm_calls)} prompts from {fpath}")
        try:
            tapes.sort(key=lambda tape: tape["metadata"]["result"]["number"])
        except Exception:
            logger.warning("Failed to sort tapes by number")
        return tapes

    def get_steps(self, tape) -> list:
        return tape["steps"]

    def get_context(self, tape) -> list:
        return []

    def load_llm_calls(self):
        pass

    def get_exp_label(self, filename: str, tapes: list) -> str:
        acc = []
        tokens_num = 0
        failure_count = 0
        for tape in tapes:
            if "result" not in tape["metadata"]:
                return "<h2>{filename}</h2"
            result = tape["metadata"]["result"]
            if isinstance(result, bool):
                acc.append(int(result))
            else:
                if "success" not in result:
                    continue
                acc.append(int(result["success"]))
            for step in tape["steps"]:
                prompt_id = step.get("metadata", {}).get("prompt_id", step.pop("prompt_id", None))
                if prompt_id and prompt_id in self.llm_calls:
                    tokens_num += (
                        self.llm_calls[prompt_id].prompt_length_tokens + self.llm_calls[prompt_id].output_length_tokens
                    )
            failure_count += len([s for s in tape["steps"] if s["kind"].endswith("failure")])
        success_rate = (sum(acc) / len(acc) if acc else 0) * 100
        return f"<h2>{filename}<br>Success {success_rate:.2f}% ({sum(acc)} out of {len(acc)})</h2><h3>Tokens spent: {tokens_num}</h3><h3>Failures: {failure_count}</h3>"

    def get_tape_name(self, i: int, tape: dict) -> str:
        result = tape["metadata"].get("result")
        if not result or not isinstance(result, dict):
            return f"tape_{i}"
        name = result["name"].split(".")[-1]
        n = result["number"]
        seed = result["seed"]
        success = ("+" if bool(result["success"]) else "") if "success" in result else "~"
        return f"{success}{n+1}_{seed}_{name}"

    def get_tape_label(self, tape: dict) -> str:
        tape_dir = tape["metadata"]["tape_dir"]
        llm_calls_num = 0
        tokens_num = 0
        for step in tape["steps"]:
            prompt_id = step.get("metadata", {}).get("prompt_id", step.pop("prompt_id", None))
            if prompt_id:
                llm_calls_num += 1
                if prompt_id in self.llm_calls:
                    tokens_num += (
                        self.llm_calls[prompt_id].prompt_length_tokens + self.llm_calls[prompt_id].output_length_tokens
                    )
        failure_count = len([s for s in tape["steps"] if s["kind"].endswith("failure")])
        label = f"""<h3>Result</h3>
            <div class="result-label">Steps: {len(tape["steps"])}</div>
            <div class="result-label">Failures: {failure_count}</div>"""
        result = tape["metadata"].get("result")
        if result is None or not isinstance(result, dict):
            return label
        success = str(result.pop("success")) if "success" in result else "UNFINISHED"
        label += f"""
            <div class="result-success"><b>Finished successfully: {success}</b></div>
            <div class="result-label">LLM Calls: {llm_calls_num}</div>
            <div class="result-label">Tokens spent: {tokens_num}</div>
            <h3>Info</h3>
            """
        for k, v in result.items():
            if k == "video":
                label += f'<div>Browser Video:<br><a href="static/{tape_dir}/videos/task_video/{v}">{v}</a></div>'
            elif k == "chat_video":
                label += f'<div>Chat Video:<br><a href="static/{tape_dir}/videos/chat_video/{v}">{v}</a></div>'
            elif v or v == 0:
                label += f"<div>{k}: {v}</div>"
        return label

    def get_tape_files(self) -> list[str]:
        runs = [d for d in os.listdir(self.tapes_folder) if os.path.isdir(os.path.join(self.tapes_folder, d, "tapes"))]
        assert runs, f"No runs found in {self.tapes_folder}"
        logger.info(f"Found {len(runs)} runs")
        return sorted(runs)

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


class WorkArenaRender(CameraReadyRenderer):
    def __init__(self, exp_dir: str) -> None:
        self.exp_dir = exp_dir
        super().__init__()

    def render_step(self, step: Step | dict, index: int, folded: bool = True, **kwargs) -> str:
        step_dict = step.model_dump() if isinstance(step, Step) else step
        screenshot_path = step_dict.get("metadata", {}).get("other", {}).get("screenshot_path")
        html = super().render_step(step, folded, **kwargs)
        if screenshot_path:
            screenshot_url = os.path.join("static", screenshot_path)
            html = f"<div class='basic-renderer-box' style='background-color:#baffc9;'><div><img src='{screenshot_url}' style='max-width: 100%;'></div>{html}</div>"
        return html


def main(dirname: str):
    renderer = WorkArenaRender(dirname)
    browser = WorkarenaTapeBrowser(dirname, renderer)
    browser.launch(static_dir=dirname, port=7861)


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python -m examples.workarena.tape_browser <dirname>"
    main(sys.argv[1])
