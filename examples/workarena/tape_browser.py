import json
import logging
import os
import sys

from tapeagents.core import Step
from tapeagents.observe import retrieve_all_llm_calls
from tapeagents.tape_browser import TapeBrowser

from ..gaia_agent.eval import load_results
from ..gaia_agent.scripts.demo import GaiaRender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


class WorkarenaTapeBrowser(TapeBrowser):
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
            tapes.append(tape_dict)
        self.prompts = {}
        base_path = os.path.dirname(fpath)
        sqlite_fpath = os.path.join(base_path, "llm_calls.sqlite")
        if os.path.exists(sqlite_fpath):
            self.prompts = {
                llm_call.prompt.id: llm_call.prompt.model_dump() for llm_call in retrieve_all_llm_calls(sqlite_fpath)
            }
        logger.info(f"Loaded {len(tapes)} tapes, {len(self.prompts)} prompts from {fpath}")
        try:
            tapes.sort(key=lambda tape: tape["metadata"]["result"]["number"])
        except Exception:
            logger.warning("Failed to sort tapes by number")
        return tapes

    def get_steps(self, tape) -> list:
        return tape["steps"]

    def get_context(self, tape) -> list:
        return []

    def get_file_label(self, filename: str, tapes: list) -> str:
        acc = []
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
        success_rate = (sum(acc) / len(acc) if acc else 0) * 100
        return f"<h2>{filename}<br>Success {success_rate:.2f}% ({sum(acc)} out of {len(acc)})</h2>"

    def get_tape_name(self, i: int, tape: dict) -> str:
        result = tape["metadata"].get("result")
        if result is None or not isinstance(result, dict):
            return f"tape_{i}"
        name = result["name"].split(".")[-1]
        n = result["number"]
        seed = result["seed"]
        success = ("+" if bool(result["success"]) else "") if "success" in result else "~"
        return f"{success}{n+1}_{seed}_{name}"

    def get_tape_label(self, tape: dict) -> str:
        tape_dir = tape["metadata"]["tape_dir"]
        tape_prompts = [s for s in tape["steps"] if s.get("metadata", {}).get("prompt_id") in self.prompts]
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
            <div>LLM Calls: {len(tape_prompts)}</div>
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
        for s in steps:
            view = self.renderer.render_step(s, tape_dir=tape["metadata"]["tape_dir"])  # type: ignore
            step_metadata = s.get("metadata", {})
            prompt_id = step_metadata.pop("prompt_id", None) if isinstance(s, dict) else getattr(step_metadata, "prompt_id", None)
            if prompt_id in self.prompts and prompt_id != last_prompt_id:
                prompt_view = self.renderer.render_llm_call(self.prompts[prompt_id], metadata=self.prompts[prompt_id])
                view = prompt_view + view
            step_views.append(view)
            last_prompt_id = prompt_id
        steps_html = "".join(step_views)
        html = f"{self.renderer.style}"
        html += f"{self.renderer.steps_header}{steps_html}"
        return html, label


class WorkArenaRender(GaiaRender):
    def __init__(self, root_folder: str) -> None:
        super().__init__()
        self.root_folder = root_folder

    def render_step(self, step: Step | dict, folded: bool = True, **kwargs) -> str:
        step_dict = step.model_dump() if isinstance(step, Step) else step
        html = super().render_step(step, folded, **kwargs)
        screenshot_path = None
        if "screenshot_path" in step_dict:
            screenshot_path = step_dict["screenshot_path"]
        if "screenshot_path" in step_dict.get("metadata", {}).get("other", {}):
            screenshot_path = step_dict["metadata"]["other"]["screenshot_path"]
        if screenshot_path:
            screenshot_url = os.path.join("static", kwargs["tape_dir"], "screenshots", screenshot_path)
            html = f"<div class='basic-renderer-box' style='background-color:#baffc9;'><div><img src='{screenshot_url}' style='max-width: 100%;'></div>{html}</div>"
        return html


def main(dirname: str):
    renderer = WorkArenaRender(dirname)
    browser = WorkarenaTapeBrowser(load_results, dirname, renderer, file_extension=".json")
    browser.launch(static_dir=dirname, server_name="0.0.0.0", port=7860)


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python -m examples.workarena.tape_browser <dirname>"
    main(sys.argv[1])
