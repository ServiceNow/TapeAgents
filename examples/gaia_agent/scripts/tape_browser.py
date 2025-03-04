import logging
import os
import sys
from collections import defaultdict

from pydantic import TypeAdapter

from tapeagents.core import Action
from tapeagents.io import load_legacy_tapes
from tapeagents.observe import retrieve_all_llm_calls
from tapeagents.renderers.camera_ready_renderer import CameraReadyRenderer
from tapeagents.tape_browser import TapeBrowser

from ..eval import calculate_accuracy, get_exp_config_dict, load_dataset, tape_correct
from ..steps import GaiaStep, GaiaTape

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class GaiaTapeBrowser(TapeBrowser):
    def __init__(self, tapes_folder: str, renderer):
        super().__init__(tape_cls=GaiaTape, tapes_folder=tapes_folder, renderer=renderer, file_extension=".json")

    def load_tapes(self, name: str) -> list:
        _, exp_dir, postfix = name.split("/", maxsplit=2)
        try:
            cfg_dir = os.path.join(self.tapes_folder, exp_dir)
            cfg = get_exp_config_dict(cfg_dir)
            try:
                tasks = load_dataset(cfg["split"])
                tasks_list = [task for level in tasks.values() for task in level]
                self.task_id_to_num = {task["task_id"]: i + 1 for i, task in enumerate(tasks_list)}
            except Exception:
                self.task_id_to_num = {}
        except Exception as e:
            logger.exception(f"Failed to load config from {cfg_dir}: {e}")
            self.task_id_to_num = {}
        tapes_path = os.path.join(self.tapes_folder, exp_dir, "tapes")
        image_dir = os.path.join(self.tapes_folder, exp_dir, "attachments", "images")
        if not os.path.exists(image_dir):
            image_dir = os.path.join(self.tapes_folder, exp_dir, "images")

        try:
            all_tapes: list[GaiaTape] = load_legacy_tapes(GaiaTape, tapes_path, step_class=TypeAdapter(GaiaStep))  # type: ignore
        except Exception as e:
            logger.error(f"Failed to load tapes from {tapes_path}: {e}")
            return []
        tapes = []
        for tape in all_tapes:
            if postfix == "all" or str(tape.metadata.level) == postfix:
                tapes.append(tape)
            for i in range(len(tape.steps)):
                if tape.steps[i].kind != "image":
                    continue
                image_path = os.path.join(image_dir, f"{tape.steps[i].metadata.id}.png")
                if os.path.exists(image_path):
                    tape.steps[i].metadata.other["image_path"] = os.path.join(
                        exp_dir, "images", f"{tape.steps[i].metadata.id}.png"
                    )
                else:
                    logger.warning(f"Image not found: {image_path}")

        self.llm_calls = {}
        sqlite_fpath = os.path.join(self.tapes_folder, exp_dir, "tapedata.sqlite")
        if not os.path.exists(sqlite_fpath):
            sqlite_fpath = os.path.join(self.tapes_folder, exp_dir, "llm_calls.sqlite")
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

    def get_exp_label(self, filename: str, tapes: list[GaiaTape]) -> str:
        acc, n_solved = calculate_accuracy(tapes)
        errors = defaultdict(int)
        prompt_tokens_num = 0
        output_tokens_num = 0
        total_cost = 0.0
        visible_prompt_tokens_num = 0
        visible_output_tokens_num = 0
        visible_cost = 0.0
        no_result = 0
        actions = defaultdict(int)
        for llm_call in self.llm_calls.values():
            prompt_tokens_num += llm_call.prompt_length_tokens
            output_tokens_num += llm_call.output_length_tokens
            total_cost += llm_call.cost
        for tape in tapes:
            if tape.metadata.result in ["", None, "None"]:
                no_result += 1
            if tape.metadata.error:
                errors["fatal"] += 1
            if tape.metadata.terminated:
                errors["terminated"] += 1
            last_action = None
            counted = set([])
            for step in tape:
                llm_call = self.llm_calls.get(step.metadata.prompt_id)
                if llm_call and step.metadata.prompt_id not in counted:
                    counted.add(step.metadata.prompt_id)
                    visible_prompt_tokens_num += llm_call.prompt_length_tokens
                    visible_output_tokens_num += llm_call.output_length_tokens
                    visible_cost += llm_call.cost
                if isinstance(step, Action):
                    actions[step.kind] += 1
                    last_action = step
                if step.kind == "search_results_observation" and not step.serp:
                    errors["search_empty"] += 1
                if step.kind == "page_observation" and step.error:
                    errors["browser"] += 1
                elif step.kind == "llm_output_parsing_failure_action":
                    errors["parsing"] += 1
                elif step.kind == "action_execution_failure":
                    if last_action:
                        errors[f"{last_action.kind}"] += 1
                    else:
                        errors["unknown_action_execution_failure"] += 1
                elif step.kind == "code_execution_result" and step.result.exit_code:
                    errors["code_execution"] += 1
        timers, timer_counts = self.aggregate_timer_times(tapes)
        html = f"<h2>Solved {acc:.2f}%, {n_solved} out of {len(tapes)}</h2>"
        if "all" in filename:
            html += f"Prompt tokens: {prompt_tokens_num}<br>Output tokens: {output_tokens_num}<br>Cost: {total_cost:.2f} USD<h3>Visible</h3>"
        html += f"Prompt tokens: {visible_prompt_tokens_num}<br>Output tokens: {visible_output_tokens_num}<br>Cost: {visible_cost:.2f} USD"
        if errors:
            errors_str = "<br>".join(f"{k}: {v}" for k, v in errors.items())
            html += f"<h2>No result: {no_result}</h2>"
            html += f"<h2>Errors: {sum(errors.values())}</h2>{errors_str}"
        if actions:
            actions_str = "<br>".join(f"{k}: {v}" for k, v in actions.items())
            html += f"<h2>Actions: {sum(actions.values())}</h2>{actions_str}"
        if timers:
            timers_str = "<br>".join(
                f"{'execute ' if k.endswith('action') else ''}{k}: {v:.1f} sec, avg. {v/timer_counts[k]:.1f} sec"
                for k, v in timers.items()
            )
            html += f"<h2>Timings</h2>{timers_str}"
        return html

    def get_tape_name(self, i: int, tape: GaiaTape) -> str:
        error = "F" if tape.metadata.error else ""
        if tape.metadata.terminated:
            error = "T"
        last_action = None
        tokens = 0
        for step in tape:
            llm_call = self.llm_calls.get(step.metadata.prompt_id)
            tokens += llm_call.prompt_length_tokens if llm_call else 0
            tokens += llm_call.output_length_tokens if llm_call else 0
            if isinstance(step, Action):
                last_action = step
            if step.kind == "search_results_observation" and not step.serp:
                error += "se"
            elif step.kind == "page_observation" and step.error:
                error += "br"
            elif step.kind == "llm_output_parsing_failure_action":
                error += "pa"
            elif step.kind == "action_execution_failure" and last_action:
                error += last_action.kind[:2]
            elif step.kind == "code_execution_result" and step.result.exit_code:
                error += "ce"
        mark = "+" if tape_correct(tape) else ("" if tape.metadata.result else "∅")
        if tape.metadata.task.get("file_name"):
            mark += "📁"
        if error:
            mark += f"[{error}]"
        if mark:
            mark += " "
        n = self.task_id_to_num.get(tape.metadata.task.get("task_id"), "")
        name = tape[0].content[:32] if hasattr(tape[0], "content") else tape[0].short_view()[:32]
        return f"{n} {mark}({tokens: }t) {name}"  # type: ignore

    def get_tape_label(self, tape: GaiaTape) -> str:
        llm_calls_num = 0
        input_tokens_num = 0
        output_tokens_num = 0
        cost = 0

        for step in tape:
            prompt_id = step.metadata.prompt_id
            if prompt_id and prompt_id in self.llm_calls:
                llm_calls_num += 1
                if prompt_id in self.llm_calls:
                    llm_call = self.llm_calls[prompt_id]
                    input_tokens_num += llm_call.prompt_length_tokens
                    output_tokens_num += llm_call.output_length_tokens
                    cost += llm_call.cost
        failure_count = len(
            [step for step in tape if "failure" in step.kind or (step.kind == "page_observation" and step.error)]
        )
        success = tape[-1].success if hasattr(tape[-1], "success") else ""  # type: ignore
        overview = tape[-1].overview if hasattr(tape[-1], "overview") else ""  # type: ignore
        label = f"""<h2>Result</h2>
            <div class="result-label expected">Golden Answer: <b>{tape.metadata.task.get('Final answer', '')}</b></div>
            <div class="result-label">Agent Answer: <b>{tape.metadata.result}</b></div>
            <div class="result-success">Finished successfully: {success}</div>
            <h2>Summary</h2>
            <div class="result-overview">{overview}</div>
            <h2>Stats</h2>
            <div class="result-label">Steps: {len(tape)}</div>
            <div class="result-label">Failures: {failure_count}</div>
            <div>LLM Calls: {llm_calls_num}</div>
            <div>Input tokens: {input_tokens_num}</div>
            <div>Output tokens: {output_tokens_num}</div>
            <div>Cost: {cost:.2f} USD</div>
        """
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
                try:
                    try:
                        cfg = get_exp_config_dict(exp_dir)
                    except Exception:
                        cfg = {}
                    if "split" in cfg:
                        set_name = cfg["split"]
                    elif "data_dir" in cfg:
                        parts = cfg["data_dir"].split("/")
                        set_name = parts[-2] if cfg["data_dir"].endswith("/") else parts[-1]
                    else:
                        set_name = ""
                except Exception as e:
                    logger.error(f"Failed to load config from {exp_dir}: {e}")
                    set_name = ""
                exps.append(f"{set_name}/{r}/{postfix}")
        return sorted(exps)

    def aggregate_timer_times(self, tapes: list[GaiaTape]):
        timer_sums = defaultdict(float)
        timer_counts = defaultdict(int)
        for tape in tapes:
            timers = tape.metadata.other.get("timers", {})
            for timer_name, exec_time in timers.items():
                timer_sums[timer_name] += exec_time
                timer_counts[timer_name] += 1
            for step in tape.steps:
                action_kind = step.metadata.other.get("action_kind")
                action_execution_time = step.metadata.other.get("action_execution_time")
                if action_kind and action_execution_time:
                    timer_sums[action_kind] += action_execution_time
                    timer_counts[action_kind] += 1
        return dict(timer_sums), dict(timer_counts)


def main(dirname: str):
    browser = GaiaTapeBrowser(dirname, CameraReadyRenderer())
    browser.launch(port=7861)


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python -m scripts.tape_browser <dirname>"
    main(sys.argv[1])
