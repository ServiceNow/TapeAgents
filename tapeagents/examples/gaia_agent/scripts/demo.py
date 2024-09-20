import os

import hydra
from omegaconf import DictConfig

from tapeagents.core import Step, Tape
from tapeagents.demo import Demo
from tapeagents.examples.gaia_agent.agent import GaiaAgent
from tapeagents.examples.gaia_agent.environment import GaiaEnvironment
from tapeagents.examples.gaia_agent.steps import GaiaQuestion
from tapeagents.examples.gaia_agent.tape import GaiaTape
from tapeagents.rendering import TapeBrowserRenderer, get_step_text, get_step_title


class GaiaDemo(Demo):
    def add_user_step(self, user_input: str, tape: Tape) -> Tape:
        return tape.append(GaiaQuestion(content=user_input))


class GaiaRender(TapeBrowserRenderer):
    @property
    def style(self) -> str:
        return (
            "<style>"
            ".basic-renderer-box { margin: 4px; padding: 10px; background: lavender; } "
            ".episode-row { display: flex; align-items: end; } "
            ".agent-column { width: 70%; } "
            ".user-column { width: 15%; } "
            ".annotator-column { width: 15%; } "
            ".prompt { margin-top: 1em; padding: 0 10px 0 10px;} "
            "table.diff { border: none !important; padding: 0 !important; } "
            "tr.diff { border: none !important; padding: 0 !important; } "
            "td.diff { border: none !important; padding: 0 !important; vertical-align: top !important;} "
            "td.diff_highlight { border: 0 none red !important; border-left: 5px solid red !important; padding: 0 !important; vertical-align: top !important;} "
            "</style>"
        )

    def render_step(self, step: Step | dict, folded: bool = True, **kwargs) -> str:
        step_dict = step.model_dump() if isinstance(step, Step) else step
        if not step_dict:
            return ""
        title = get_step_title(step_dict)
        text = get_step_text(step_dict, exclude_fields={"kind", "role", "prompt_id"})
        role = "Agent Action"
        color = "#ffffba"
        if step_dict["kind"] == "question":
            role = "Question"
            color = "#bae1ff"
            text = step_dict.get("content", step_dict.get("question", ""))
            title = ""
            folded = False
        elif step_dict["kind"] == "plan_thought":
            role = "Agent Thought"
            color = "#ffffdb"
            folded = False
        elif step_dict["kind"] == "start_subtask_thought":
            role = "Agent Thought"
            color = "#e5e5a7"
        elif step_dict["kind"] == "finish_subtask_thought":
            role = "Agent Thought"
            color = "#e5e5a7"
        elif step_dict["kind"] == "gaia_answer":
            role = "Answer"
            color = "#bae1ff"
            title = ""
        elif step_dict["kind"].endswith("failure"):
            role = "Failure"
            color = "#ffdfba"
        elif step_dict["kind"].endswith("observation"):
            role = "Observation"
            color = "#baffc9"
        elif step_dict["kind"].endswith("thought"):
            role = "Agent Thought"
            color = "#ffffdb"

        # fold when too long or too many lines
        fold = folded and (text.count("\n") > 15 or len(text) > 3000)
        if not fold:
            max_len = 2000
            if len(text) > max_len + 100:
                text = text[:max_len] + "\n" + ("=" * 100) + f"\n ... and {len(text[max_len:])} more characters"
        text = self.wrap_urls_in_anchor_tag(text)
        if fold:
            html = f"""<div class='basic-renderer-box' style='background-color:{color};'><details>
                <summary><b>{role}: {title}</b></summary>
                <pre style='font-size: 12px; white-space: pre-wrap;word-wrap: break-word;'>{text}</pre>
            </details></div>"""
        else:
            html = f"""<div class='basic-renderer-box' style='background-color:{color};'>
                <h4 style='margin: 2pt 2pt 2pt 0 !important;font-size: 1em;'>{role}: {title}</h4>
                <pre style='font-size: 12px; white-space: pre-wrap;word-wrap: break-word;'>{text}</pre>
            </div>"""
        return html


@hydra.main(
    version_base=None,
    config_path="../../../conf/tapeagent",
    config_name="gaia_openai",
)
def main(cfg: DictConfig) -> None:
    llm = hydra.utils.instantiate(cfg.llm)
    env = GaiaEnvironment(vision_lm=llm)
    agent = GaiaAgent(llms={"default": llm}, **cfg.agent)
    demo = GaiaDemo(agent, GaiaTape(), env, GaiaRender())
    demo.launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
