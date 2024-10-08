import json
import re
from typing import Any, Type

import yaml
from pydantic import BaseModel

from tapeagents.agent import Agent
from tapeagents.container_executor import CodeBlock
from tapeagents.core import Action, Episode, Observation, Step, Tape, Thought
from tapeagents.dialog_tape import (
    AssistantStep,
    DialogContext,
    DialogTape,
    SystemStep,
    ToolCalls,
    ToolResult,
    UserStep,
)
from tapeagents.environment import CodeExecutionResult, ExecuteCode
from tapeagents.observe import LLMCall, retrieve_tape_llm_calls
from tapeagents.view import Call, Respond

YELLOW = "#ffffba"
LIGHT_YELLOW = "#ffffdb"
SAND = "#e5e5a7"
RED = "#ffdfba"
GREEN = "#baffc9"
BLUE = "#bae1ff"


def render_dialog_plain_text(tape: DialogTape | None) -> str:
    if tape is None:
        return ""
    lines = []
    for step in tape:
        if isinstance(step, UserStep):
            lines.append(f"User: {step.content}")
        elif isinstance(step, AssistantStep):
            lines.append(f"Assistant: {step.content}")
        elif isinstance(step, ToolCalls):
            for tc in step.tool_calls:
                lines.append(f"Tool calls: {tc.function}")
        elif isinstance(step, ToolResult):
            lines.append(f"Tool result: {step.content}")
    return "\n".join(lines)


class BasicRenderer:
    metadata_header = "<h3>Metadata</h3>"
    context_header = "<h3>Context</h3>"
    steps_header = "<h3>Steps</h3>"
    agent_tape_header = "<h1> Agent Tape </h1>"
    user_tape_header = "<h1> User Tapes </h1>"
    annotator_tape_header = "<h1> Annotator Tapes </h1>"

    def __init__(self, filter_steps: tuple[Type, ...] | None = None, render_llm_calls: bool = True):
        self.filter_steps = filter_steps
        self.render_llm_calls = render_llm_calls

    @property
    def style(self) -> str:
        return (
            "<style>"
            ".basic-renderer-box { margin: 4px; padding: 2px; background: lavender; white-space: pre-wrap; color: black;}"
            ".basic-prompt-box { margin: 4px; padding: 2px; background: lavender; color: black;}"
            ".episode-row { display: flex; align-items: end; }"
            ".agent-column { width: 50%; }"
            ".user-column { width: 25%; }"
            ".annotator-column { width: 25%; }"
            ".inner-tape-container { display: flex }"
            ".inner-tape-indent { width: 10%; }"
            ".inner-tape { width: 90%; }"
            "</style>"
        )

    def render_as_box(self, data: Any):
        if isinstance(data, dict):
            str_ = yaml.dump(data, indent=2)
        else:
            str_ = str(data)
        return f"<div class='basic-renderer-box'>{str_}</div>"

    def render_metadata(self, tape: Tape):
        return f"<details> <summary> Show / Hide </summary> {self.render_as_box(tape.metadata.model_dump())} </details>"

    def render_context(self, tape: Tape):
        if isinstance(tape.context, Tape):
            summary = f"Tape of {len(tape.context)} steps." f"<div>ID: {tape.context.metadata.id}</div>"
            return (
                "<div class=inner-tape-container>"
                "<div class=inner-tape-indent> </div>"
                "<div class=inner-tape> <details>"
                f"<summary>{summary}</b></summary>"
                f"{self.render_tape(tape.context)}</details>"
                "</div></div>"
            )
        else:
            context_str = tape.context.model_dump() if isinstance(tape.context, BaseModel) else tape.context
            return self.render_as_box(context_str)

    def render_step(self, step: Step, index: int, **kwargs) -> str:
        step_dict = step.model_dump()
        return self.render_as_box(step_dict)

    def render_steps(self, tape: Tape, llm_calls: dict[str, LLMCall] = {}) -> str:
        chunks = []
        last_prompt_id = None
        for index, step in enumerate(tape):
            if self.filter_steps and not isinstance(step, self.filter_steps):
                continue
            if self.render_llm_calls:
                if step.metadata.prompt_id != last_prompt_id:
                    if last_prompt_id:
                        chunks.append("<hr style='margin: 2pt 0pt 2pt 0pt;'>")
                    llm_call = llm_calls.get(step.metadata.prompt_id)
                    if llm_call:
                        chunks.append(self.render_llm_call(llm_call))
                    last_prompt_id = step.metadata.prompt_id
            chunks.append(self.render_step(step, index))
        return "".join(chunks)

    def render_tape(self, tape: Tape, llm_calls: dict[str, LLMCall] = {}) -> str:
        metadata_html = self.render_metadata(tape)
        context_html = self.render_context(tape)
        steps_html = self.render_steps(tape, llm_calls)
        return (
            f"{self.metadata_header}{metadata_html}"
            + (f"{self.context_header}{context_html}" if tape.context is not None else "")
            + f"{self.steps_header}{steps_html}"
        )

    def render_episode(self, episode: Episode) -> str:
        chunks = []

        def wrap_agent(html: str) -> str:
            return f"<div class='agent-column'>{html}</div>"

        def wrap_user(html: str) -> str:
            return f"<div class='user-column'>{html}</div>"

        def wrap_annotator(html: str) -> str:
            return f"<div class='annotator-column'>{html}</div>"

        def row(user: str, agent: str, annotator: str):
            return f"<div class='episode-row'>{wrap_user(user)}{wrap_agent(agent)}{wrap_annotator(annotator)}</div>"

        chunks.append(row(self.user_tape_header, self.agent_tape_header, self.annotator_tape_header))
        chunks.append(row("", self.context_header, ""))
        chunks.append(row("", self.render_context(episode.tape), ""))
        chunks.append(row("", self.steps_header, ""))
        for index, (user_tape, step, annotator_tapes) in enumerate(episode.group_by_step()):
            if user_tape:
                user_html = f"{self.steps_header}{self.render_steps(user_tape)}"
            else:
                user_html = ""
            agent_html = self.render_step(step, index)
            annotations_html = "".join([f"{self.steps_header}{self.render_steps(tape)}" for tape in annotator_tapes])
            chunks.append(row(user_html, agent_html, annotations_html))
        return "".join(chunks)

    def render_llm_call(self, llm_call: LLMCall | None) -> str:
        if llm_call is None:
            return ""
        if llm_call.prompt.tools:
            prompt_messages = [f"tool_schemas: {json.dumps(llm_call.prompt.tools, indent=2)}"]
        else:
            prompt_messages = []
        for m in llm_call.prompt.messages:
            role = f"{m['role']} ({m['name']})" if "name" in m else m["role"]
            prompt_messages.append(f"{role}: {m['content'] if 'content' in m else m['tool_calls']}")
        prompt_text = "\n--\n".join(prompt_messages)
        prompt_length_str = (
            f"{llm_call.prompt_length_tokens} tokens"
            if llm_call.prompt_length_tokens
            else f"{len(prompt_text)} characters"
        )
        label = f"Prompt {prompt_length_str} {', cached' if llm_call.cached else ''}"
        html = f"""<div class='basic-prompt-box' style='background-color:#ffffba; margin: 0 4px;'>
        <details>
            <summary>{label}</summary>
            <pre style='font-size: 12px; white-space: pre-wrap;word-wrap: break-word;'>{prompt_text.strip()}</pre>
        </details>
        </div>"""
        if llm_call.output:
            html += f"""<div class='basic-prompt-box' style='background-color:#ffffba; margin: 0 4px;'>
                <details>
                    <summary>LLM Output {llm_call.output_length_tokens} tokens</summary>
                    <pre style='font-size: 12px; white-space: pre-wrap; word-wrap: break-word;'>{llm_call.output}</pre>
                </details>
                </div>"""
        return html


class PrettyRenderer(BasicRenderer):
    """Rendering enhancements for a handful of known steps."""

    def __init__(self, show_metadata=False, **kwargs):
        self.show_metadata = show_metadata
        super().__init__(**kwargs)

    @property
    def style(self):
        return super().style + (
            "<style>"
            ".observation { background-color: #baffc9;; }"
            ".error_observation { background-color: #dd0000; }"
            ".action { background-color: #cccccc; }"
            ".thought { background-color: #ffffdb; }"
            ".call { background-color: #ffffff; }"
            ".respond { background-color: #ffffff; }"
            ".step-header { margin: 2pt 2pt 2pt 0 !important; }"
            ".step-text { font-size: 12px; white-space: pre-wrap; word-wrap: break-word;}"
            "</style>"
        )

    def render_step(self, step: Step, index: int, **kwargs):
        title = type(step).__name__
        if isinstance(step, UserStep):
            role = "User"
            title = ""
            class_ = "observation"
        elif isinstance(step, SystemStep):
            role = "System"
            title = ""
            class_ = "observation"
        elif isinstance(step, AssistantStep):
            role = "Assistant"
            title = ""
            class_ = "action"
        elif isinstance(step, DialogContext):
            role = ""
            class_ = "observation"
        elif isinstance(step, Call):
            role = ""
            title = f"{step.metadata.agent.split('/')[-1]} calls {step.agent_name}"
            class_ = "call"
        elif isinstance(step, Respond):
            role = ""
            parts = step.metadata.agent.split("/")
            title = f"{parts[-1]} responds to {parts[-2]}" if len(parts) > 1 else f"{step.metadata.agent} responds"
            class_ = "respond"
        elif isinstance(step, Thought):
            role = "Thought"
            class_ = "thought"
        elif isinstance(step, Action):
            role = "Action"
            class_ = "action"
        elif isinstance(step, CodeExecutionResult):
            role = "Observation"
            class_ = "error_observation" if step.result.exit_code != 0 else "observation"
        elif isinstance(step, Observation):
            role = "Observation"
            class_ = "observation"
        else:
            raise ValueError(f"Unknown object type: {type(step)}")

        dump = step.model_dump()
        if not self.show_metadata:
            dump.pop("metadata", None)

        def pretty_yaml(d: dict):
            return yaml.dump(d, sort_keys=False, indent=2) if d else ""
        def maybe_fold(content: Any):
            content = str(content)
            summary = f"{len(content)} characters ..." 
            if len(content) > 1000:
                return f"<details><summary>{summary}</summary>{content}</details>"
            return content

        if (content := getattr(step, "content", None)) is not None:
            # TODO: also show metadata here
            del dump["content"]
            text = pretty_yaml(dump) + ("\n" + maybe_fold(content) if content else "")
        elif isinstance(step, ExecuteCode):
            del dump["code"]

            def format_code_block(block: CodeBlock) -> str:
                return f"```{block.language}\n{block.code}\n```"

            code_blocks = "\n".join([format_code_block(block) for block in step.code])
            text = pretty_yaml(dump) + "\n" + maybe_fold(code_blocks)
        elif isinstance(step, CodeExecutionResult):
            del dump["result"]["output"]
            text = pretty_yaml(dump) + "\n" + maybe_fold(step.result.output)
        else:
            text = pretty_yaml(dump)

        index_str = f"[{index}]"
        header_text = title if not role else (role if not title else f"{role}: {title}")
        header = f"{index_str} {header_text}"

        return (
            f"<div class='basic-renderer-box {class_}'>"
            f"<h4 class='step-header'>{header}</h4>"
            f"<pre class='step-text'>{text}</pre>"
            f"</div>"
        )


class GuidedAgentRender(BasicRenderer):
    @property
    def style(self) -> str:
        return (
            "<style>"
            ".basic-renderer-box { margin: 0 4px 4px 4px; padding: 10px; background: lavender; } "
            ".basic-prompt-box { margin: 4px; padding: 0 10px; background: lavender;}"
            ".episode-row { display: flex; align-items: end; } "
            ".agent-column { width: 70%; } "
            ".user-column { width: 15%; } "
            ".annotator-column { width: 15%; } "
            ".inner-tape-container { display: flex }"
            ".inner-tape-indent { width: 10%; }"
            ".inner-tape { width: 90%; }"
            ".prompt { margin-top: 1em; padding: 0 10px 0 10px;} "
            "table.diff { border: none !important; padding: 0 !important; } "
            "tr.diff { border: none !important; padding: 0 !important; } "
            "td.diff { border: none !important; padding: 0 !important; vertical-align: top !important;} "
            "td.diff_highlight { border: 0 none red !important; border-left: 5px solid red !important; padding: 0 !important; vertical-align: top !important;} "
            "</style>"
        )

    def render_step(
        self, step: Step | dict, index: int, folded: bool = True, min_fold_lines: int = 20, **kwargs
    ) -> str:
        step_dict = step.model_dump() if isinstance(step, Step) else step
        if not step_dict:
            return ""
        step_dict.pop("metadata", None)
        title = get_step_title(step_dict)
        text = get_step_text(step_dict, exclude_fields={"kind", "role", "metadata"})
        role = "Agent Action"
        color = YELLOW

        if step_dict["kind"] == "question":
            role = "Question"
            color = BLUE
            text = step_dict.get("content", step_dict.get("question", ""))
            title = ""
            folded = False
        elif step_dict["kind"] == "task":
            role = "Task"
            color = BLUE
            text = ""
            title = step_dict["task"]
            folded = False
        elif step_dict["kind"].endswith("_subtask_thought"):
            role = "Agent Thought"
            color = SAND
        elif step_dict["kind"].endswith("_answer_action"):
            role = "Answer"
            color = BLUE
            title = ""
        elif step_dict["kind"].endswith("_failure"):
            role = "Failure"
            color = RED
        elif step_dict["kind"].endswith("_observation"):
            role = "Observation"
            color = GREEN
        elif step_dict["kind"].endswith("_thought"):
            role = "Agent Thought"
            color = LIGHT_YELLOW

        # fold when too long or too many lines
        max_len = min_fold_lines * 100
        fold = folded and (text.count("\n") > min_fold_lines or len(text) > max_len)
        if not fold:
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

    def render_context(self, tape: Tape):
        if isinstance(tape.context, Tape):
            return (
                "<div class=inner-tape-container>"
                "<div class=inner-tape-indent> </div>"
                f"<div class=inner-tape> {self.render_tape(tape.context)} </div>"
                "</div>"
            )
        else:
            context_str = tape.context.model_dump() if isinstance(tape.context, BaseModel) else tape.context
            return self.render_as_box(context_str)

    def wrap_urls_in_anchor_tag(self, text: str) -> str:
        url_pattern = re.compile(r"(https?://\S+)")

        def replace_url(match):
            url = match.group(0)
            return f'<a target="_blank" href="{url}">{url}</a>'

        return url_pattern.sub(replace_url, text)


def step_view(step: Step, trim: bool = False) -> str:
    title = get_step_title(step)
    text = get_step_text(step, trim)
    return f"{title}:\n{text}"


def get_step_title(step: Step | dict) -> str:
    title = ""
    step_dict = step if isinstance(step, dict) else step.model_dump()
    if kind := step_dict.get("kind", None):
        if kind.endswith("_thought"):
            kind = kind[:-8]
        elif kind.endswith("_action"):
            kind = kind[:-7]
        elif kind.endswith("_observation"):
            kind = kind[:-12]
        title = kind.replace("_", " ").title()
    return title


def get_step_text(step: Step | dict, trim: bool = False, exclude_fields={"kind", "role", "prompt_id"}) -> str:
    step_dict = step if isinstance(step, dict) else step.model_dump()
    clean_dict = {
        k: (v[:300] if trim and isinstance(v, str) else v)
        for k, v in step_dict.items()
        if k not in exclude_fields and v != [] and v != {}
    }
    if not clean_dict:
        return ""
    return to_pretty_str(clean_dict).strip()


def to_pretty_str(a: Any, prefix: str = "", indent: int = 2) -> str:
    view = ""
    if isinstance(a, list) and len(a):
        if len(str(a)) < 80:
            view = str(a)
        else:
            lines = []
            for item in a:
                value_view = to_pretty_str(item, prefix + " " * indent)
                if "\n" in value_view:
                    value_view = f"\n{value_view}"
                lines.append(f"{prefix}- " + value_view)
            view = "\n".join(lines)
    elif isinstance(a, dict) and len(a):
        lines = []
        for k, v in a.items():
            value_view = to_pretty_str(v, prefix + " " * indent)
            if "\n" in value_view:
                value_view = f"\n{value_view}"
            lines.append(f"{prefix}{k}: {value_view}")
        view = "\n".join(lines)
    else:
        view = str(a)
    return view


def render_agent_tree(agent: Agent):
    """Draw an ASCII tree of the agent's structure.

    Like this:

    - The Manager
      - His Assistant 1
      - His Helper 2

    """

    def render_subagents(agent, indent=4):
        lines = []
        for subagent in agent.subagents:
            lines.append("  " * indent + f"- {subagent.name}")
            lines.extend(render_subagents(subagent, indent + 4))
        return lines

    return agent.name + "\n" + "\n".join(render_subagents(agent))


def render_tape_with_prompts(tape: Tape, renderer: BasicRenderer):
    llm_calls = retrieve_tape_llm_calls(tape)
    return renderer.style + renderer.render_tape(tape, llm_calls)
