import ast
import json
import os

import yaml

from tapeagents.core import Action, Error, Observation, SetNextNode, Step, Thought
from tapeagents.dialog_tape import (
    AssistantStep,
    DialogContext,
    SystemStep,
    UserStep,
)
from tapeagents.environment import CodeExecutionResult, ExecuteCode
from tapeagents.io import UnknownStep
from tapeagents.llms import LLMCall
from tapeagents.renderers.basic import BasicRenderer
from tapeagents.steps import ReasoningThought
from tapeagents.tool_calling import ToolCalls, ToolResult
from tapeagents.tools.code_executor import PythonCodeAction
from tapeagents.tools.container_executor import ANSI_ESCAPE_REGEX, CodeBlock
from tapeagents.view import Broadcast, Call, Respond

YELLOW = "#ffffba"
LIGHT_YELLOW = "#ffffdb"
SAND = "#e5e5a7"
WHITE = "#ffffff"
PURPLE = "#E6E6FA"
RED = "#ff7b65"
GREEN = "#6edb8f"
BLUE = "#bae1ff"


class CameraReadyRenderer(BasicRenderer):
    def __init__(self, show_metadata=False, render_agent_node=True, show_content=True, **kwargs):
        self.show_metadata = show_metadata
        self.show_content = show_content
        super().__init__(render_agent_node=render_agent_node, **kwargs)

    @property
    def style(self):
        return super().style + (
            "<style>"
            f".observation {{ background-color: {GREEN} ;}}"
            f".error {{ background-color: {RED} !important; }}"
            f".action {{ background-color: {BLUE}; }}"
            f".thought {{ background-color: {PURPLE}; }}"
            f".call {{ background-color: {LIGHT_YELLOW}; }}"
            f".respond {{ background-color: {LIGHT_YELLOW}; }}"
            f".broadcast {{ background-color: {LIGHT_YELLOW}; }}"
            ".step-header { margin: 2pt 2pt 2pt 0 !important; }"
            ".step-text { font-size: 12px; white-space: pre-wrap; word-wrap: break-word;}"
            "</style>"
        )

    def render_step(self, step: Step, index: int, **kwargs):
        ### Set Default Values and Remove Fields ####
        dump = step.model_dump()
        dump.pop("kind", None)
        if not self.show_metadata:
            dump.pop("metadata", None)

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
            dump.pop("agent_name", None)
        elif isinstance(step, Respond):
            role = ""
            parts = step.metadata.agent.split("/")
            title = f"{parts[-1]} responds to {parts[-2]}" if len(parts) > 1 else f"{step.metadata.agent} responds"
            class_ = "respond"
            dump.pop("copy_output", None)
        elif isinstance(step, Broadcast):
            role = ""
            parts = step.metadata.agent.split("/")
            title = f"{step.metadata.agent.split('/')[-1]} broadcasts"
            class_ = "broadcast"
        elif isinstance(step, SetNextNode):
            role = ""
            title = f"Set Next Node: {step.next_node}"
            class_ = "thought"
            dump.pop("next_node", None)
        elif isinstance(step, Thought):
            role = "Thought"
            class_ = "thought"
        elif isinstance(step, Action):
            role = "Action"
            class_ = "action"
        elif isinstance(step, CodeExecutionResult):
            role = "Observation"
            class_ = "error" if step.result.exit_code != 0 else "observation"
        elif isinstance(step, ToolResult):
            role = "Observation"
            class_ = "observation"
            dump.pop("tool_call_id", None)
        elif isinstance(step, Observation):
            role = "Observation"
            class_ = "error" if getattr(step, "error", False) else "observation"
        elif isinstance(step, UnknownStep):
            role = "Unknown Step"
            class_ = "error"
        else:
            raise ValueError(f"Unknown object type: {type(step)}")

        if isinstance(step, Error):
            class_ += " error"

        ##### Render text #####
        def pretty_yaml(d: dict):
            return yaml.dump(d, sort_keys=False, indent=2) if d else ""

        def maybe_fold(content: str, len_max: int = 60):
            content = str(content)
            if len(content) > len_max:
                summary = f"{content[:len_max]}...".replace("\n", "\\n")
                return f"<details><summary>{summary}</summary>---<br>{content}</details>"
            return content

        if isinstance(step, Broadcast):
            to = f"to: {', '.join(dump['to'])}"
            text = maybe_fold(to)
        elif isinstance(step, ToolCalls):
            function_calls = []
            for tool_call in dump["tool_calls"]:
                function_calls.append(
                    f"{tool_call['function']['name']}({dict_to_params(tool_call['function']['arguments'])})"
                )
            text = maybe_fold(", ".join(function_calls))
        elif isinstance(step, ExecuteCode):
            del dump["code"]

            def format_code_block(block: CodeBlock) -> str:
                return f"```{block.language}\n{block.code}\n```"

            code_blocks = "\n".join([format_code_block(block) for block in step.code])
            text = pretty_yaml(dump) + "\n" + maybe_fold(code_blocks)
        elif isinstance(step, CodeExecutionResult):
            text = f"exit_code:{step.result.exit_code}\n" if step.result.exit_code else ""
            text += f"{maybe_fold(step.result.output, 2000)}"
            text = ANSI_ESCAPE_REGEX.sub("", text)
            if step.result.exit_code == 0 and step.result.output_files:
                for file in step.result.output_files:
                    text += render_image(file)
        elif isinstance(step, PythonCodeAction):
            text = f"# {step.name}\n{maybe_fold(step.code, 2000)}"
        elif isinstance(step, ReasoningThought):
            text = step.reasoning
        else:
            foldable_keys = ["content", "text"]
            content = ""
            for key in step.__dict__:
                for attr in foldable_keys:
                    if key.endswith(attr):
                        del dump[key]
                        content += str(getattr(step, key, ""))
            text = pretty_yaml(dump) + ("\n" + maybe_fold(content))

        # Augment text with media
        if (video_path := getattr(step, "video_path", None)) is not None:
            text += render_video(
                video_path, getattr(step, "thumbnail_path", None), getattr(step, "subtitle_path", None)
            )
            if (image_paths := getattr(step, "video_contact_sheet_paths", None)) is not None:
                for image_path in image_paths:
                    text += render_image(image_path)
        elif (image_path := getattr(step, "image_path", getattr(step, "thumbnail_path", None))) is not None:
            text += render_image(image_path, getattr(step, "image_caption", None))

        if not self.show_content:
            text = ""

        index_str = f"[{index}]"
        header_text = title if not role else (role if not title else f"{role}: {title}")
        header = f"{index_str} {header_text}"

        return (
            f"<div class='basic-renderer-box {class_}'>"
            f"<h4 class='step-header'>{header}</h4>"
            f"<pre class='step-text'>{text}</pre>"
            f"</div>"
        )

    def render_llm_call(self, llm_call: LLMCall | None) -> str:
        if self.render_llm_calls is False:
            return ""
        if llm_call is None:
            return ""
        prompt_messages = [f"tool_schemas: {json.dumps(llm_call.prompt.tools, indent=2)}"]
        for m in llm_call.prompt.messages:
            # Replace image encoded in base64 with a placeholder
            if "content" in m and isinstance(m["content"], list):
                for c in m["content"]:
                    if c.get("image_url"):
                        if url := c.get("image_url").get("url"):
                            if url.startswith("data:image/"):
                                c["image_url"]["url"] = "data:image/(base64_content)"
            role = f"{m['role']} ({m['name']})" if "name" in m else m["role"]
            prompt_messages.append(f"{role}: {m['content'] if 'content' in m else m['tool_calls']}")
        prompt_text = "\n--\n".join(prompt_messages)
        output = llm_call.output.content or ""
        if llm_call.output.tool_calls:
            tool_calls = "\n".join([call.to_json() for call in llm_call.output.tool_calls])
            output += f"\nTool calls:\n{tool_calls}"
        prompt_length_str = (
            f"{llm_call.prompt_length_tokens} tokens"
            if llm_call.prompt_length_tokens
            else f"{len(prompt_text)} characters"
        )
        completion_length_str = f"{llm_call.output_length_tokens} tokens" if llm_call.output else "No completion"
        label = (
            f"Prompt {prompt_length_str} {' (cached)' if llm_call.cached else ''} | Completion {completion_length_str}"
        )
        html = f"""
        <div class='basic-prompt-box' style='background-color:{WHITE};'>
            <details>
                <summary>{label}</summary>
                <div style='display: flex;'>
                    <div style='flex: 1; margin-right: 10px;'>
                        <pre style='padding-left: 6px; font-size: 12px; white-space: pre-wrap; word-wrap: break-word; word-break: break-all; overflow-wrap: break-word;'>{prompt_text.strip()}</pre>
                    </div>"""

        if llm_call.output:
            html += f"""
                    <div style='flex: 1;'>
                        <pre style='font-size: 12px; white-space: pre-wrap; word-wrap: break-word; word-break: break-all; overflow-wrap: break-word;'>{output}</pre>
                    </div>"""

        html += """
                </div>
            </details>
        </div>"""
        return html


def dict_to_params(arguments: str | dict) -> str:
    """
    Transform a dictionary into a function parameters string.
    Example: {'a': 1, 'b': 2} -> 'a=1, b=2'
    """
    if isinstance(arguments, str):
        arguments = str_to_dict(arguments)
    return ", ".join(f"{key}={value!r}" for key, value in arguments.items())


def str_to_dict(s: str) -> dict:
    """
    Convert a string representation of a dictionary into an actual dictionary.
    Example: "{'a': 1, 'b': 2}" -> {'a': 1, 'b': 2}
    """
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        raise ValueError("Invalid string representation of a dictionary")


def render_video(video_path, thumbnail_path, subtitles_path) -> str:
    html = ""
    video_path = path_to_static(video_path)
    poster_attribute = f"poster='{path_to_static(thumbnail_path)}'" if thumbnail_path else ""
    subtitles_track = (
        f"<track kind='subtitles' src='{path_to_static(subtitles_path)}' srclang='en' label='English' default>"
        if subtitles_path
        else ""
    )
    html = f"""
    <video controls {poster_attribute}>
        <source src='{video_path}' type="video/mp4">
        {subtitles_track}
    </video>
    """
    return html


def render_image(image_path: str, image_caption: str | None = None) -> str:
    if not os.path.exists(image_path):
        return ""
    image_url = path_to_static(image_path)
    figcaption_attribute = f"<pre>{image_caption}</pre>" if image_caption else ""
    return f"<img src='{image_url}' style='max-width: 100%; height: 250px; padding: 4px;'>{figcaption_attribute}"


def path_to_static(path: str) -> str:
    return os.path.join("static", os.path.basename(path))
