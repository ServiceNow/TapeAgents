import ast
import json
import os

import yaml
from tapeagents.container_executor import CodeBlock
from tapeagents.core import Action, Error, Observation, SetNextNode, Step, Thought
from tapeagents.dialog_tape import (
    AssistantStep,
    DialogContext,
    SystemStep,
    ToolCalls,
    ToolResult,
    UserStep,
)
from tapeagents.environment import CodeExecutionResult, ExecuteCode
from tapeagents.observe import LLMCall
from tapeagents.rendering import (
    BLUE,
    GREEN,
    LIGHT_YELLOW,
    PURPLE,
    RED,
    WHITE,
    BasicRenderer,
)
from tapeagents.view import Broadcast, Call, Respond


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
            del dump["result"]["output"]
            text = maybe_fold(pretty_yaml(dump["result"]))
            if step.result.exit_code == 0:
                if step.result.output_files:
                    for output_file in step.result.output_files:
                        for file in output_file.split(","):
                            text += render_image(file)
                elif step.result.output:
                    text += f"\n {maybe_fold(step.result.output)}"
        else:
            foldable_keys = ["content", "text"]
            content = ""
            for key in step.__dict__:
                for attr in foldable_keys:
                    if key.endswith(attr):
                        del dump[key]
                        content += getattr(step, key, "")
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
            role = f"{m['role']} ({m['name']})" if "name" in m else m["role"]
            prompt_messages.append(f"{role}: {m['content'] if 'content' in m else m['tool_calls']}")
        prompt_text = "\n--\n".join(prompt_messages)
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
                        <pre style='font-size: 12px; white-space: pre-wrap; word-wrap: break-word; word-break: break-all; overflow-wrap: break-word;'>{llm_call.output.content}</pre>
                    </div>"""

        html += """
                </div>
            </details>
        </div>"""
        return html


def dict_to_params(arguments: str) -> str:
    """
    Transform a dictionary into a function parameters string.
    Example: {'a': 1, 'b': 2} -> 'a=1, b=2'
    """
    if type(arguments) is str:
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
    html = ""
    image_path = path_to_static(image_path)
    figcaption_attribute = f"<pre>{image_caption}</pre>" if image_caption else ""
    html += f"""
    <img src='{image_path}' style='max-width: 100%; height: 250px; padding: 4px;'>
    {figcaption_attribute}
    """
    return html


def path_to_static(path: str) -> str:
    return os.path.join("static", path.split("/")[-1])
