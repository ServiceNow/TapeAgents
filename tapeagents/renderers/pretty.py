from typing import Any

import yaml

from tapeagents.core import Action, Observation, Step, Thought
from tapeagents.dialog_tape import AssistantStep, DialogContext, SystemStep, UserStep
from tapeagents.environment import CodeExecutionResult, ExecuteCode
from tapeagents.renderers.basic import BasicRenderer
from tapeagents.tools.container_executor import CodeBlock
from tapeagents.view import Call, Respond


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
