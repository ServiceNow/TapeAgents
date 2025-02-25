"""
Renderers that produce HTML from the tapes used by various GUI scripts.
"""

from typing import Any

from tapeagents.agent import Agent
from tapeagents.core import Step, Tape
from tapeagents.dialog_tape import AssistantStep, DialogTape, UserStep
from tapeagents.observe import retrieve_tape_llm_calls
from tapeagents.renderers.basic import BasicRenderer
from tapeagents.tool_calling import ToolCalls, ToolResult


def render_dialog_plain_text(tape: DialogTape | None) -> str:
    """
    Renders a dialog tape into a plain text format.

    Takes a DialogTape object containing conversation steps and formats them into human-readable text,
    with each dialog step on a new line prefixed by the speaker/action type.

    Args:
        tape (Union[DialogTape, None]): A DialogTape object containing conversation steps, or None

    Returns:
        str: A string containing the formatted dialog, with each step on a new line.
            Returns empty string if tape is None.
    """
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


def to_pretty_str(a: Any, prefix: str = "", indent: int = 2) -> str:
    """Convert any Python object to a pretty formatted string representation.

    This function recursively formats nested data structures (lists and dictionaries)
    with proper indentation and line breaks for improved readability.

    Args:
        a (Any): The object to be formatted. Can be a list, dictionary, or any other type.
        prefix (str, optional): String to prepend to each line. Used for recursive indentation. Defaults to "".
        indent (int, optional): Number of spaces to use for each level of indentation. Defaults to 2.

    Returns:
        str: A formatted string representation of the input object.

    Example:
        ```python
        data = {"foo": [1, 2, {"bar": "baz"}]}
        print(to_pretty_str(data))
        ```
        foo:
          - 1
          - 2
          - bar: baz
    """
    view = ""
    if isinstance(a, list) and len(a):
        if len(str(a)) < 8:
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


def get_step_text(step: Step | dict, trim: bool = False, exclude_fields={"kind", "role"}) -> str:
    step_dict = step if isinstance(step, dict) else step.model_dump()
    clean_dict = {
        k: (v[:300] if trim and isinstance(v, str) else v)
        for k, v in step_dict.items()
        if k not in exclude_fields and v != [] and v != {}
    }
    if not clean_dict:
        return ""
    return to_pretty_str(clean_dict).strip()


def step_view(step: Step, trim: bool = False) -> str:
    title = get_step_title(step)
    text = get_step_text(step, trim)
    return f"{title}:\n{text}"


def render_agent_tree(agent: Agent, show_nodes: bool = True, indent_increment: int = 4) -> str:
    """
    Renders an ASCII tree representation of an agent's hierarchical structure.

    This function creates a visual tree diagram showing the relationships between agents,
    their nodes, and subagents using ASCII characters. Each level is indented to show
    the hierarchy.

    Args:
        agent (Agent): The root agent object to render the tree from.
        show_nodes (bool, optional): Whether to display the agent's nodes in the tree.
            Defaults to True.
        indent_increment (int, optional): Number of spaces to indent each level.
            Defaults to 4.

    Returns:
        str: A string containing the ASCII tree representation.

    Example:
        ```
        > The Manager
            .node1
            .node2
            > His Assistant 1
                .node1
                .node2
            > His Helper 2
                .node1
                .node2
        ```
    """

    def render(agent: Agent, indent: int = 0) -> str:
        lines = [f"{' ' * indent}> {agent.name}"]
        if show_nodes:
            lines.extend([f"{' ' * (indent + indent_increment)}.{node.name}" for node in agent.nodes])
        for subagent in agent.subagents:
            lines.append(render(subagent, indent + indent_increment))
        return "\n".join(lines)

    return render(agent)


def render_tape_with_prompts(tape: Tape, renderer: BasicRenderer):
    """
    Renders a tape with prompts using the specified renderer.

    This function combines the tape's LLM calls with the renderer's style and rendering
    to produce a complete rendered output of the tape's content.

    Args:
        tape (Tape): The tape object to be rendered
        renderer (BasicRenderer): The renderer to use for rendering the tape

    Returns:
        (str): The rendered tape content with applied styling, combining both
             the renderer's style and the tape's content with LLM calls
    """
    llm_calls = retrieve_tape_llm_calls(tape)
    return renderer.style + renderer.render_tape(tape, llm_calls)
