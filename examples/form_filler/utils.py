from typing import Any, Type

from . import steps


def render_chat_template(messages: list[dict[str, Any]], context: dict[str, Any]) -> list[dict[str, Any]]:
    formatted_chat = []
    for message in messages:
        formatted_content = message["content"].format(**context)
        formatted_chat.append(
            dict(
                role=message["role"],
                content=formatted_content,
            )
        )
    return formatted_chat


def sanitize_json_completion(completion: str) -> str:
    """
    Return only content inside the first pair of triple backticks if they are present.
    Skips starting text until a json format is detected.
    """
    tiks_counter = 0
    opened = False  # flag to detect if the json section has started
    open_sq_brackets = 0  # count [ and ] to detect the end of the json section
    open_cu_brackets = 0  # count { and } to detect the end of the json section
    lines = completion.strip().split("\n")
    if lines[-1].strip() == "```":
        lines.pop()  # remove last ticks in case the llm forgot to open them
    clean_lines = []
    for line in lines:
        line = line.replace("\\", "")  # remove all backslashes
        line = " ".join(line.split())  # remove all extra spaces
        if line.startswith("```"):
            tiks_counter += 1
            if tiks_counter == 1:
                clean_lines = []
            elif tiks_counter == 2:
                break
            continue
        elif line.startswith("[") or line.startswith("{"):  # detected start of the json section
            if not opened:
                opened = True
                clean_lines = []
        # update bracket counters
        if opened and "[" in line:
            open_sq_brackets += line.count("[")
        if opened and "{" in line:
            open_cu_brackets += line.count("{")
        if opened and "]" in line:
            open_sq_brackets -= line.count("]")
        if opened and "}" in line:
            open_cu_brackets -= line.count("}")
        clean_lines.append(line)
        # detect the end of the json section
        if opened and open_sq_brackets == 0 and open_cu_brackets == 0:
            break
    return "\n".join(clean_lines)


def get_step_cls(obj_name: str) -> Type:
    return getattr(steps, obj_name)
