"""
Various utility functions.
"""

import base64
import difflib
import json
import os
from typing import Any

import jsonref
from pydantic import TypeAdapter
from termcolor import colored


# Custom exception for fatal errors
class FatalError(Exception):
    pass


def json_value_from_str(v: Any) -> Any:
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            pass
    return v


def diff_dicts(a: dict, b: dict) -> str:
    str_a = json.dumps(a, indent=2, ensure_ascii=False, sort_keys=True)
    str_b = json.dumps(b, indent=2, ensure_ascii=False, sort_keys=True)
    return diff_strings(str_a, str_b, by_words=True)


def diff_strings(a: str, b: str, use_html: bool = False, by_words: bool = False) -> str:
    def html_colored(s, on_color):
        color = on_color[3:] if on_color.startswith("on_") else on_color
        style = f"color: {color} !important;"
        if color == "red":
            style += "text-decoration: line-through !important;"  # strike-through for deleted text
        return f'<span style="{style}">{s}</span>'

    color_fn = html_colored if use_html else colored
    output = []
    if by_words:
        a = a.replace("\n", " \n").split(" ")  # type: ignore
        b = b.replace("\n", " \n").split(" ")  # type: ignore
    matcher = difflib.SequenceMatcher(None, a, b)
    splitter = " " if by_words else ""
    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        a_segment = splitter.join(a[a0:a1])
        b_segment = splitter.join(b[b0:b1])
        if opcode == "equal":
            output.append(a_segment)
        elif opcode == "insert":
            output.append(color_fn(f"{b_segment}", on_color="on_green"))
        elif opcode == "delete":
            output.append(color_fn(f"{a_segment}", on_color="on_red"))
        elif opcode == "replace":
            output.append(color_fn(f"{b_segment}", on_color="on_green"))
            output.append(color_fn(f"{a_segment}", on_color="on_red"))
    return splitter.join(output)


def sanitize_json_completion(completion: str) -> str:
    """
    Return only content inside the first pair of triple backticks if they are present.
    """
    tiks_counter = 0
    lines = completion.strip().split("\n")
    clean_lines = []
    for line in lines:
        if line.startswith("```"):
            tiks_counter += 1
            if tiks_counter == 1:
                clean_lines = []
            elif tiks_counter == 2:
                break
            continue
        clean_lines.append(line)
    return "\n".join(clean_lines)


def without(d: dict, key: str) -> dict:
    d.pop(key)
    return d


def get_step_schemas_from_union_type(cls) -> str:
    schema = TypeAdapter(cls).json_schema()
    dereferenced_schema: dict = dict(jsonref.replace_refs(schema, proxies=False))  # type: ignore
    clean_schema = []
    for step in dereferenced_schema["oneOf"]:
        step = without(step, "title")
        step["properties"] = without(step["properties"], "metadata")
        for prop in step["properties"]:
            step["properties"][prop] = without(step["properties"][prop], "title")
        step["properties"]["kind"] = {"const": step["properties"]["kind"]["const"]}
        clean_schema.append(step)
    return json.dumps(clean_schema, ensure_ascii=False)


def image_base64_message(image_path: str) -> dict:
    image_extension = os.path.splitext(image_path)[1][1:]
    content_type = f"image/{image_extension}"
    base64_image = encode_image(image_path)
    message = {"type": "image_url", "image_url": {"url": f"data:{content_type};base64,{base64_image}"}}
    return message


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
