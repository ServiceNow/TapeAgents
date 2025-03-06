"""
Various utility functions.
"""

import base64
import difflib
import fcntl
import importlib
import io
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import jsonref
from PIL import Image
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
    if key in d:
        d.pop(key)
    return d


def get_step_schemas_from_union_type(cls, simplify: bool = True) -> str:
    schema = TypeAdapter(cls).json_schema()
    dereferenced_schema: dict = dict(jsonref.replace_refs(schema, proxies=False))  # type: ignore
    clean_schema = []
    for step in dereferenced_schema["oneOf"]:
        step["properties"] = without(step["properties"], "metadata")
        if simplify:
            step = without(step, "title")
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


def resize_base64_message(message: dict, max_size: int = 1280) -> dict:
    base64_image = message["image_url"]["url"].split(",", maxsplit=1)[1]
    image = Image.open(io.BytesIO(base64.b64decode(base64_image)))
    if image.size[0] > max_size or image.size[1] > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    with tempfile.NamedTemporaryFile() as tmp:
        image.save(tmp.name, "PNG")
        base64_image = encode_image(tmp.name)
        content_type = "image/png"
    message = {"type": "image_url", "image_url": {"url": f"data:{content_type};base64,{base64_image}"}}
    return message


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


@contextmanager
def acquire_timeout(lock, timeout):
    result = lock.acquire(timeout=timeout)
    try:
        yield result
    finally:
        if result:
            lock.release()


def class_for_name(full_name: str) -> Any:
    if "." in full_name:
        module_name, class_name = full_name.rsplit(".", 1)
    else:
        module_name = "."
        class_name = full_name
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c


class Lock:
    def __init__(self, name: str):
        self.name = f"./.{name}.lock"
        Path(self.name).touch()

    def __enter__(self):
        self.fp = open(self.name)
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX)

    def __exit__(self, _type, value, tb):
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
        self.fp.close()
