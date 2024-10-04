import contextlib
import difflib
import json
import os
import shutil
import tempfile
from pathlib import Path
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


@contextlib.contextmanager
def run_test_in_tmp_dir(test_name: str):
    """Copy test resources to a temporary directory and run the test there"""
    tmpdir = tempfile.mkdtemp()
    test_data_dir = Path(f"tests/res/{test_name}").resolve()
    os.chdir(tmpdir)
    shutil.copytree(test_data_dir, tmpdir, dirs_exist_ok=True)
    yield


@contextlib.contextmanager
def run_in_tmp_dir_to_make_test_data(test_name: str, keep_llm_cache=False):
    tmpdir = tempfile.mkdtemp()
    os.chdir(tmpdir)
    try:
        yield
    # find all non-directory files that got created
    finally:
        created_files = []
        for root, _, files in os.walk(tmpdir):
            for file in files:
                # For most of the code we test in TapeAgents we create ReplayLLM
                # that looks up prompts and outputs in the SQLite database. For this
                # reason, by default we don't save the LLM cache files. If you want
                # make test data for a Jupyter notebook, you can use the keep_llm_cache
                # to save the LLM cache files.
                if file.startswith("llm_cache") and not keep_llm_cache:
                    continue
                created_files.append(os.path.relpath(os.path.join(root, file), tmpdir))
        cp_source = " ".join(f"$TMP/{f}" for f in created_files)
        test_data_dir = f"tests/res/{test_name}"
        print("Saved test data to ", tmpdir)
        print("To update test data, run these commands:")
        print(f"mkdir {test_data_dir}")
        print(f"TMP={tmpdir}; cp {cp_source} {test_data_dir}")
