# Modified from the original source: https://github.com/microsoft/autogen/blob/main/autogen/coding/docker_commandline_code_executor.py
# MIT License

# Copyright (c) Microsoft Corporation.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

from __future__ import annotations

import atexit
import logging
import os
import re
import shutil
from hashlib import md5
from pathlib import Path
from time import sleep
from types import TracebackType
from typing import Any, ClassVar, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field
from typing_extensions import Self

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ANSI_ESCAPE_REGEX = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
DEFAULT_CONTAINER = os.environ.get("TAPEAGENTS_CODE_SANDBOX", "tapeagents-code-exec")


def _wait_for_ready(container: Any, timeout: int = 60, stop_time: float = 0.1) -> None:
    elapsed_time = 0.0
    while container.status != "running" and elapsed_time < timeout:
        sleep(stop_time)
        elapsed_time += stop_time
        container.reload()
        continue
    if container.status != "running":
        raise ValueError("Container failed to start")


__all__ = ("ContainerExecutor",)

DEFAULT_EXECUTION_POLICY = {
    "bash": True,
    "shell": True,
    "sh": True,
    "p-sh": True,
    "powershell": True,
    "ps1": True,
    "python": True,
    "javascript": False,
    "html": False,
    "css": False,
}
LANGUAGE_ALIASES = {"py": "python", "js": "javascript"}


class ContainerExecutor:
    DEFAULT_EXECUTION_POLICY: ClassVar[Dict[str, bool]] = DEFAULT_EXECUTION_POLICY
    LANGUAGE_ALIASES: ClassVar[Dict[str, str]] = LANGUAGE_ALIASES

    def __init__(
        self,
        image: str = "jupyter/scipy-notebook",
        container_name: Optional[str] = None,
        timeout: int = 60,
        work_dir: Union[Path, str] = Path("."),
        bind_dir: Optional[Union[Path, str]] = None,
        auto_remove: bool = True,
        stop_container: bool = True,
        restart_if_exists: bool = False,
        execution_policies: Optional[Dict[str, bool]] = None,
        no_deps: bool = False,
    ):
        """(Experimental) A code executor class that executes code through
        a command line environment in a Docker container.

        The executor first saves each code block in a file in the working
        directory, and then executes the code file in the container.
        The executor executes the code blocks in the order they are received.
        Currently, the executor only supports Python and shell scripts.
        For Python code, use the language "python" for the code block.
        For shell scripts, use the language "bash", "shell", or "sh" for the code
        block.

        Args:
            image (_type_, optional): Docker image to use for code execution.
                Defaults to "python:3-slim".
            container_name (Optional[str], optional): Name of the Docker container
                which is created. If None, will autogenerate a name. Defaults to None.
            timeout (int, optional): The timeout for code execution. Defaults to 60.
            work_dir (Union[Path, str], optional): The working directory for the code
                execution. Defaults to Path(".").
            bind_dir (Union[Path, str], optional): The directory that will be bound
                to the code executor container. Useful for cases where you want to spawn
                the container from within a container. Defaults to work_dir.
            auto_remove (bool, optional): If true, will automatically remove the Docker
                container when it is stopped. Defaults to True.
            stop_container (bool, optional): If true, will automatically stop the
                container when stop is called, when the context manager exits or when
                the Python process exits with atext. Defaults to True.

        Raises:
            ValueError: On argument error, or if the container fails to start.
        """
        self.no_deps = no_deps
        if timeout < 1:
            raise ValueError("Timeout must be greater than or equal to 1.")

        if isinstance(work_dir, str):
            work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        if bind_dir is None:
            bind_dir = work_dir
        elif isinstance(bind_dir, str):
            bind_dir = Path(bind_dir)

        import podman as docker

        client = docker.from_env()
        # Check if the image exists
        try:
            client.images.get(image)
        except docker.errors.ImageNotFound:
            logger.info(f"Pulling image {image}...")
            # Let the docker exception escape if this fails.
            client.images.pull(image)

        if container_name is None:
            container_name = DEFAULT_CONTAINER

        # Start a container from the image, read to exec commands later
        try:
            self._container = client.containers.get(container_name)
            if restart_if_exists:
                logger.info(f"Restarting container {container_name}")
                self._container.stop()
                raise docker.errors.NotFound("Container not found")
            else:
                logger.info(f"Container {container_name} already exists, reuse")
        except docker.errors.NotFound:
            logger.info(f"Creating container {container_name} from image {image}")
            self._container = client.containers.create(
                image,
                name=container_name,
                # Note this change: was needed for Podman
                # entrypoint="/bin/sh",
                entrypoint=["/bin/sh"],
                tty=True,
                auto_remove=auto_remove,
                # volumes={str(bind_dir.resolve()): {"bind": "/workspace", "mode": "rw"}},
                mounts=[
                    {
                        "type": "bind",
                        "source": str(bind_dir.resolve()),
                        "target": "/workspace",
                    }
                ],
                working_dir="/workspace",
            )
            self._container.start()
            _wait_for_ready(self._container)
            logger.info(f"Started container {container_name} from image {image}")

        _wait_for_ready(self._container)

        def cleanup() -> None:
            try:
                container = client.containers.get(container_name)
                container.stop()
            except docker.errors.NotFound:
                pass
            atexit.unregister(cleanup)

        if stop_container:
            atexit.register(cleanup)

        self._cleanup = cleanup

        # Check if the container is running
        if self._container.status != "running":
            raise ValueError(f"Failed to start container from image {image}. Logs: {self._container.logs()}")

        self._timeout = timeout
        self._work_dir: Path = work_dir
        self._bind_dir: Path = bind_dir
        self.execution_policies = self.DEFAULT_EXECUTION_POLICY.copy()
        if execution_policies is not None:
            self.execution_policies.update(execution_policies)

    @property
    def timeout(self) -> int:
        """(Experimental) The timeout for code execution."""
        return self._timeout

    @property
    def work_dir(self) -> Path:
        """(Experimental) The working directory for the code execution."""
        return self._work_dir

    @property
    def bind_dir(self) -> Path:
        """(Experimental) The binding directory for the code execution container."""
        return self._bind_dir

    def execute_code_blocks(self, code_blocks: List[CodeBlock]) -> CommandLineCodeResult:
        """(Experimental) Execute the code blocks and return the result.

        Args:
            code_blocks (List[CodeBlock]): The code blocks to execute.

        Returns:
            CommandlineCodeResult: The result of the code execution."""
        return execute_code_in_container(
            code_blocks,
            str(self._work_dir),
            container_name=self._container.name,
            timeout=self._timeout,
        )

    def restart(self) -> None:
        """(Experimental) Restart the code executor."""
        self._container.restart()
        if self._container.status != "running":
            raise ValueError(f"Failed to restart container. Logs: {self._container.logs()}")

    def stop(self) -> None:
        """(Experimental) Stop the code executor."""
        self._cleanup()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
        self.stop()


def execute_code_in_container(
    code_blocks: List[CodeBlock],
    work_dir: str,
    input_files: list[str] | None = None,
    container_name: str = DEFAULT_CONTAINER,
    container_work_dir: str = "/workspace",
    mounted_dir: str = "",
    timeout: int = 60,
) -> CommandLineCodeResult:
    """Execute the code blocks and return the result. Suitable to run in child processes.

    Args:
        code_blocks (List[CodeBlock]): The code blocks to execute.

    Returns:
        CommandlineCodeResult: The result of the code execution."""

    if len(code_blocks) == 0:
        raise ValueError("No code blocks to execute.")
    import podman as docker

    client = docker.from_env()
    if container_name is None:
        container_name = DEFAULT_CONTAINER
    container = client.containers.get(container_name)
    _wait_for_ready(container)

    work_dir = Path(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    outputs = []
    output_files = []
    files: list[Path] = []
    last_exit_code = 0
    if mounted_dir:
        mounted_dir = Path(mounted_dir)
        if not mounted_dir.is_dir():
            mounted_dir.mkdir(parents=True)
    for code_block in code_blocks:
        lang = LANGUAGE_ALIASES.get(code_block.language.lower(), code_block.language.lower())
        if lang not in DEFAULT_EXECUTION_POLICY:
            outputs.append(f"Unsupported language {lang}\n")
            last_exit_code = 1
            break

        execute_code = DEFAULT_EXECUTION_POLICY.get(lang, False)
        code = silence_pip(code_block.code, lang)

        # Check if there is a filename comment
        try:
            code_filename = _get_file_name_from_content(code, work_dir)
        except ValueError:
            outputs.append("Filename is not in the workspace")
            last_exit_code = 1
            break

        if not code_filename:
            code_filename = f"tmp_code_{md5(code.encode()).hexdigest()}.{lang}"
        if input_files is not None:
            for input_file_path_str in input_files:
                input_file_path = Path(input_file_path_str)
                copy_file_path = work_dir / input_file_path.name
                container_file_path = os.path.join(container_work_dir, input_file_path.name)
                shutil.copy(input_file_path, copy_file_path)
                if mounted_dir:
                    shutil.copy(input_file_path, mounted_dir / input_file_path.name)
                    logger.info(f"Copy `{input_file_path}` to mounted dir `{mounted_dir / input_file_path.name}`")
                logger.info(f"Put `{input_file_path}` at `{container_file_path}` inside container")
                code = code.replace(input_file_path_str, container_file_path)
        code_path = work_dir / code_filename
        with code_path.open("w", encoding="utf-8") as fout:
            fout.write(code)
        files.append(code_path)
        if mounted_dir:
            shutil.copy(code_path, mounted_dir / code_filename)
            logger.info(f"Copy `{code_path}` to mounted dir `{mounted_dir / code_filename}`")

        if not execute_code:
            outputs.append(f"Code saved to {str(code_path)}\n")
            continue
        import podman as docker

        command = ["timeout", str(timeout), _cmd(lang), os.path.join(container_work_dir, code_filename)]
        try:
            exit_code, output = container.exec_run(command, tty=True)
            logger.info(f"Executed: {command}, Exit code: {exit_code}\n Output: {output}")
        except docker.errors.APIError as e:
            logger.exception(f"Failed to execute code: {e}")
            raise e
        assert isinstance(output, bytes)
        output = output.decode("utf-8")
        output = ANSI_ESCAPE_REGEX.sub("", output)
        if exit_code == 124:
            output += "\n" + "Timeout"
        outputs.append(output)
        if file_output := _get_file_name_from_output(output, work_dir):
            output_files.extend(file_output)

        last_exit_code = exit_code
        if exit_code != 0:
            break

    return CommandLineCodeResult(
        exit_code=last_exit_code,
        output="".join(outputs),
        output_files=output_files,
        code_files=[str(file) for file in files],
    )


# utils:

CODE_BLOCK_PATTERN = r"```[ \t]*(\w+)?[ \t]*\r?\n(.*?)\r?\n[ \t]*```"
UNKNOWN = "unknown"


def infer_lang(code: str) -> str:
    """infer the language for the code.
    TODO: make it robust.
    """
    if code.startswith("python ") or code.startswith("pip") or code.startswith("python3 "):
        return "sh"

    # check if code is a valid python code
    try:
        compile(code, "test", "exec")
        return "python"
    except SyntaxError:
        # not a valid python code
        return UNKNOWN


class CodeBlock(BaseModel):
    """A class that represents a code block."""

    code: str = Field(description="The code to execute.")
    language: str = Field(description="The language of the code.")


class CodeResult(BaseModel):
    """A class that represents the result of a code execution."""

    exit_code: int = Field(description="The exit code of the code execution.")
    output: str = Field(description="The output of the code execution.")
    output_files: list[str] | None = Field(default=None, description="The output files of the code execution.")


class CommandLineCodeResult(CodeResult):
    """(Experimental) A code result class for command line code executor."""

    code_files: list[str] | None = Field(
        default=None,
        description="The file that the executed code block was saved to.",
    )


def extract_code_blocks(message: str) -> List[CodeBlock]:
    """(Experimental) Extract code blocks from a message. If no code blocks are found,
    return an empty list.

    Args:
        message (str): The message to extract code blocks from.

    Returns:
        List[CodeBlock]: The extracted code blocks or an empty list.
    """

    text = message
    match = re.findall(CODE_BLOCK_PATTERN, text, flags=re.DOTALL)
    if not match:
        return []
    code_blocks = []
    for lang, code in match:
        if lang == "":
            lang = infer_lang(code)
        if lang == UNKNOWN:
            lang = ""
        code_blocks.append(CodeBlock(code=code, language=lang))
    return code_blocks


def _cmd(lang: str) -> str:
    if lang in ["python", "Python", "py"]:
        return "python"
    if lang.startswith("python") or lang in ["bash", "sh"]:
        return lang
    if lang in ["shell"]:
        return "sh"
    if lang == "javascript":
        return "node"
    raise ValueError(f"Unsupported language {lang}")


FILENAME_PATTERNS = [
    re.compile(r"^<!-- (filename:)?(.+?) -->", re.DOTALL),
    re.compile(r"^/\* (filename:)?(.+?) \*/", re.DOTALL),
    re.compile(r"^// (filename:)?(.+?)$", re.DOTALL),
    re.compile(r"^# (filename:)?(.+?)$", re.DOTALL),
]
# //


def _get_file_name_from_content(code: str, workspace_path: Path) -> Optional[str]:
    first_line = code.split("\n")[0].strip()
    # TODO - support other languages
    for pattern in FILENAME_PATTERNS:
        matches = pattern.match(first_line)
        if matches is not None:
            filename = matches.group(2).strip()

            # Handle relative paths in the filename
            path = Path(filename)
            if not path.is_absolute():
                path = workspace_path / path
            path = path.resolve()
            # Throws an error if the file is not in the workspace
            relative = path.relative_to(workspace_path.resolve())
            return str(relative)
    return None


def _get_file_name_from_output(output: str, workspace_path: Path) -> Optional[list[str]]:
    pattern = r"\S+\.png|jpg|jpeg"  # TODO support more file types
    compiled_pattern = re.compile(pattern)
    matches = compiled_pattern.findall(output)
    filenames = []
    for match in matches:
        path = Path(match.strip("'").strip("`").strip('"'))
        if not path.is_absolute():
            path = workspace_path / path
        filenames.append(str(path))
    return filenames if len(filenames) > 0 else None


def silence_pip(code: str, lang: str) -> str:
    """Apply -qqq flag to pip install commands."""
    if lang == "python":
        regex = r"^! ?pip install"
    elif lang in ["bash", "shell", "sh", "pwsh", "powershell", "ps1"]:
        regex = r"^pip install"
    else:
        return code

    # Find lines that start with pip install and make sure "-qqq" flag is added.
    lines = code.split("\n")
    for i, line in enumerate(lines):
        # use regex to find lines that start with pip install.
        match = re.search(regex, line)
        if match is not None:
            if "-qqq" not in line:
                lines[i] = line.replace(match.group(0), match.group(0) + " -qqq")
    return "\n".join(lines)


def maybe_get_code_sandbox(exp_path: str) -> ContainerExecutor | None:
    code_path = os.path.join(exp_path, "code")
    os.makedirs(code_path, exist_ok=True)
    try:
        code_sandbox = ContainerExecutor(work_dir=code_path)
    except Exception as e:
        logger.exception(f"Failed to create code sandbox: {e}")
        code_sandbox = None
    return code_sandbox


def init_code_sandbox(exp_path: str, no_deps: bool = False) -> None:
    code_path = os.path.join(exp_path, "code")
    os.makedirs(code_path, exist_ok=True)
    container_name = exp_path.replace("/", "-")
    ContainerExecutor(work_dir=code_path, container_name=container_name, restart_if_exists=True, no_deps=no_deps)
