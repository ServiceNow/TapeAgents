import logging
import os
import subprocess
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field

from tapeagents.core import Action, Observation
from tapeagents.environment import CodeExecutionResult
from tapeagents.tools.base import Tool
from tapeagents.tools.container_executor import (
    CodeBlock,
    CommandLineCodeResult,
    _get_file_name_from_content,
    execute_code_in_container,
)

logger = logging.getLogger(__name__)


class PythonCodeAction(Action):
    """
    Action to execute the python code snippet. Can be used to perform calculations, simulations or data processing.
    """

    kind: Literal["python_code_action"] = "python_code_action"  # type: ignore
    name: str = Field(description="name of the program, lowercase, no spaces, unique, ends with .py")
    code: str = Field(
        description="snippet of python code with escaped newlines and quotes to fit json format. Last line should print the result"
    )
    input_files: list[str] = Field(description="list of input file paths to be mounted in the container")


class CodeExecutor(Tool):
    """
    Executes the python code snippet
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    action: type[Action] = PythonCodeAction
    observation: type[Observation] = CodeExecutionResult
    cached: bool = True
    exp_path: str = ""
    max_output_length: int = 3000
    reuse_computer_container: bool = False
    mounted_dir: str = ""
    container_work_dir: str = "/workspace"

    def execute_action(self, action: PythonCodeAction) -> CodeExecutionResult:
        code = self.prepare_code(action)
        code_dir = os.path.join(self.exp_path, "code")
        container_name = (
            os.environ["COMPUTER_CONTAINER_NAME"] if self.reuse_computer_container else self.exp_path.replace("/", "-")
        )
        logger.info(f"Executing code in container {container_name}")
        result = execute_code_in_container(
            [CodeBlock(code=code, language="python")],
            work_dir=code_dir,
            input_files=action.input_files,
            container_name=container_name,
            mounted_dir=self.mounted_dir,
            container_work_dir=self.container_work_dir,
        )
        result.output = result.output[: self.max_output_length].strip()
        obs = CodeExecutionResult(result=result)
        return obs

    def prepare_code(self, action: PythonCodeAction) -> str:
        lines = action.code.splitlines()
        if len(lines) == 1 and "\\n" in lines[0]:
            lines = lines[0].split("\\n")
        lines = [f"# {action.name}"] + lines
        if "print(" not in lines[-1] and "break" not in lines[-1]:
            if " = " in lines[-1]:
                name = lines[-1].split("=")[0].strip()
                lines.append(f"print({name})")
            else:
                lines[-1] = f"print({lines[-1]})"
        return "\n".join(lines)

    def trim_output(self, output: str) -> str:
        if len(output) > self.max_output_length:
            half = self.max_output_length // 2
            output = f"{output[:half]} ... {output[-half:]}"
        return output


class CodeExecutorWithApproval(CodeExecutor):
    """
    Tool to execute the python code snippet.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    action: type[Action] = PythonCodeAction
    observation: type[Observation] = CodeExecutionResult
    cached: bool = True
    exp_path: str = ""
    max_output_length: int = 3000
    _chat = None

    def execute_action(self, action: PythonCodeAction) -> CodeExecutionResult:
        code = self.prepare_code(action)
        code_dir = os.path.join(self.exp_path, "code")
        fname = _get_file_name_from_content(code, Path(code_dir))
        fpath = os.path.join(code_dir, fname)
        self._chat.add_message(role="assistant", msg=f"Requesting permission to run the code(Y/n):\n\n{code}")
        self._chat.wait_for_user_message()
        user_response = self._chat.messages[-1]["message"]
        if user_response.lower() != "y":
            return CodeExecutionResult(result=CommandLineCodeResult(output="Code execution denied", exit_code=1))
        else:
            self._chat.add_message(role="assistant", msg="Running the code...")
        with open(fpath, "w") as f:
            f.write(code)
        try:
            process = subprocess.run(["python", fpath], capture_output=True, text=True)
            result = CommandLineCodeResult(output=process.stdout + process.stderr, exit_code=process.returncode)
        except Exception as e:
            result = CommandLineCodeResult(output=str(e), exit_code=1)
        result.output = result.output[: self.max_output_length].strip()
        obs = CodeExecutionResult(result=result)
        return obs
