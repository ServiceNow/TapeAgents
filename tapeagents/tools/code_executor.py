import os
from typing import Literal

from pydantic import ConfigDict, Field

from tapeagents.core import Action, Observation
from tapeagents.environment import CodeExecutionResult
from tapeagents.tools.base import Tool
from tapeagents.tools.container_executor import (
    CodeBlock,
    CommandLineCodeResult,
    execute_code_in_container,
)
from tapeagents.tools.python_interpreter import run_python_code


class PythonCodeAction(Action):
    """
    Action to execute the python code snippet. Can be used to perform calculations, simulations or data processing.
    """

    kind: Literal["python_code_action"] = "python_code_action"  # type: ignore
    name: str = Field(description="name of the program, lowercase, no spaces, ends with .py")
    code: str = Field(
        description="snippet of python code with escaped newlines and quotes to fit json format. Last line should print the result"
    )
    input_files: list[str] = Field(description="list of input file paths to be mounted in the container")


class CodeExecutor(Tool):
    """
    Tool to execute the python code snippet.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    action: type[Action] = PythonCodeAction
    observation: type[Observation] = CodeExecutionResult
    cached: bool = True
    exp_path: str = ""
    max_output_length: int = 3000

    def execute_action(self, action: PythonCodeAction) -> CodeExecutionResult:
        code = self.prepare_code_block(action)
        code_dir = os.path.join(self.exp_path, "code")
        result = execute_code_in_container([CodeBlock(code=code, language="python")], code_dir, action.input_files)
        result.output = result.output[: self.max_output_length].strip()
        obs = CodeExecutionResult(result=result)
        return obs

    def _run_restricted_python(self, code: str) -> CodeExecutionResult:
        result, stdout, stderr = run_python_code(code, {})
        output = f"{result[:self.max_output_length].strip()}\n\nstdout:\n{stdout}\n\nstderr:\n{stderr}"
        return CodeExecutionResult(result=CommandLineCodeResult(output=output, exit_code=0 if not stderr else 1))

    def prepare_code_block(self, action: PythonCodeAction) -> str:
        lines = action.code.splitlines()
        lines = [f"# {action.name}"] + lines
        if "print(" not in lines[-1]:
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
