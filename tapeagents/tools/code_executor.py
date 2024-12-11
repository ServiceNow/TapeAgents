from typing import Literal

from pydantic import ConfigDict, Field

from tapeagents.core import Action, Observation
from tapeagents.environment import CodeExecutionResult
from tapeagents.tools.base import Tool
from tapeagents.tools.container_executor import CodeBlock, CommandLineCodeResult, ContainerExecutor
from tapeagents.tools.python_interpreter import logger, run_python_code


class PythonCodeAction(Action):
    """
    Action to execute the python code snippet.
    """

    kind: Literal["python_code_action"] = "python_code_action"  # type: ignore
    code: str = Field(
        description="snippet of python code with escaped newlines and quotes to fit json format. Last line should print the result"
    )


class CodeExecutor(Tool):
    """
    Tool to execute the python code snippet.
    """

    action: type[Action] = PythonCodeAction
    observation: type[Observation] = CodeExecutionResult
    cached: bool = True
    _sandbox: ContainerExecutor | None = None

    def run(self, action: PythonCodeAction) -> CodeExecutionResult:
        code = self._add_print_to_last_line(action.code)
        if self._sandbox is None:
            logger.warning(f"Code sandbox is not provided, running code locally!\n{code}")
            obs = self._run_restricted_python(code)
        else:
            result = self._sandbox.execute_code_blocks([CodeBlock(code=code, language="python")])
            result.output = result.output[:1000].strip()
            obs = CodeExecutionResult(result=result)
        return obs

    def _run_restricted_python(self, code: str) -> CodeExecutionResult:
        result, stdout, stderr = run_python_code(code, {})
        output = f"{result[:1000].strip()}\n\nstdout:\n{stdout}\n\nstderr:\n{stderr}"
        return CodeExecutionResult(result=CommandLineCodeResult(output=output, exit_code=0 if not stderr else 1))

    def _add_print_to_last_line(self, python_code: str) -> str:
        lines = python_code.splitlines()
        if "print(" in lines[-1]:
            return python_code
        if " = " in lines[-1]:
            name = lines[-1].split("=")[0].strip()
            lines.append(f"print({name})")
        else:
            lines[-1] = f"print({lines[-1]})"
        return "\n".join(lines)
