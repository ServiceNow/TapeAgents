import json
import logging
from abc import ABC, abstractmethod
from typing import Callable, Generic, Literal

from langchain_core.tools import BaseTool, tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import TypeAdapter

from tapeagents.container_executor import CodeBlock, CommandLineCodeResult, ContainerExecutor
from tapeagents.utils import FatalError

from .agent import TapeType
from .core import Action, Observation, Tape
from .dialog_tape import AssistantStep, DialogTape, FunctionCall, ToolCalls, ToolResult, ToolSpec

logger = logging.getLogger(__name__)


class ExternalObservationNeeded(Exception):
    """Environments raise this when they can't make the required observation for an action"""

    def __init__(self, action: Action, *args, **wargs):
        super().__init__(*args, **wargs)
        self.action = action

    def __str__(self) -> str:
        return f"Environment needs external observation for action {self.action}"


class NoActionsToReactTo(Exception):
    """Environments raise this when there are no actions to react to"""

    def __init__(self, *args, **wargs):
        super().__init__(*args, **wargs)


class Environment(ABC, Generic[TapeType]):
    @abstractmethod
    def react(self, tape: TapeType) -> TapeType:
        pass

    def raise_external_observation_needed(self, action: Action):
        raise ExternalObservationNeeded(
            action,
            f"Environment {type(self).__name__} does not known how to produce observations that should follow"
            f" an action of type {type(action).__name__}. Please add obserations to the tape manually.",
        )

    def raise_unexpected_action(self, action: Action):
        raise ValueError(f"Unexpected action type: {action}")


class EmptyEnvironment(Environment):
    def react(self, tape: Tape) -> list[Observation]:
        # TOOD: move prompting to a separate agent?
        action = tape.steps[-1]
        if isinstance(action, AssistantStep):
            self.raise_external_observation_needed(action)
        else:
            self.raise_unexpected_action(action)
        return []


class ToolEnvironment(Environment):
    def __init__(self, tools: list[BaseTool | Callable]):
        self.tools: list[BaseTool] = [t if isinstance(t, BaseTool) else tool(t) for t in tools]  # type: ignore
        self._name2tool = {t.name: t for t in self.tools}

    def get_tool_schemas(self) -> list[ToolSpec]:
        return [TypeAdapter(ToolSpec).validate_python(convert_to_openai_tool(tool)) for tool in self.tools]

    def get_tool_schema_dicts(self) -> list[dict]:
        return [t.model_dump() for t in self.get_tool_schemas()]

    def react(self, tape: DialogTape) -> DialogTape:
        orphan_actions = []
        for i in reversed(range(len(tape))):
            if isinstance(tape.steps[i], Observation):
                break
            if isinstance(tape.steps[i], Action):
                orphan_actions.append(tape.steps[i])
        if not orphan_actions:
            raise NoActionsToReactTo("No actions to react to")
        for action in orphan_actions:
            if isinstance(action, ToolCalls):
                for tc in action.tool_calls:
                    if not isinstance(tc.function, FunctionCall) or not isinstance(tc.function.name, str):
                        raise ValueError("Tool call must be Function and must have a name")
                    tool = self._name2tool[tc.function.name]
                    args = tc.function.arguments
                    if isinstance(args, str):
                        args = json.loads(args)
                    try:
                        content = tool.run(tool_input=args)
                    except FatalError as e:
                        raise e
                    except Exception as e:
                        content = str(e)
                    tape = tape.append(ToolResult(content=content, tool_call_id=tc.id))
            elif isinstance(action, AssistantStep):
                self.raise_external_observation_needed(action)
            else:
                self.raise_unexpected_action(action)

        return tape


class ExecuteCode(Action):
    kind: Literal["execute_code"] = "execute_code"
    code: list[CodeBlock]

    def llm_view(self, indent: int | None = 2) -> str | list[dict]:
        blocks = [f"```{block.language}\n{block.code}\n```" for block in self.code]
        return "\n\n".join(blocks)


class CodeExecutionResult(Observation):
    kind: Literal["code_execution_result"] = "code_execution_result"
    result: CommandLineCodeResult


class CodeExecutionEnvironment(Environment):
    """
    Environment for the team agents
    The only action that the environment can perform is to execute the code blocks
    """

    def __init__(self, container_executor: ContainerExecutor):
        self.container_executor = container_executor

    def react(self, tape: Tape) -> Tape:
        match step := tape.steps[-1]:
            case ExecuteCode():
                result = self.container_executor.execute_code_blocks(step.code)
                return tape.append(CodeExecutionResult(result=result))
            case _:
                return tape
