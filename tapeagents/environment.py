"""
Base classes for environments that execute actions and produce observations
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Callable, Generic, Literal

from langchain_core.tools import BaseTool as LangchainBaseTool, tool as tool_wrapper
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import TypeAdapter

from tapeagents.agent import TapeType
from tapeagents.core import Action, LLMOutputParsingFailureAction, Observation, Tape, last_actions
from tapeagents.dialog_tape import AssistantStep, DialogTape, UserStep
from tapeagents.tool_calling import FunctionCall, ToolCalls, ToolResult, ToolSpec
from tapeagents.tools.base import AsyncBaseTool, BaseTool, StatefulTool, Tool
from tapeagents.tools.container_executor import CodeBlock, CommandLineCodeResult, ContainerExecutor
from tapeagents.utils import FatalError
from tapeagents.view import defaultdict

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
    def initialize(self):
        pass

    def start_task(self, task: dict) -> dict:
        """
        Start a new task in the environment.
        This method should be overridden by subclasses to implement task-specific initialization.
        """
        return {}

    @abstractmethod
    def react(self, tape: TapeType) -> TapeType:
        pass

    def step(self, action: Action) -> Observation:
        raise NotImplementedError

    def raise_external_observation_needed(self, action: Action):
        raise ExternalObservationNeeded(
            action,
            f"Environment {type(self).__name__} does not known how to produce observations that should follow"
            f" an action of type {type(action).__name__}. Please add obserations to the tape manually.",
        )

    def raise_unexpected_action(self, action: Action):
        raise ValueError(f"Unexpected action type: {action}")

    def actions(self) -> tuple[type[Action], ...]:
        return tuple()

    def tools_description(self) -> str:
        desc_list = [f"{a.__class__.__name__} - {a.__doc__ or '[no description]'}" for a in self.actions()]
        return "\n".join(f"- {desc}" for desc in desc_list)

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass


class AsyncEnvironment(Environment):
    async def ainitialize(self):
        pass

    async def areact(self, tape: TapeType) -> TapeType:
        raise NotImplementedError

    async def astep(self, action: Action) -> Observation:
        raise NotImplementedError

    async def areset(self) -> None:
        pass

    async def aclose(self) -> None:
        pass


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
    def __init__(self, tools: list[LangchainBaseTool | Callable]):
        self.tools: list[LangchainBaseTool] = [
            t if isinstance(t, LangchainBaseTool) else tool_wrapper(t) for t in tools
        ]  # type: ignore
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
    kind: Literal["execute_code"] = "execute_code"  # type: ignore
    code: list[CodeBlock]

    def llm_view(self, indent: int | None = 2) -> str | list[dict]:
        blocks = [f"```{block.language}\n{block.code}\n```" for block in self.code]
        return "\n\n".join(blocks)


class CodeExecutionResult(Observation):
    kind: Literal["code_execution_result"] = "code_execution_result"  # type: ignore
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


class ToolCollectionEnvironment(AsyncEnvironment):
    def __init__(
        self,
        tools: list[BaseTool],
        loop_detection: bool = False,
        loop_warning_after_n_steps: int = 3,
        loop_warning: str = "You seem to be stuck producing the same action. Consider a new approach and avoid repeating previously attempted ineffective steps.",
    ) -> None:
        if loop_warning_after_n_steps <= 0:
            raise ValueError("loop_warning_after_n_steps must be positive")

        super().__init__()
        self.tools = tools
        self.action_map = {tool.action: tool for tool in tools if isinstance(tool, Tool)}
        for tool in tools:
            if isinstance(tool, StatefulTool):
                self.action_map |= {action: tool for action in tool.actions}
        self.loop_detection = loop_detection
        self.loop_warning_after_n_steps = loop_warning_after_n_steps
        self.loop_warning = loop_warning

    def actions(self) -> tuple[type[Action] | ToolSpec, ...]:
        return tuple(self.action_map.keys())

    def tools_description(self) -> str:
        desc_list = [tool.description() for tool in self.tools]
        return "\n".join(f"- {desc}" for desc in desc_list)

    def react(self, tape: Tape) -> Tape:
        if self.loop_detection and self.loop_detected(tape):
            logger.warning(f"Loop detected in tape: {tape}")
            obs = UserStep(content=self.loop_warning)
            return tape.append(obs)
        for action in last_actions(tape):
            observation = self.step(action)
            tape = tape.append(observation)
        return tape

    def loop_detected(self, tape: Tape) -> bool:
        actions = last_actions(tape)
        all_actions_counter = defaultdict(int)
        for step in tape.steps:
            if isinstance(step, Action):
                all_actions_counter[step.llm_view()] += 1
        for last_action in actions:
            n_args = len(last_action.llm_dict())
            logger.info(f"Action {last_action.kind} has {n_args} args")
            if n_args < 2:
                # skip action without args that only has kind field
                continue
            cnt = all_actions_counter[last_action.llm_view()]
            if cnt >= self.loop_warning_after_n_steps:
                logger.warning(f"Loop, action {last_action.kind} repeated {cnt} times")
                return True
        return False

    def step(self, action: Action) -> Observation:
        t = time.perf_counter()
        action_type = type(action)
        if isinstance(action, LLMOutputParsingFailureAction):
            return UserStep(content="Try again")
        if action_type not in self.action_map:
            raise Exception(f"Unknown action: {action_type}")
        tool = self.action_map[action_type]
        observation = tool.run(action)
        observation.metadata.other["action_execution_time"] = time.perf_counter() - t
        observation.metadata.other["action_kind"] = action.kind
        return observation

    def reset(self) -> None:
        for tool in self.tools:
            tool.reset()

    def close(self) -> None:
        for tool in self.tools:
            tool.close()

    async def areact(self, tape: Tape) -> Tape:
        for action in last_actions(tape):
            observation = await self.astep(action)
            tape = tape.append(observation)
        return tape

    async def astep(self, action: Action) -> Observation:
        t = time.perf_counter()
        action_type = type(action)
        if isinstance(action, LLMOutputParsingFailureAction):
            return UserStep(content="Try again")
        if action_type not in self.action_map:
            raise Exception(f"Unknown action: {action_type}")
        tool = self.action_map[action_type]
        if isinstance(tool, AsyncBaseTool):
            observation = await tool.arun(action)
        else:
            logger.warning(f"Tool {tool} is not async and could slowdown the rollouts!")
            observation = tool.run(action)
        observation.metadata.other["action_execution_time"] = time.perf_counter() - t
        observation.metadata.other["action_kind"] = action.kind
        return observation

    async def areset(self) -> None:
        for tool in self.tools:
            if not isinstance(tool, AsyncBaseTool):
                logger.warning(f"Tool {tool} is not async and could slowdown the rollouts!")
                tool.reset()
            else:
                await tool.areset()

    async def aclose(self) -> None:
        for tool in self.tools:
            if not isinstance(tool, AsyncBaseTool):
                logger.warning(f"Tool {tool} is not async and could slowdown the rollouts!")
                tool.close()
            else:
                await tool.aclose()
