from __future__ import annotations

import logging
from typing import Any, Callable, Literal

import jsonref
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel

from tapeagents.core import Action, Observation, Step
from tapeagents.llms import LLMOutput
from tapeagents.steps import ExplainableAction
from tapeagents.tools.base import AsyncBaseTool

logger = logging.getLogger(__name__)


class FunctionSpec(BaseModel):
    """
    A class representing the specification of a function.

    Attributes:
        name (str): The name of the function.
        description (str): A brief description of the function.
        parameters (dict): A dictionary containing the parameters of the function.
    """

    name: str
    description: str
    parameters: dict


class ToolSpec(AsyncBaseTool):
    """
    ToolSpec is a model that represents a tool specification with a type and a function.

    Attributes:
        type (Literal["function"]): The type of the tool, which is always "function".
        function (FunctionSpec): The specification of the function.
    """

    type: Literal["function"] = "function"
    function: FunctionSpec

    def description(self) -> str:
        return f"{self.function.name} - {self.function.description}"

    @classmethod
    def from_function(cls, function: Callable):
        """
        Creates an instance of the class by validating the model from a given function.

        Args:
            function (Callable): The function to be converted and validated.

        Returns:
            (ToolSpec): An instance of the class with the validated model.
        """
        return cls.model_validate(convert_to_openai_tool(function))


class FunctionCall(BaseModel):
    """
    A class representing a function call.

    Attributes:
        name (str): The name of the function being called.
        arguments (Any): The arguments to be passed to the function.
    """

    name: str
    arguments: Any


class ToolCall(BaseModel):
    """
    ToolCall is a model representing a tool call with a specific function, id, and type.

    Attributes:
        function (FunctionCall): The function call associated with the tool.
        id (str): The identifier for the tool call. Defaults to an empty string.
        type (str): The type of the tool call. Defaults to "function".
    """

    function: FunctionCall
    id: str = ""
    type: str = "function"


class ToolCallAction(ExplainableAction):
    kind: Literal["tool_call"] = "tool_call"  # type: ignore
    id: str = ""
    function: FunctionCall


class ToolCalls(Action):
    """Action that wraps one-or-many tool calls.

    We structure this class similar to OpenAI tool calls, but we let function arguments be Any, not just str
    (see `FunctionCall` class)

    Attributes:
        tool_calls (list[ToolCall]): The list of tool calls to be made.
        kind (Literal["assistant"]): The type of step, which is always "assistant".
    """

    tool_calls: list[ToolCall]
    kind: Literal["assistant"] = "assistant"

    @staticmethod
    def from_dicts(dicts: list):
        """
        Create a ToolCalls instance from a list of dictionaries.

        Args:
            dicts (list): A list of dictionaries where each dictionary represents a tool call.

        Returns:
            (ToolCalls): An instance of ToolCalls created from the provided list of dictionaries.
        """
        return ToolCalls.model_validate({"tool_calls": dicts})

    @staticmethod
    def from_llm_output(llm_output: LLMOutput) -> ToolCalls:
        """
        Converts an LLMOutput object to a ToolCalls object.

        Args:
            llm_output (LLMOutput): The output from the language model, which contains tool calls.

        Returns:
            ToolCalls: An object containing a list of ToolCall objects.

        Raises:
            ValueError: If the llm_output does not contain any tool calls.
        """
        if not llm_output.tool_calls:
            raise ValueError()
        tool_calls = [
            ToolCall(
                function=FunctionCall(name=tc.function.name, arguments=tc.function.arguments),
                id=tc.id,
            )
            for tc in llm_output.tool_calls
        ]
        return ToolCalls(tool_calls=tool_calls)


class ToolResult(Observation):
    """
    ToolResult is a subclass of Observation that represents the result of a tool call.

    Attributes:
        content (Any): The content of the tool result.
        tool_call_id (str): The unique identifier for the tool call. Defaults to an empty string.
        kind (Literal["tool"]): The kind of result, which is always "tool". Defaults to "tool".
    """

    content: Any
    tool_call_id: str = ""
    kind: Literal["tool"] = "tool"


def as_openai_tool(action: type[Step] | ToolSpec, decription_chars_limit: int = 1024) -> ToolSpec:
    if isinstance(action, ToolSpec):
        return action
    schema = action.model_json_schema()
    schema: dict = dict(jsonref.replace_refs(schema, proxies=False))  # type: ignore
    schema.pop("$defs", None)
    props = schema["properties"]
    props.pop("metadata", None)
    props.pop("kind", None)
    name = schema["title"]
    description = schema.get("description", "")
    if name.lower().endswith("action"):
        name = name[:-6]  # len("action")
    elif name.lower().endswith("thought"):
        name = f"Produce{name}"
        description = f"Produce {description}"
    if len(description) > decription_chars_limit:  # OAI limit
        description = description[:decription_chars_limit]
        logger.warning(f"Description of the '{name}' tool truncated to {decription_chars_limit} chars")
    return ToolSpec(
        function=FunctionSpec(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": props,
                "required": schema.get("required", []),
            },
        )
    )


def as_function_call(action: Action) -> str:
    args = action.llm_dict()
    name = args.pop("kind")
    return f"{name}({', '.join(args)})"


def as_function_def(action: Action) -> str:
    fdef = ""
    type_aliases = {
        "integer": "int",
        "number": "float",
        "string": "str",
        "boolean": "bool",
        "array": "list",
        "object": "dict",
        "null": "None",
        "any": "Any",
    }
    tool_spec = as_openai_tool(action)
    fdef += f"def {tool_spec['function']['name']}("
    for param, param_spec in tool_spec["function"]["parameters"]["properties"].items():
        ptype = type_aliases.get(param_spec["type"], param_spec["type"])
        fdef += f"{param}: {ptype}, "
    fdef = fdef[:-2] + "):"
    fdef += f'\n    """{tool_spec["function"]["description"]}\n'
    for param, param_spec in tool_spec["function"]["parameters"]["properties"].items():
        fdef += f"    {param}: {param_spec['description']}\n"
    fdef += '    """\n'
    fdef += "    pass"
    return fdef
