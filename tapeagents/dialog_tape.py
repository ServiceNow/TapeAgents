"""
Types and classes for dialog tapes and annotators.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, TypeAlias

from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel

from .agent import Annotator
from .core import (
    Action,
    AgentEvent,
    Call,
    FinalStep,
    LLMOutput,
    Observation,
    Pass,
    Respond,
    SetNextNode,
    Tape,
    Thought,
)


class SystemStep(Observation):
    """
    Step rendered into system message of the prompt.

    Attributes:
        content (str): The content of the system step.
        kind (Literal["system"]): A literal indicating the type of step, which is always "system".
    """

    content: str
    kind: Literal["system"] = "system"


class UserStep(Observation):
    """
    Represents a step taken by a user in a dialog.

    Attributes:
        content (str): The content of the user's step.
        kind (Literal["user"]): The type of step, which is always "user".
    """

    content: str
    kind: Literal["user"] = "user"


class AssistantThought(Thought):
    """
    Represents a thought generated by an assistant.

    Attributes:
        content (Any): The content of the assistant's thought.
        kind (Literal["assistant_thought"]): A literal string indicating the type of thought.
    """

    content: Any
    kind: Literal["assistant_thought"] = "assistant_thought"


class AssistantStep(Action):
    """
    Represents a step taken by an assistant in a dialog.

    Attributes:
        content (str): The content of the assistant's response.
        kind (Literal["assistant"]): The type of step, which is always "assistant".
    """

    content: str
    kind: Literal["assistant"] = "assistant"


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


DialogStep: TypeAlias = (
    # observations
    UserStep
    | ToolResult
    | SystemStep
    # thoughts
    | AssistantThought
    | SetNextNode
    | Pass
    | Call
    | Respond
    # actions
    | FinalStep
    | AssistantStep
    | ToolCalls
)
"""Type alias for dialog steps."""


# TODO: define type signature for tools including JSONSchema and etc
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


class ToolSpec(BaseModel):
    """
    ToolSpec is a model that represents a tool specification with a type and a function.

    Attributes:
        type (Literal["function"]): The type of the tool, which is always "function".
        function (FunctionSpec): The specification of the function.
    """

    type: Literal["function"] = "function"
    function: FunctionSpec

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


class DialogContext(BaseModel):
    """
    Context for dialog agents, containing tools and other information.
    """

    # TODO: define type signature for tools including JSONSchema and etc
    tools: list[ToolSpec]


# Tape for dialog agents, containing dialog steps and optional context
DialogTape = Tape[DialogContext | None, DialogStep]
"""Type alias for dialog tapes."""


DialogEvent: TypeAlias = AgentEvent[DialogTape]
"""Type alias for dialog events."""


class AnnotatorFreeFormThought(Thought):
    """
    AnnotatorFreeFormThought is a subclass of Thought that represents a free-form thought provided by an annotator.

    Attributes:
        kind (Literal["annotator_free_form_thought"]): A constant string that identifies the type of thought.
        content (str): The content of the free-form thought provided by the annotator.
    """

    kind: Literal["annotator_free_form_thought"] = "annotator_free_form_thought"
    content: str


class AnnotationAction(Action):
    """
    AnnotationAction is a subclass of Action that represents an action produced by an annotator.

    Attributes:
        kind (Literal["annotation_action"]): A string literal indicating the type of action.
        annotation (dict): A dictionary containing annotation data.
    """

    kind: Literal["annotation_action"] = "annotation_action"
    annotation: dict


DialogAnnotatorTape: TypeAlias = Tape[DialogTape, AnnotatorFreeFormThought | AnnotationAction]
"""Type alias for dialog annotator tapes."""


class DialogAnnotator(Annotator[DialogTape, DialogAnnotatorTape]):
    """
    DialogAnnotator is a class that extends the Annotator agent with specific types DialogTape and DialogAnnotatorTape.

    Methods:
        make_own_tape(tape: DialogTape) -> DialogAnnotatorTape:
            Creates and returns a DialogAnnotatorTape instance using the provided DialogTape instance.
    """

    def make_own_tape(self, tape: DialogTape) -> DialogAnnotatorTape:
        """
        Creates a DialogAnnotatorTape instance using given DialogTape as context.

        Args:
            tape (DialogTape): The DialogTape instance to be converted.

        Returns:
            DialogAnnotatorTape: A new instance of DialogAnnotatorTape with the provided context.
        """
        return DialogAnnotatorTape(context=tape)
