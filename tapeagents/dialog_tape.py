import json
from typing import Any, Callable, Literal, TypeAlias

from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel

from .agent import Annotator, ObservationMaker
from .core import (
    Action,
    AgentEvent,
    Call,
    FinalStep,
    MakeObservation,
    Observation,
    Pass,
    Respond,
    SetNextNode,
    Tape,
    Thought,
)


class SystemStep(Observation):
    content: str
    kind: Literal["system"] = "system"


class UserStep(Observation):
    content: str
    kind: Literal["user"] = "user"


class AssistantThought(Thought):
    content: Any
    kind: Literal["assistant_thought"] = "assistant_thought"


class AssistantStep(Action):
    content: str
    kind: Literal["assistant"] = "assistant"


class FunctionCall(BaseModel):
    name: str
    arguments: Any


class ToolCall(BaseModel):
    function: FunctionCall
    id: str = ""


class ToolCalls(Action):
    """Action that wraps one-or-many tool calls.
    
    We structure this class similar to OpenAI tool calls, but we let function arguments be Any, not just str
    (see `FunctionCall` class)
    
    """
    tool_calls: list[ToolCall]
    kind: Literal["assistant"] = "assistant"
    
    @staticmethod
    def from_dicts(dicts: list):
        return ToolCalls.model_validate({"tool_calls": dicts})
        

class ToolResult(Observation):
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


# TODO: define type signature for tools including JSONSchema and etc
class FunctionSpec(BaseModel):
    name: str
    description: str
    parameters: dict


class ToolSpec(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionSpec

    @classmethod
    def from_function(cls, function: Callable):
        return cls.model_validate(convert_to_openai_tool(function))


class DialogContext(BaseModel):
    """
    Context for dialog agents, containing tools and other information.
    """

    # TODO: define type signature for tools including JSONSchema and etc
    tools: list[ToolSpec]


# Tape for dialog agents, containing dialog steps and optional context
DialogTape = Tape[DialogContext | None, DialogStep]


DialogEvent: TypeAlias = AgentEvent[DialogTape]


class AnnotatorFreeFormThought(Thought):
    content: str


class AnnotationAction(Action):
    annotation: dict


DialogAnnotatorTape: TypeAlias = Tape[DialogTape, AnnotatorFreeFormThought | AnnotationAction]


class DialogAnnotator(Annotator[DialogTape, DialogAnnotatorTape]):
    def make_own_tape(self, tape: DialogTape) -> DialogAnnotatorTape:
        return DialogAnnotatorTape(context=tape)


class UserModelInstruction(Observation):
    instruction: str


class UserModelFreeFormThought(Observation):
    content: str


UserModelTape = Tape[DialogTape, MakeObservation[UserStep] | UserModelFreeFormThought | UserModelInstruction]
UserModelEvent = AgentEvent[UserModelTape]


class UserModel(ObservationMaker[DialogTape, UserModelTape]):
    instruction: str

    def make_own_tape(self, tape: DialogTape) -> UserModelTape:
        return UserModelTape(context=tape).append(UserModelInstruction(instruction=self.instruction))

    @property
    def signature(self) -> str:
        return json.dumps({"model": "user model", "instruction": self.instruction})
