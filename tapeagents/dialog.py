import json
from typing import Any, Literal, TypeAlias

from litellm.utils import ChatCompletionMessageToolCall
from pydantic import BaseModel

from .agent import Annotator, ObservationMaker
from .core import (
    Action,
    AgentEvent,
    Jump,
    MakeObservation,
    Observation,
    Pass,
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
    content: str
    kind: Literal["assistant_thought"] = "assistant_thought"


class AssistantStep(Action):
    content: str
    kind: Literal["assistant"] = "assistant"


class ToolCalls(Action):
    tool_calls: list[ChatCompletionMessageToolCall]
    kind: Literal["assistant"] = "assistant"


class ToolResult(Observation):
    content: Any
    tool_call_id: str
    kind: Literal["tool"] = "tool"


DialogStep: TypeAlias = UserStep | AssistantStep | SystemStep | AssistantThought | Jump | Pass
FunctionDialogStep: TypeAlias = DialogStep | ToolCalls | ToolResult


# TODO: define type signature for tools including JSONSchema and etc
class FunctionSpec(BaseModel):
    name: str
    description: str
    parameters: dict


class ToolSpec(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionSpec


class DialogContext(BaseModel):
    # TODO: define type signature for tools including JSONSchema and etc
    tools: list[ToolSpec]


# TODO: call me MessageTape?
Dialog = Tape[DialogContext | None, FunctionDialogStep]


DialogEvent: TypeAlias = AgentEvent[Dialog]


class AnnotatorFreeFormThought(Thought):
    content: str


class AnnotationAction(Action):
    annotation: dict


DialogAnnotatorTape: TypeAlias = Tape[Dialog, AnnotatorFreeFormThought | AnnotationAction]


class DialogAnnotator(Annotator[Dialog, DialogAnnotatorTape]):
    def make_own_tape(self, tape: Dialog) -> DialogAnnotatorTape:
        return DialogAnnotatorTape(context=tape)


class UserModelInstruction(Observation):
    instruction: str


class UserModelFreeFormThought(Observation):
    content: str


UserModelTape = Tape[Dialog, MakeObservation[UserStep] | UserModelFreeFormThought | UserModelInstruction]
UserModelEvent = AgentEvent[UserModelTape]


class UserModel(ObservationMaker[Dialog, UserModelTape]):
    instruction: str

    def make_own_tape(self, tape: Dialog) -> UserModelTape:
        return UserModelTape(context=tape).append(UserModelInstruction(instruction=self.instruction))

    @property
    def signature(self) -> str:
        return json.dumps({"model": "user model", "instruction": self.instruction})
