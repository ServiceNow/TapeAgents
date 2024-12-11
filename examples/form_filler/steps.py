import json
from typing import Literal, TypeAlias, Union

from pydantic import BaseModel, Field, JsonValue

from tapeagents.core import Action, LLMOutputParsingFailureAction, Observation, SetNextNode, StopStep, Thought
from tapeagents.dialog_tape import AssistantStep, UserStep

from .error import _FormFillerStateError
from .schema import FunctionSchema
from .types import FunctionName, ParameterName


class FunctionCandidate(BaseModel):
    """
    A candidate for a function that was resolved from a user message.
    Attributes:
        function: The name of the function.
        short_description: The short description of the function.
    """

    function: str
    short_description: str


class ExitStep(AssistantStep, StopStep):
    kind: Literal["exit_step"] = "exit_step"

    def llm_view(self) -> str:
        return "Exiting the conversation."


class ResolveFunction(Action):
    query: str | None
    kind: Literal["resolve_function_action"] = "resolve_function_action"

    def llm_view(self) -> str:
        return "Looking for function candidates" + f" for the user query `{self.query}`." if self.query else "."


class FunctionCandidates(Observation):
    candidates: list[FunctionCandidate]
    kind: Literal["function_candidates"] = "function_candidates"

    def llm_view(self) -> str:
        return "These are the most relevant actions I can do:" + "\n".join(
            f"- {candidate.function}: {candidate.short_description}" for candidate in self.candidates
        )


class InspectFunction(Action):
    function: str
    kind: Literal["inspect_function_action"] = "inspect_function_action"

    def llm_view(self) -> str:
        return f"Inspecting the function `{self.function}`."


class RequestFunction(Thought):
    functions: None | list[FunctionName] = Field(
        default=None,
        description="The list of functions to request. If the field is None, the user can select any function.",
    )
    kind: Literal["request_function_thought"] = "request_function_thought"

    def llm_view(self) -> str:
        view = "I should now ask the user to select an action."
        if self.functions:
            view += f"\nAvailable actions: {', '.join(self.functions)}."
        return view


class CallFunction(Action):
    kind: Literal["call_function_action"] = "call_function_action"
    function: FunctionName

    def llm_view(self) -> str:
        return f"Calling the function `{self.function}`."


class FunctionResult(Observation):
    kind: Literal["function_result"] = "function_result"
    function: FunctionName
    result: JsonValue

    def llm_view(self) -> str:
        return f"The results are: {json.dumps(self.result, indent=2)}"


class RequestFunctionParameters(Thought):
    function: FunctionName
    parameters: list[ParameterName]
    kind: Literal["request_function_parameters_thought"] = "request_function_parameters_thought"

    def llm_view(self) -> FunctionName:
        return f"I should ask for {', '.join(self.parameters)}."


class RequestFunctionCallConfirmation(Thought):
    function: FunctionName
    kind: Literal["request_function_call_confirmation_thought"] = "request_function_call_confirmation_thought"

    def llm_view(self) -> str:
        return f"I should now confirm with the user before doing `{self.function}`."


class RequestExitConfirmation(Thought):
    kind: Literal["request_exit_confirmation_thought"] = "request_exit_confirmation_thought"

    def llm_view(self) -> str:
        return "I should ask for confirmation before exiting the conversation."


class UpdateFunctionParameters(Thought):
    function: FunctionName
    assign: dict[ParameterName, JsonValue] = Field(
        default_factory=dict,
        description="The assignments to update parameters with.",
    )
    skip: list[ParameterName] = Field(
        default_factory=list,
        description="The names of parameters to skip.",
    )
    kind: Literal["update_function_parameters_thought"] = "update_function_parameters_thought"

    def llm_view(self) -> str:
        view = []
        if self.assign:
            for p_name, p_value in self.assign.items():
                view.append(f"I note that: {p_name}={str(p_value)}.")
        if self.skip:
            for p_name in self.skip:
                view.append(f"I note that the user skipped {p_name}.")
        return " ".join(view)


class GatherValuesThought(Thought):
    """
    Used in chain of prompting (e.g., multi-node prompter) where each thought is a small step toward generating an agent message.
    This thought is associated with extracting parameters from the user's message.
    """

    kind: Literal["gather_values_thought"] = "gather_values_thought"
    function: FunctionName
    parameters: dict[ParameterName, JsonValue] = Field(
        default_factory=dict,
        description="The extracted parameters.",
    )

    def llm_view(self) -> str:
        view = []
        for p_name, p_value in self.parameters.items():
            view.append(f"(intermediate) I extracted {p_name}={str(p_value)}.")
        return "\n".join(view)


class VerifyValuesThought(Thought):
    """
    Used in chain of prompting (e.g., multi-node prompter) where each thought is a small step toward generating an agent message.
    This thought is associated with verifying parameters, extracted from the user's message.
    """

    kind: Literal["verify_values_thought"] = "verify_values_thought"
    function: FunctionName
    parameters: dict[ParameterName, JsonValue] = Field(
        default_factory=dict,
        description="The verified parameters.",
    )

    def llm_view(self) -> str:
        view = []
        for p_name, value_dict in self.parameters.items():
            view.append(f"(intermediate) I checked {p_name}={str(value_dict['value'])} is {value_dict['status']}")
            if explanation := value_dict.get("explanation", None):
                view[-1] += f" because {explanation}"
            else:
                view[-1] += "."
        return "\n".join(view)


class RefuseInvalidFunctionParameterValue(Thought):
    function: FunctionName
    parameter: ParameterName
    parameter_value: None | JsonValue = Field(
        default=None,
        description="The value the user tried to set for the parameter.",
    )
    kind: Literal["refuse_invalid_function_parameter_value_thought"] = "refuse_invalid_function_parameter_value_thought"

    def llm_view(self) -> str:
        if self.parameter_value:
            return f"I should inform the user that value '{str(self.parameter_value)}' is invalid for parameter {self.parameter}."
        else:
            return f"I should inform the user their provided value is invalid for parameter {self.parameter}."


class RefuseInexistentFunctionParameter(Thought):
    kind: Literal["refuse_inexistent_function_parameter_thought"] = "refuse_inexistent_function_parameter_thought"
    function: FunctionName

    def llm_view(self) -> str:
        return "I should inform the user that this parameter is not available."


class RefuseInvalidFunctionParameterSkip(Thought):
    function: FunctionName
    parameter: ParameterName
    kind: Literal["refuse_invalid_function_parameter_skip_thought"] = "refuse_invalid_function_parameter_skip_thought"

    def llm_view(self) -> str:
        return f"I should inform the user that parameter {self.parameter} cannot be skipped."


class RefuseInexistentFunction(Thought):
    kind: Literal["refuse_inexistent_function_thought"] = "refuse_inexistent_function_thought"

    def llm_view(self) -> str:
        return "I should inform the user that I cannot do what they want."


class RefuseToEngage(Thought):
    kind: Literal["refuse_to_engage_thought"] = "refuse_to_engage_thought"

    def llm_view(self) -> str:
        return "Part of the user message is irrelevant."


class AnswerFromFunctionSchema(Thought):
    function: FunctionName
    answerable_questions: list[str] = Field(
        default=[], description="List of questions derived from the user message that should be answered."
    )
    kind: Literal["answer_from_function_schema_thought"] = "answer_from_function_schema_thought"

    def llm_view(self) -> str:
        return "I should answer the user based on the available documentation."


class NoAnswerFromFunctionSchema(Thought):
    function: FunctionName
    declined_questions: list[str] = Field(
        default=[], description="List of questions derived from the user message that should be declined."
    )
    kind: Literal["no_answer_in_function_schema_thought"] = "no_answer_in_function_schema_thought"

    def llm_view(self) -> str:
        return "I should inform the user that their question cannot be answered based on the available documentation."


class Exit(StopStep):
    kind: Literal["exit_action"] = "exit_action"

    def llm_view(self) -> str:
        return "Exiting the conversation."


FormFillerStep = Union[
    UserStep,
    AssistantStep,
    ResolveFunction,
    FunctionCandidates,
    InspectFunction,
    FunctionSchema,
    RequestFunction,
    CallFunction,
    FunctionResult,
    RequestFunctionParameters,
    RequestFunctionCallConfirmation,
    RequestExitConfirmation,
    UpdateFunctionParameters,
    GatherValuesThought,
    VerifyValuesThought,
    RefuseInvalidFunctionParameterValue,
    RefuseInvalidFunctionParameterSkip,
    RefuseInexistentFunction,
    RefuseInexistentFunctionParameter,
    RefuseToEngage,
    AnswerFromFunctionSchema,
    NoAnswerFromFunctionSchema,
    Exit,
    LLMOutputParsingFailureAction,
    SetNextNode,
    StopStep,
    _FormFillerStateError,
]

I_SHOULD_STEPS = Union[
    RequestFunction,
    RequestFunctionParameters,
    RequestFunctionCallConfirmation,
    RequestExitConfirmation,
    AnswerFromFunctionSchema,
    NoAnswerFromFunctionSchema,
    RefuseInexistentFunction,
    RefuseInexistentFunctionParameter,
    RefuseInvalidFunctionParameterValue,
    RefuseInvalidFunctionParameterSkip,
    RefuseToEngage,
]
I_NOTE_STEPS: TypeAlias = UpdateFunctionParameters
ACTION_STEPS = (AssistantStep, ResolveFunction, InspectFunction, CallFunction, Exit)
