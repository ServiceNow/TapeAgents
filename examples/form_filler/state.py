import logging
from typing import Literal

from pydantic import BaseModel, Field, JsonValue
from typing_extensions import Self

from tapeagents.core import LLMOutputParsingFailureAction, SetNextNode, StopStep
from tapeagents.dialog_tape import AssistantStep, UserStep

from .error import (
    FormFillerStateError,
    InvalidFunctionSchemaError,
    UnknownFunctionError,
    UnknownFunctionParameterError,
    UnknownFunctionSchemaError,
    ValidFunctionParameterSkipError,
    ValidFunctionParameterValueError,
)
from .schema import FunctionSchema
from .steps import (
    AnswerFromFunctionSchema,
    CallFunction,
    FormFillerStep,
    FunctionCandidates,
    FunctionResult,
    GatherValuesThought,
    InspectFunction,
    NoAnswerFromFunctionSchema,
    RefuseInexistentFunction,
    RefuseInexistentFunctionParameter,
    RefuseInvalidFunctionParameterSkip,
    RefuseInvalidFunctionParameterValue,
    RefuseToEngage,
    RequestExitConfirmation,
    RequestFunction,
    RequestFunctionCallConfirmation,
    RequestFunctionParameters,
    ResolveFunction,
    UpdateFunctionParameters,
    VerifyValuesThought,
)
from .tape import FormFillerTape
from .types import FunctionName, ParameterName

logger = logging.getLogger(__name__)


class FormFillerState(BaseModel):
    """
    The state of the agent.
    Attributes:
        functions (dict[FunctionName, None]): The functions that are known to the agent (default: empty dict).
        function_schemas (dict[FunctionName, FunctionSchema]): The function schemas that are known to the agent
            (default: empty dict).
        function_parameters_filled (dict[FunctionName, dict[ParameterName, JsonValue]]): The function parameters
            that have been filled (default: empty dict).
        function_parameters_skipped (dict[FunctionName, list[ParameterName]]): The function parameters
            that have been skipped (default: empty dict).
        function_call_results (dict[FunctionName, JsonValue]): The results of the function calls
            that have been made (default: empty dict).
    """

    messages: list[UserStep | AssistantStep] = Field(
        default_factory=list, description="The messages that have been exchanged between the user and the assistant."
    )
    functions: dict[FunctionName, None] = Field(
        default_factory=dict, description="The functions that are known to the agent."
    )
    function_schemas: dict[FunctionName, FunctionSchema] = Field(
        default_factory=dict, description="The function schemas that are known to the agent."
    )
    function_parameters_filled: dict[FunctionName, dict[ParameterName, JsonValue]] = Field(
        default_factory=dict,
        alias="function_parameters",
        description="The function parameters that have been filled.",
    )
    function_parameters_skipped: dict[FunctionName, list[ParameterName]] = Field(
        default_factory=dict,
        description="The function parameters that have been skipped.",
    )
    raw_function_parameters: dict[FunctionName, dict[ParameterName, JsonValue]] = Field(
        default_factory=dict,
        alias="raw_function_parameters",
        description="The extracted function parameters that have not been verified yet.",
    )
    function_parameters_verified: dict[FunctionName, dict[ParameterName, JsonValue]] = Field(
        default_factory=dict,
        description="The function parameters that have been verified.",
    )
    function_call_results: dict[FunctionName, JsonValue] = Field(
        default_factory=dict,
        description="The results of the function calls that have been made.",
    )

    def add_function(self: Self, function: FunctionName) -> Self:
        return self.model_copy(update={"functions": self.functions | {function: None}})

    def add_function_schema(self: Self, function: FunctionName, schema: None | FunctionSchema) -> Self:
        """
        Adds a function schema to the state for the given function, overwriting any existing schema
        for it.
        The schema is assumed to be valid. No validation is performed.
        Args:
            function (FunctionName): The function name.
            schema (None | FunctionSchema): The function schema.
        """
        if isinstance(schema, FunctionSchema):
            return self.model_copy(update={"function_schemas": self.function_schemas | {function: schema}})
        elif schema is None:
            return self.model_copy()
        else:
            raise ValueError(f"Invalid schema: {schema}")

    def check_if_function_known(self: Self, function: FunctionName) -> Literal[True] | FormFillerStateError:
        """
        Checks that the function is known
        """
        if function in self.functions:
            return True
        return UnknownFunctionError(function=function, message=f"No function named {function} known.")

    def check_if_function_schema_known(self: Self, function: FunctionName) -> Literal[True] | FormFillerStateError:
        """
        Checks that the function and its schema are known
        """

        def _check() -> Literal[True] | FormFillerStateError:
            if function in self.function_schemas:
                return True
            return UnknownFunctionSchemaError(function=function, message=f"No function schema for function {function}")

        return self.check_if_function_known(function) and _check()

    def check_if_function_parameter_in_schema(
        self: Self, function: FunctionName, parameter: ParameterName
    ) -> Literal[True] | FormFillerStateError:
        """
        Checks that:
        - the function and its schema are known,
        - the parameter is in the schema
        """

        def _check() -> Literal[True] | FormFillerStateError:
            if (
                (parameters := self.function_schemas[function].parameters) is not None
                and parameters.properties is not None
                and parameter in parameters.properties
            ):
                return True
            return UnknownFunctionParameterError(function=function, parameter=parameter)

        return self.check_if_function_schema_known(function=function) and _check()

    def check_if_function_parameter_value_valid(
        self: Self,
        function: FunctionName,
        parameter: ParameterName,
        parameter_value: JsonValue,
    ) -> Literal[True] | FormFillerStateError:
        """
        Checks that:
        - the function and its schema are known,
        - the parameter is in the schema,
        - the parameter value is valid
        """

        def _validate() -> Literal[True] | FormFillerStateError:
            return self.function_schemas[function].validate_parameter_value(
                parameter=parameter,
                parameter_value=parameter_value,
            )

        return self.check_if_function_parameter_in_schema(function=function, parameter=parameter) and _validate()

    def check_if_function_parameter_is_skippable(
        self: Self,
        function: FunctionName,
        parameter: ParameterName,
    ) -> Literal[True] | FormFillerStateError:
        """
        Checks that:
        - the function and its schema are known,
        - the parameter is in the schema,
        - the parameter is skippable
        """

        def _check() -> Literal[True] | FormFillerStateError:
            if (function_schema := self.function_schemas.get(function)) is None:
                return UnknownFunctionSchemaError(
                    function=function,
                    message=f"No function schema for function {function}",
                )
            return function_schema.validate_parameter_skip(parameter=parameter)

        return self.check_if_function_parameter_in_schema(function=function, parameter=parameter) and _check()

    def check_if_function_parameters_valid(
        self: Self,
        function: FunctionName,
    ) -> Literal[True] | FormFillerStateError:
        """
        Checks that:
        - the function and its schema are known,
        - all filled parameter values are valid
        """

        def _validate() -> Literal[True] | FormFillerStateError:
            parameter_values = self.function_parameters_filled.get(function, {})
            return self.function_schemas[function].validate_parameter_values(parameter_values=parameter_values)

        return self.check_if_function_schema_known(function=function) and _validate()

    def check_if_function_schema_valid(
        self: Self,
        function: FunctionName,
        function_schema: FunctionSchema,
    ) -> Literal[True] | FormFillerStateError:
        """
        Checks that that the function schema is valid
        """
        if function_schema.name != function:
            return InvalidFunctionSchemaError(
                function=function,
                function_schema=function_schema,
                message=f"Function name in schema ({function_schema.name}) does not match function name ({function}).",
            )
        return function_schema.validate_parameters_schema()

    def check_if_function_return_value_valid(
        self: Self,
        function: FunctionName,
        return_value: JsonValue,
    ) -> Literal[True] | FormFillerStateError:
        """
        Checks that:
        - the function and its schema are known,
        - the return value is valid
        """
        if (function_schema := self.function_schemas.get(function)) is None:
            return UnknownFunctionSchemaError(
                function=function,
                message=f"No function schema for function {function}",
            )
        return self.check_if_function_schema_known(function=function) and function_schema.validate_return_value(
            return_value=return_value
        )

    def get_requestable_function_parameters(self: Self, function: FunctionName) -> list[ParameterName]:
        """
        Get the list of parameter names that can be requested for the given function.
        If the function is unknown, an empty list is returned.
        Args:
            function (FunctionName): The function name.
        Returns:
            list[ParameterName]: The list of parameters that can be requested for the given function.
        """
        if function not in self.function_schemas:
            return []
        return self.function_schemas[function].get_requestable_parameter_names(
            parameters_filled=self.function_parameters_filled.get(function, {}),
            parameters_skipped=self.function_parameters_skipped.get(function, []),
        )


def update_form_filler_state(state: FormFillerState, step: FormFillerStep) -> FormFillerState | FormFillerStateError:
    new_state = state.model_copy()
    match step:
        case UserStep() | AssistantStep():
            new_state.messages.append(step)
        case FunctionCandidates():
            for function in step.candidates:
                new_state = new_state.add_function(function.function)
        case InspectFunction():
            # check that the function is known
            if not (error := new_state.check_if_function_known(step.function)):
                return error.add_cause(step)
        case FunctionSchema():
            # check that the function is known
            if not (error := new_state.check_if_function_known(step.name)):
                return error.add_cause(step)
            # check that function schema is valid
            if not (error := new_state.check_if_function_schema_valid(step.name, step)):
                return error.add_cause(step)
            # add function schema to the state
            new_state = new_state.add_function_schema(step.name, step)
        case RequestFunction():
            # check that all functions are known
            if step.functions:
                errors = [
                    check
                    for check in (new_state.check_if_function_known(function) for function in step.functions)
                    if not check
                ]
                if errors:
                    assert isinstance(errors[0], FormFillerStateError)
                    return errors[0].add_cause(step)
        case CallFunction():
            # check that the function parameters are valid (also checks if the function and its schema are known)
            if not (error := new_state.check_if_function_parameters_valid(step.function)):
                return error.add_cause(step)
        case FunctionResult():
            # check that the function result is valid (also checks if the function and its schema are known)
            if not (error := new_state.check_if_function_return_value_valid(step.function, step.result)):
                return error.add_cause(step)
            # add function call result to the state
            new_state = new_state.model_copy(
                update={"function_call_results": new_state.function_call_results | {step.function: step.result}}
            )
        case RequestFunctionParameters():
            # check that the function parameters are in the schema (also checks if the function and its schema are known)
            errors = [
                check
                for check in (
                    new_state.check_if_function_parameter_in_schema(function=step.function, parameter=parameter)
                    for parameter in step.parameters
                )
                if not check
            ]
            if errors:
                assert isinstance(errors[0], FormFillerStateError)
                return errors[0].add_cause(step)
        case RequestFunctionCallConfirmation():
            # check that the function parameters are valid (also checks if the function and its schema are known)
            if not (error := new_state.check_if_function_parameters_valid(step.function)):
                return error.add_cause(step)
        case UpdateFunctionParameters():
            if step.assign:
                # check that each parameter value is valid (also checks if the function and its schema are known, and that the parameter is in the schema)
                errors = [
                    check
                    for check in (
                        new_state.check_if_function_parameter_value_valid(
                            function=step.function,
                            parameter=parameter,
                            parameter_value=parameter_value,
                        )
                        for parameter, parameter_value in step.assign.items()
                    )
                    if not check
                ]
                if errors:
                    assert isinstance(errors[0], FormFillerStateError)
                    return errors[0].add_cause(step)
                filled = new_state.function_parameters_filled
                filled[step.function] = filled.get(step.function, {}) | step.assign
            if step.skip:
                # check that each parameter is skippable (also checks if the function and its schema are known, and that the parameter is in the schema)
                errors = [
                    check
                    for check in (
                        new_state.check_if_function_parameter_is_skippable(function=step.function, parameter=parameter)
                        for parameter in step.skip
                    )
                    if not check
                ]
                if errors:
                    assert isinstance(errors[0], FormFillerStateError)
                    return errors[0].add_cause(step)
                filled = new_state.function_parameters_filled
                skipped = new_state.function_parameters_skipped
                if step.function not in skipped:
                    skipped[step.function] = []
                for k in step.skip:
                    filled.get(step.function, {}).pop(k, None)
                    skipped[step.function].append(k)
        case GatherValuesThought():
            # update raw function parameters
            new_state = new_state.model_copy(
                update={"raw_function_parameters": new_state.raw_function_parameters | {step.function: step.parameters}}
            )
        case VerifyValuesThought():
            # update function parameters verified
            new_state = new_state.model_copy(
                update={
                    "function_parameters_verified": new_state.function_parameters_verified
                    | {step.function: step.parameters}
                }
            )
        case RefuseInvalidFunctionParameterValue():
            # check that the parameter value is NOT valid (also checks if the function and its schema are known, and that the parameter is in the schema)
            if new_state.check_if_function_parameter_value_valid(
                function=step.function,
                parameter=step.parameter,
                parameter_value=step.parameter_value,
            ):
                return ValidFunctionParameterValueError(
                    function=step.function,
                    parameter=step.parameter,
                    parameter_value=step.parameter_value,
                ).add_cause(step)
        case RefuseInvalidFunctionParameterSkip():
            # check that the parameter is NOT skippable (also checks if the function and its schema are known, and that the parameter is in the schema)
            if new_state.check_if_function_parameter_is_skippable(
                function=step.function,
                parameter=step.parameter,
            ):
                return ValidFunctionParameterSkipError(
                    function=step.function,
                    parameter=step.parameter,
                ).add_cause(step)
        case RefuseInexistentFunctionParameter() | AnswerFromFunctionSchema() | NoAnswerFromFunctionSchema():
            # in all these cases, we just need to check that the function and its schema are known
            if not (error := new_state.check_if_function_schema_known(step.function)):
                return error.add_cause(step)
        case (
            ResolveFunction()
            | RequestExitConfirmation()
            | RefuseInexistentFunction()
            | RefuseToEngage()
            | LLMOutputParsingFailureAction()
            | SetNextNode()
            | StopStep()
        ):
            # in all these cases, there is nothing to check and nothing to update the state
            pass
        case _:
            # unknown step type: we should verify that they do not need to be handled
            logger.warning(
                f"Unknown step type: {step}. Verify that it does not need to be handled while updating form filler state. Will be ignored."
            )

    return new_state


def compute_form_filler_state(tape: FormFillerTape) -> FormFillerState | FormFillerStateError:
    # TODO: cache
    state = FormFillerState()
    for step in tape:
        result = update_form_filler_state(state, step)
        if isinstance(result, FormFillerStateError):
            return result
        state = result
    return state
