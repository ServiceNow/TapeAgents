### copy-paste of llmd2.core.error ####

from abc import ABC, abstractmethod
from io import StringIO
from typing import Any, Literal, Union

from pydantic import Field, JsonValue
from typing_extensions import Self

from tapeagents.core import Action, Error, Step

from .types import FunctionName, ParameterName

# from .steps import FormFillerStep


class FormFillerStateError(ABC, Action, Error):
    """
    A base class for the form filler state errors.
    Errors are used to indicate that a step failed.
    They are not raised as exceptions but rather returned as a special step.
    They can be converted to exceptions using the :meth:`to_exc()` method.
    """

    kind: Literal[  # type: ignore
        "unknown_function_error",
        "unknown_function_schema_error",
        "invalid_function_schema_error",
        "unknown_function_parameter_error",
        "invalid_function_parameter_value_error",
        "valid_function_parameter_value_error",
        "invalid_function_parameter_skip_error",
        "valid_function_parameter_skip_error",
        "invalid_function_parameters_error",
        "function_call_error",
        "invalid_function_return_value_error",
        "unknown_error",
    ]
    # we use generic type here to avoid circular imports
    cause: Any = None

    @abstractmethod
    def to_exc(self: Self, turn: None | int = None, step: None | int = None) -> BaseException:
        """
        Convert the error to an exception that can be raised.
        """
        ...

    def add_cause(self, cause: Step) -> Self:
        return self.model_copy(update={"cause": cause})

    def __bool__(self: Self) -> bool:
        """
        Errors are always falsey.
        """
        return False


class ValueErrorWithContext(ValueError):
    def __init__(self, message: str, turn: int | None = None, step: int | None = None):
        super().__init__(message)
        self.turn = turn
        self.step = step


class UnknownFunctionError(FormFillerStateError):
    """
    An error that indicates that the function is unknown.
    """

    kind: Literal["unknown_function_error"] = "unknown_function_error"
    function: FunctionName
    message: str = ""

    def to_exc(self: Self, turn: None | int = None, step: None | int = None) -> BaseException:
        msg = StringIO()
        if turn is not None:
            msg.write(f"Turn {turn}: ")
        if step is not None:
            msg.write(f"Step {step}: ")
        msg.write(f"Unknown function {self.function!r}")
        if self.message:
            msg.write(f": {self.message}")
        return ValueErrorWithContext(msg.getvalue(), turn=turn, step=step)


class UnknownFunctionSchemaError(FormFillerStateError):
    """
    An error that indicates that the function schema is unknown.
    """

    kind: Literal["unknown_function_schema_error"] = "unknown_function_schema_error"
    function: FunctionName
    message: str = ""

    def to_exc(self: Self, turn: None | int = None, step: None | int = None) -> BaseException:
        msg = StringIO()
        if turn is not None:
            msg.write(f"Turn {turn}: ")
        if step is not None:
            msg.write(f"Step {step}: ")
        msg.write(f"Unknown schema for function {self.function!r}")
        if self.message:
            msg.write(f": {self.message}")
        return ValueErrorWithContext(msg.getvalue(), turn=turn, step=step)


class InvalidFunctionSchemaError(FormFillerStateError):
    """
    An error that indicates that the function schema is invalid.
    """

    kind: Literal["invalid_function_schema_error"] = "invalid_function_schema_error"
    function: FunctionName
    # we use Any here to avoid circular imports
    function_schema: Any
    message: str = ""

    def to_exc(self: Self, turn: None | int = None, step: None | int = None) -> BaseException:
        msg = StringIO()
        if turn is not None:
            msg.write(f"Turn {turn}: ")
        if step is not None:
            msg.write(f"Step {step}: ")
        msg.write(f"Invalid schema {self.function_schema!r} for function {self.function!r}")
        if self.message:
            msg.write(f": {self.message}")
        return ValueErrorWithContext(msg.getvalue(), turn=turn, step=step)


class UnknownFunctionParameterError(FormFillerStateError):
    """
    An error that indicates that the function parameter is unknown.
    """

    kind: Literal["unknown_function_parameter_error"] = "unknown_function_parameter_error"
    function: FunctionName
    parameter: ParameterName
    message: str = ""

    def to_exc(self: Self, turn: None | int = None, step: None | int = None) -> BaseException:
        msg = StringIO()
        if turn is not None:
            msg.write(f"Turn {turn}: ")
        if step is not None:
            msg.write(f"Step {step}: ")
        msg.write(f"Unknown parameter {self.parameter!r} of function {self.function!r}")
        if self.message:
            msg.write(f": {self.message}")
        return ValueErrorWithContext(msg.getvalue(), turn=turn, step=step)


class InvalidFunctionParameterValueError(FormFillerStateError):
    """
    An error that indicates that the value of a function parameter is invalid.
    """

    kind: Literal["invalid_function_parameter_value_error"] = "invalid_function_parameter_value_error"
    function: FunctionName
    parameter: ParameterName
    parameter_value: JsonValue
    message: str = ""

    def to_exc(self: Self, turn: None | int = None, step: None | int = None) -> BaseException:
        msg = StringIO()
        if turn is not None:
            msg.write(f"Turn {turn}: ")
        if step is not None:
            msg.write(f"Step {step}: ")
        msg.write(
            f"Invalid value {self.parameter_value!r} for parameter {self.parameter!r} of function {self.function!r}"
        )
        if self.message:
            msg.write(f": {self.message}")
        return ValueErrorWithContext(msg.getvalue(), turn=turn, step=step)


class ValidFunctionParameterValueError(FormFillerStateError):
    """
    An error that indicates that the value of a function parameter is valid but should have been invalid.
    """

    kind: Literal["valid_function_parameter_value_error"] = "valid_function_parameter_value_error"
    function: FunctionName
    parameter: ParameterName
    parameter_value: JsonValue
    message: str = ""

    def to_exc(self: Self, turn: None | int = None, step: None | int = None) -> BaseException:
        msg = StringIO()
        if turn is not None:
            msg.write(f"Turn {turn}: ")
        if step is not None:
            msg.write(f"Step {step}: ")
        msg.write(
            f"Valid value {self.parameter_value!r} for parameter {self.parameter!r} of function {self.function!r}"
        )
        if self.message:
            msg.write(f": {self.message}")
        return ValueErrorWithContext(msg.getvalue(), turn=turn, step=step)


class InvalidFunctionParameterSkipError(FormFillerStateError):
    """
    An error that indicates that the parameter was skipped but was not allowed to be skipped.
    """

    kind: Literal["invalid_function_parameter_skip_error"] = "invalid_function_parameter_skip_error"
    function: FunctionName
    parameter: ParameterName
    message: str = ""

    def to_exc(self: Self, turn: None | int = None, step: None | int = None) -> BaseException:
        msg = StringIO()
        if turn is not None:
            msg.write(f"Turn {turn}: ")
        if step is not None:
            msg.write(f"Step {step}: ")
        msg.write(f"Unexpected skip for parameter {self.parameter!r} of function {self.function!r}")
        if self.message:
            msg.write(f": {self.message}")
        return ValueErrorWithContext(msg.getvalue(), turn=turn, step=step)


class ValidFunctionParameterSkipError(FormFillerStateError):
    """
    An error that indicates that the parameter could have been skipped but was not skipped.
    """

    kind: Literal["valid_function_parameter_skip_error"] = "valid_function_parameter_skip_error"
    function: FunctionName
    parameter: ParameterName
    message: str = ""

    def to_exc(self: Self, turn: None | int = None, step: None | int = None) -> BaseException:
        msg = StringIO()
        if turn is not None:
            msg.write(f"Turn {turn}: ")
        if step is not None:
            msg.write(f"Step {step}: ")
        msg.write(f"Valid skip for parameter {self.parameter!r} of function {self.function!r}")
        if self.message:
            msg.write(f": {self.message}")
        return ValueErrorWithContext(msg.getvalue(), turn=turn, step=step)


class InvalidFunctionParametersError(FormFillerStateError):
    """
    An error that indicates that the values of the function parameters are invalid as a whole.
    """

    kind: Literal["invalid_function_parameters_error"] = "invalid_function_parameters_error"
    function: FunctionName
    parameters: dict[ParameterName, JsonValue]
    missing_parameters: list[ParameterName] = Field(default=[])
    message: str = ""

    def to_exc(self: Self, turn: None | int = None, step: None | int = None) -> BaseException:
        msg = StringIO()
        if turn is not None:
            msg.write(f"Turn {turn}: ")
        if step is not None:
            msg.write(f"Step {step}: ")
        msg.write(f"Invalid values {self.parameters!r} for parameters of function {self.function!r}")
        if self.message:
            msg.write(f": {self.message}")
        return ValueErrorWithContext(msg.getvalue(), turn=turn, step=step)


class FunctionCallError(FormFillerStateError):
    """
    An error that indicates that the call to a function failed.
    The function was successfullly called but returned an empty result or an error.
    """

    kind: Literal["function_call_error"] = "function_call_error"
    function: FunctionName
    message: str = ""

    def to_exc(self: Self, turn: None | int = None, step: None | int = None) -> BaseException:
        msg = StringIO()
        if turn is not None:
            msg.write(f"Turn {turn}: ")
        if step is not None:
            msg.write(f"Step {step}: ")
        msg.write(f"Call to function {self.function!r} failed")
        if self.message:
            msg.write(f": {self.message}")
        return ValueErrorWithContext(msg.getvalue(), turn=turn, step=step)


class InvalidFunctionReturnValueError(FormFillerStateError):
    """
    An error that indicates that the return value of a function is invalid given its schema.
    """

    kind: Literal["invalid_function_return_value_error"] = "invalid_function_return_value_error"
    function: FunctionName
    return_value: JsonValue
    message: str = ""

    def to_exc(self: Self, turn: None | int = None, step: None | int = None) -> BaseException:
        msg = StringIO()
        if turn is not None:
            msg.write(f"Turn {turn}: ")
        if step is not None:
            msg.write(f"Step {step}: ")
        msg.write(f"Return value {self.return_value!r} of function {self.function!r} is invalid")
        if self.message:
            msg.write(f": {self.message}")
        return ValueErrorWithContext(msg.getvalue(), turn=turn, step=step)


class EnvironmentActionError(FormFillerStateError):
    """
    An error that indicates that an error occurred during action execution the environment.
    """

    kind: Literal["environment_action_error"] = "environment_action_error"
    message: str = ""

    def to_exc(self: Self, turn: None | int = None, step: None | int = None) -> BaseException:
        return RuntimeError(f"Environment action error: {self.message}")


class UnknownError(FormFillerStateError):
    """
    An error that indicates that an unknown error occurred.
    """

    kind: Literal["unknown_error"] = "unknown_error"
    message: str = ""

    def to_exc(self: Self, turn: None | int = None, step: None | int = None) -> BaseException:
        msg = StringIO()
        if turn is not None:
            msg.write(f"Turn {turn}: ")
        if step is not None:
            msg.write(f"Step {step}: ")
        msg.write("Unknown error")
        if self.message:
            msg.write(f": {self.message}")
        return ValueErrorWithContext(self.message, turn=turn, step=step)


_FormFillerStateError = Union[
    UnknownFunctionError,
    UnknownFunctionSchemaError,
    InvalidFunctionSchemaError,
    UnknownFunctionParameterError,
    InvalidFunctionParameterValueError,
    ValidFunctionParameterValueError,
    InvalidFunctionParameterSkipError,
    ValidFunctionParameterSkipError,
    InvalidFunctionParametersError,
    FunctionCallError,
    InvalidFunctionReturnValueError,
    EnvironmentActionError,
    UnknownError,
]
