from __future__ import annotations

import warnings
from typing import Any, Literal, Mapping, Sequence, TypeAlias

import jsonref
import jsonschema
import referencing.jsonschema
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FieldSerializationInfo,
    JsonValue,
    SerializerFunctionWrapHandler,
    TypeAdapter,
    model_serializer,
)
from typing_extensions import Self

from tapeagents.core import Observation

from .error import (
    FormFillerStateError,
    InvalidFunctionParametersError,
    InvalidFunctionParameterSkipError,
    InvalidFunctionParameterValueError,
    InvalidFunctionReturnValueError,
    InvalidFunctionSchemaError,
    UnknownFunctionParameterError,
    UnknownFunctionSchemaError,
)
from .types import FunctionName, ParameterName

JsonType: TypeAlias = Literal["null", "boolean", "object", "array", "number", "string", "integer"]


class JsonSchema(BaseModel):
    """
    This is a simplified version of JSON Schema,
    which is used to describe the schema of a function's parameters and return value.
    See https://json-schema.org/understanding-json-schema/index.html for JSON Schema
    and https://datatracker.ietf.org/doc/html/draft-pbryan-zyp-json-ref-03 for JSON references.
    """

    # camel case (e.g. maxLength) is used in the JSON schema standard
    # some of our data uses underscore-separated names (e.g. max_length)
    # with this setting this class can load both
    model_config = ConfigDict(populate_by_name=True)

    title: None | str = Field(default=None, description="The title of the schema.")
    description: None | str = Field(default=None, description="The description of the schema.")
    type: None | JsonType | list[JsonType] = Field(default=None, description="The type(s) of the schema.")
    format: None | str = Field(default=None, description="The format of the schema.")
    enum: None | list[JsonValue] = Field(default=None, description="The enum values of this schema.")
    const: None | JsonValue = Field(default=None, description="The constant value of this schema.")
    default: None | JsonValue = Field(default=None, description="The default value of this schema.")
    properties: None | dict[str, JsonSchema] = Field(
        default=None,
        description="The properties (key-subschema pairs) of the object that this schema describes.",
    )
    required: None | list[str] = Field(
        default=None,
        description="The list of required properties of the object that this schema describes.",
    )
    items: None | JsonSchema | list[JsonSchema] = Field(
        default=None,
        description="The subschema(s) of the items of the array that this schema describes.",
    )
    all_of: None | list[JsonSchema] = Field(
        alias="allOf",
        default=None,
        description="To validate against allOf, the given data must be valid against all of the given subschemas. This is an intersection type.",
    )
    any_of: None | list[JsonSchema] = Field(
        alias="anyOf",
        default=None,
        description="To validate against anyOf, the given data must be valid against any (one or more) of the given subschemas. This is a union type.",
    )
    one_of: None | list[JsonSchema] = Field(
        alias="oneOf",
        default=None,
        description="To validate against oneOf, the given data must be valid against exactly one of the given subschemas. This is an exclusive union type.",
    )
    minimum: None | int | float = Field(
        default=None,
        description="The minimum value of this schema.",
    )
    maximum: None | int | float = Field(
        default=None,
        description="The maximum value of this schema.",
    )
    exclusive_minimum: None | bool = Field(
        alias="exclusiveMinimum",
        default=None,
        description="Whether the minimum value is exclusive.",
    )
    exclusive_maximum: None | bool = Field(
        alias="exclusiveMaximum",
        default=None,
        description="Whether the maximum value is exclusive.",
    )
    multiple_of: None | int | float = Field(
        alias="multipleOf",
        default=None,
        description="The multiple of this schema.",
    )
    min_length: None | int = Field(
        alias="minLength",
        default=None,
        description="The minimum length of this schema.",
    )
    max_length: None | int = Field(
        alias="maxLength",
        default=None,
        description="The maximum length of this schema.",
    )
    pattern: None | str = Field(
        default=None,
        description="The regular expression pattern of this schema.",
    )
    min_items: None | int = Field(
        alias="minItems",
        default=None,
        description="The minimum number of items of this schema.",
    )
    max_items: None | int = Field(
        alias="maxItems",
        default=None,
        description="The maximum number of items of this schema.",
    )
    unique_items: None | bool = Field(
        alias="uniqueItems",
        default=None,
        description="Whether the items of this schema are unique.",
    )
    prefixItems: None | list[JsonSchema] = Field(
        alias="prefixItems",
        default=None,
        description="The prefix items of this schema.",
    )
    contains: None | JsonSchema = Field(
        default=None,
        description="The schema of the items that this schema contains.",
    )
    minContains: None | int = Field(
        alias="minContains",
        default=None,
        description="The minimum number of items that this schema contains.",
    )
    maxContains: None | int = Field(
        alias="maxContains",
        default=None,
        description="The maximum number of items that this schema contains.",
    )
    additional_properties: None | bool | JsonSchema = Field(
        alias="additionalProperties",
        default=None,
        description="Whether additional properties are allowed for this schema.",
    )
    ref: None | str = Field(alias="$ref", default=None, description="A reference to another schema.")

    @model_serializer(mode="wrap")
    def serialize_model(
        self: Self,
        handler: SerializerFunctionWrapHandler,
        _: FieldSerializationInfo,
    ) -> dict[str, Any]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            result = handler(self)
        return {k: v for k, v in result.items() if v is not None}

    def get_possible_values(self) -> list[tuple[Any, str]]:
        if self.enum and len(self.enum) > 0 or self.one_of and len(self.one_of) > 0:
            if self.enum:
                possible_values = [(value, str(value)) for value in self.enum if value]
            elif self.one_of:
                possible_values = [(choice.const, str(choice.title)) for choice in self.one_of if choice.const]
            else:
                raise ValueError(f"Unknown schema type: {self}")
        else:
            possible_values = []
        return possible_values


class FunctionSchema(Observation):
    name: FunctionName
    description: str = Field(description="The description of the function.")
    parameters: None | JsonSchema = Field(default=None, description="The JSON schema of the function's parameters.")
    return_value: None | JsonSchema = Field(default=None, description="The JSON schema of the function's return value.")
    definitions: None | dict[str, JsonSchema] = Field(
        default=None,
        description="The definitions (key-subschema pairs) that can be referenced in the parameters and return value schemas.",
    )
    kind: Literal["function_schema"] = "function_schema"

    @model_serializer(mode="wrap")
    def serialize_model(
        self: Self,
        handler: SerializerFunctionWrapHandler,
        _: FieldSerializationInfo,
    ) -> dict[str, Any]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            result = handler(self)
        return {k: v for k, v in result.items() if v is not None}

    @property
    def with_replaced_refs(self: Self) -> Self:
        """
        Property that returns a copy of the function schema with all references replaced by their target schemas.
        Returns:
            FunctionSchema: The schema with all references replaced by their target schemas.
        """
        # Implementation removed to avoid depenendecy on jsonref
        return self.model_validate(jsonref.replace_refs(self.model_dump(by_alias=True)))

    def get_parameter_schema(self: Self, parameter: ParameterName) -> None | JsonSchema:
        """
        Get the schema of the given parameter.
        Returns `None` if the schema is not available.
        Args:
            parameter (ParameterName): The name of the parameter.
        Returns:
            None | JsonSchema: The schema of the parameter or `None` if not available.
        """
        schema = self
        if schema.parameters is None:
            return None
        if schema.parameters.properties is None:
            return None
        return schema.parameters.properties.get(parameter, None)

    def is_optional_parameter(self: Self, parameter: ParameterName) -> bool:
        """
        Check whether the given parameter is optional.
        Args:
            parameter (ParameterName): The name of the parameter.
        Raises:
            ValueError: If the schema for the parameters is not available.
        Returns:
            bool: Whether the parameter is optional.
        """
        if self.parameters is None:
            raise ValueError(f"Paremeters of function {self.name!r} have no schema.")
        if self.parameters.required is None:
            return False
        return parameter not in self.parameters.required

    def get_parameter_description(self: Self, parameter: ParameterName) -> None | str:
        """
        Get the description of the given parameter.
        Args:
            parameter (ParameterName): The name of the parameter.
        Returns:
            None | str: The description of the parameter or `None` if not available.
        """
        parameter_schema = self.get_parameter_schema(parameter)
        if parameter_schema is None:
            return None
        return parameter_schema.description

    def get_parameter_default(self: Self, parameter: ParameterName) -> None | Any:
        """
        Get the default value of the given parameter.
        Args:
            parameter (ParameterName): The name of the parameter.
        Returns:
            None | Any: The default value of the parameter or `None` if not available.
        """
        parameter_schema = self.get_parameter_schema(parameter)
        if parameter_schema is None:
            return None
        return parameter_schema.default

    @property
    def parameter_names(self: Self) -> list[ParameterName]:
        """
        Get the names of the input parameters.
        Raises:
            ValueError: If the schema for the parameters is not available.
        Returns:
            list[ParameterName]: The names of the input parameters.
        """
        if self.parameters is None:
            raise ValueError(f"Parameters of function {self.name!r} have no schema.")
        if self.parameters.properties is None:
            return []
        return [parameter for parameter in self.parameters.properties.keys()]

    def get_requestable_parameter_names(
        self: Self,
        parameters_filled: Mapping[ParameterName, JsonValue],
        parameters_skipped: Sequence[ParameterName],
    ) -> list[ParameterName]:
        """
        Return a list of parameter names that can be requested.
        The returned list contains all parameter names that are not assigned and not skipped.
        The order of the returned list is the same as the order of the parameters in the schema.
        Args:
            parameters_filled (Mapping[ParameterName, JsonValue]): The parameters that are already filled.
                Only the keys are used.
            parameters_skipped (Sequence[ParameterName]): The parameters that are skipped.
        Returns:
            list[ParameterName]: The names of the parameters that can be requested.
        """
        return [
            parameter
            for parameter in self.parameter_names
            if parameter not in parameters_filled and parameter not in parameters_skipped
        ]

    def get_parameter_enum_values(self: Self, parameter_name: str):
        param_schema = self.get_parameter_schema(parameter_name)
        if param_schema:
            return [value_tuple[0] for value_tuple in param_schema.get_possible_values()]
        return []

    def get_parameter_enum_labels(self: Self, parameter_name: str) -> dict[str, str]:
        param_schema = self.get_parameter_schema(parameter_name)
        if param_schema:
            return {value_tuple[1]: value_tuple[0] for value_tuple in param_schema.get_possible_values()}
        return {}

    def is_parameter_type_enum(self: Self, parameter_name: str):
        param_schema = self.get_parameter_schema(parameter_name)
        if param_schema and param_schema.enum is not None:
            return True
        return False

    def is_parameter_type_one_of_enum(self: Self, parameter_name: str):
        param_schema = self.get_parameter_schema(parameter_name)
        if (
            param_schema
            and param_schema.one_of is not None
            and any(
                one_of_schema.type == "string" and one_of_schema.const is not None
                for one_of_schema in param_schema.one_of
            )
        ):
            return True
        return False

    def validate_parameters_schema(self: Self) -> Literal[True] | FormFillerStateError:
        """
        Validate the parameters JSON schema of the function.
        Returns:
            Literal[True] | FormFillerStateError: `True` if the schema is valid, an error with falsey value otherwise.
        """
        schema = self
        if schema.parameters is None:
            return InvalidFunctionSchemaError(
                function=schema.name,
                function_schema=self,
                message=f"Schema of function {self.name!r} has no parameters.",
            )
        parameters_schema_obj: referencing.jsonschema.Schema = schema.parameters.model_dump(by_alias=True)
        try:
            jsonschema.Draft202012Validator.check_schema(
                schema=parameters_schema_obj,
                format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER,
            )
        except jsonschema.SchemaError as e:
            return InvalidFunctionSchemaError(
                function=schema.name,
                function_schema=self,
                message=e.message,
            )
        return True

    def validate_parameter_value(
        self: Self,
        parameter: ParameterName,
        parameter_value: JsonValue,
    ) -> Literal[True] | FormFillerStateError:
        """
        Validate the given parameter value against the JSON schema of the given parameter.
        Args:
            parameter (ParameterName): The name of the parameter.
            parameter_value (JsonValue): The value of the parameter.
        Returns:
            Literal[True] | FormFillerStateError: `True` if the value is valid, an error with falsey value otherwise.
        """
        schema = self
        if not isinstance(schema.parameters, JsonSchema):
            return UnknownFunctionParameterError(
                function=self.name,
                parameter=parameter,
                message=f"Schema of function {self.name!r} has no parameters.",
            )
        if schema.parameters.properties is None:
            return UnknownFunctionParameterError(
                function=self.name,
                parameter=parameter,
                message=f"Parameters of function {self.name!r} have no properties.",
            )
        parameter_schema = schema.get_parameter_schema(parameter)
        if parameter_schema is None:
            return UnknownFunctionParameterError(
                function=self.name,
                parameter=parameter,
                message=f"Schema for parameter {parameter!r} is not defined for function {self.name!r}.",
            )
        parameter_schema_obj: referencing.jsonschema.Schema = parameter_schema.model_dump(by_alias=True)
        parameter_value_obj = TypeAdapter(JsonValue).dump_python(parameter_value)
        try:
            jsonschema.validate(
                instance=parameter_value_obj,
                schema=parameter_schema_obj,
                cls=jsonschema.Draft202012Validator,
                format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER,
            )
        except jsonschema.ValidationError as e:
            return InvalidFunctionParameterValueError(
                function=self.name,
                parameter=parameter,
                parameter_value=parameter_value,
                message=e.message,
            )
        return True

    def validate_parameter_skip(
        self: Self,
        parameter: ParameterName,
    ) -> Literal[True] | FormFillerStateError:
        """
        Validate that the given parameter can be skipped.
        Args:
            parameter (ParameterName): The name of the parameter to skip.
        Returns:
            Literal[True] | FormFillerStateError: `True` if the parameter can be skipped, an error with falsey value otherwise.
        """
        schema = self
        if not isinstance(schema.parameters, JsonSchema):
            return UnknownFunctionParameterError(
                function=self.name,
                parameter=parameter,
                message=f"Schema of function {self.name!r} has no parameters.",
            )
        if schema.parameters.properties is None:
            return UnknownFunctionParameterError(
                function=self.name,
                parameter=parameter,
                message=f"Parameters of function {self.name!r} have no properties.",
            )
        if isinstance(required := schema.parameters.required, list) and parameter in required:
            return InvalidFunctionParameterSkipError(
                function=self.name,
                parameter=parameter,
                message=f"Parameter {parameter!r} of function {self.name!r} is required and cannot be skipped.",
            )
        parameter_schema = schema.get_parameter_schema(parameter)
        if parameter_schema is None:
            return UnknownFunctionParameterError(
                function=self.name,
                parameter=parameter,
                message=f"Schema for parameter {parameter!r} is not defined for function {self.name!r}.",
            )
        if parameter_schema.default is not None:
            return InvalidFunctionParameterSkipError(
                function=self.name,
                parameter=parameter,
                message=f"Parameter {parameter!r} of function {self.name!r} has default value {parameter_schema.default!r} and cannot be skipped.",
            )
        return True

    def validate_parameter_values(
        self: Self,
        parameter_values: dict[ParameterName, JsonValue],
    ) -> Literal[True] | FormFillerStateError:
        """
        Validate the given parameters against the parameters JSON schema of the function.
        Args:
            parameter_values (dict[ParameterName, JsonValue]): The parameters and their values to validate.
        Returns:
            Literal[True] | FormFillerStateError: `True` if the values are valid, an error with falsey value otherwise.
        """
        schema = self
        if not isinstance(schema.parameters, JsonSchema):
            return UnknownFunctionSchemaError(
                function=self.name,
                message=f"Schema of function {self.name!r} has no parameters.",
            )
        parameters_schema_obj: referencing.jsonschema.Schema = schema.parameters.model_dump(by_alias=True)
        parameters_obj = {}
        json_value_adapter = TypeAdapter(JsonValue)
        for parameter, parameter_value in parameter_values.items():
            parameters_obj[parameter] = json_value_adapter.dump_python(parameter_value)
        try:
            jsonschema.validate(
                instance=parameters_obj,
                schema=parameters_schema_obj,
                cls=jsonschema.Draft202012Validator,
                format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER,
            )
        except jsonschema.ValidationError as e:
            missing_parameters = []
            if e.validator == "required":
                missing_parameters = [
                    parameter for parameter in e.validator_value if parameter not in e.instance.keys()
                ]
            return InvalidFunctionParametersError(
                function=self.name,
                parameters=parameter_values,
                missing_parameters=missing_parameters,
                message=e.message,
            )
        return True

    def validate_return_value_schema(self: Self) -> Literal[True] | FormFillerStateError:
        """
        Validate the return value JSON schema of the function.
        Returns:
            Literal[True] | FormFillerStateError: `True` if the schema is valid, an error with falsey value otherwise.
        """
        schema = self
        if not isinstance(schema.return_value, JsonSchema):
            # It is ok to have no return value schema.
            return True
        return_value_schema_obj: referencing.jsonschema.Schema = schema.return_value.model_dump(by_alias=True)
        try:
            jsonschema.Draft202012Validator.check_schema(
                schema=return_value_schema_obj,
                format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER,
            )
        except jsonschema.SchemaError as e:
            return InvalidFunctionSchemaError(
                function=schema.name,
                function_schema=self,
                message=e.message,
            )
        return True

    def validate_return_value(
        self: Self,
        return_value: JsonValue,
    ) -> Literal[True] | FormFillerStateError:
        """
        Validate the given return value against the JSON schema of the function.
        Args:
            return_value (JsonValue): The return value.
        Returns:
            Literal[True] | FormFillerStateError: `True` if the value is valid, an error with falsey value otherwise.
        """
        schema = self
        if not isinstance(schema.return_value, JsonSchema):
            # It is ok to have no return value schema.
            return True
        return_value_schema_obj: referencing.jsonschema.Schema = schema.return_value.model_dump(by_alias=True)
        return_value_obj = TypeAdapter(JsonValue).dump_python(return_value)
        try:
            jsonschema.validate(
                instance=return_value_obj,
                schema=return_value_schema_obj,
                cls=jsonschema.Draft202012Validator,
                format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER,
            )
        except jsonschema.ValidationError as e:
            return InvalidFunctionReturnValueError(
                function=self.name,
                return_value=return_value,
                message=e.message,
            )
        return True
