import logging
from abc import abstractmethod
from typing import Any

import numpy as np
from pydantic import BaseModel

from tapeagents.core import Step

from .schema import FunctionSchema, JsonSchema
from .state import FormFillerState
from .steps import InspectFunction, RequestFunctionParameters
from .tape import FormFillerTape

logger = logging.getLogger(__name__)


class StepFilter(BaseModel):
    @abstractmethod
    def filter(self, state: FormFillerState, step: Step) -> bool:
        raise NotImplementedError()


class AlwaysTrue(StepFilter):
    def filter(self, state: FormFillerState, step: Step) -> bool:
        return True


def is_nonboolean_enum(param_schema: JsonSchema):
    values = param_schema.get_possible_values()
    if not values:
        return False
    if len(values) == 2 and any(
        str(const).lower() in ["yes", "no"] or str(title) in ["yes", "no"] for const, title in values
    ):
        return False
    return True


def get_parameter_type(parameter_schema: JsonSchema, possible_values: list[Any] | None = None) -> Any:
    if possible_values is None:
        possible_values = parameter_schema.get_possible_values()
    if possible_values:
        return "Categorical"
    elif parameter_schema.format:
        return parameter_schema.format
    elif parameter_schema.type:
        return parameter_schema.type
    else:
        return "unknown"


def render_function_doc(schema: FunctionSchema) -> str:
    function_doc = schema.with_replaced_refs

    doc_as_text = []
    doc_as_text.append(function_doc.description.strip())

    if function_doc.parameters is not None and function_doc.parameters.properties is not None:
        doc_as_text += "\n\nArgs:"
        for parameter_name, parameter_schema in function_doc.parameters.properties.items():
            parameter_type = get_parameter_type(parameter_schema)
            is_required = function_doc.parameters.required and parameter_name in function_doc.parameters.required
            possible_values = parameter_schema.get_possible_values()

            var_info = f"\n- {parameter_name} [{parameter_type}, {'required' if is_required else 'optional'}]: {schema.description or ''}"

            if possible_values:
                possible_values_rendered = ", ".join(
                    [f'{title} ("{const}")' if const != title else f'"{const}"' for const, title in possible_values]
                )
                var_info += f" Possible values are [{possible_values_rendered}]."

            if parameter_schema.default:
                # fetch the "title" of the default value "const" if it exists
                if (
                    possible_values
                    and parameter_schema.default in [const for const, _ in possible_values]
                    and parameter_schema.default not in [title for _, title in possible_values]
                ):
                    default_title = [title for const, title in possible_values if const == parameter_schema.default][0]
                    default_value = f'{default_title} ("{parameter_schema.default}")'
                else:
                    default_value = f'"{parameter_schema.default}"'
                var_info += f" Default value is {default_value}."

            doc_as_text.append(var_info)

    else:
        doc_as_text += "There are not arguments for this function"

    doc_as_text = "".join(doc_as_text)

    return doc_as_text


def extract_specific_parameters(all_params: list[str], requested_elements: list[str]) -> list[str]:
    delimiters = ["_", "-"]
    target = []

    def contains_requested_substring(param: str) -> bool:
        for substring in requested_elements:
            for delimiter in delimiters:
                # Checking for patterns with leading, enclosing, and trailing delimiters
                if _matches_pattern(param, substring, delimiter):
                    return True
        return False

    def _matches_pattern(param: str, substring: str, delimiter: str) -> bool:
        pattern = f"{delimiter}{substring}{delimiter}"
        return (
            param == substring
            or pattern in param
            or param.startswith(f"{substring}{delimiter}")
            or param.endswith(f"{delimiter}{substring}")
        )

    # for each parameter name, check if we have a substring we want
    for param in all_params:
        if contains_requested_substring(param):
            target.append(param)
    return target


def check_all_parameters_have_mininum_description_len(schema: FunctionSchema) -> bool:
    if schema.with_replaced_refs.parameters is None:
        return False
    all_params_schema = schema.with_replaced_refs.parameters.properties
    if all_params_schema is None:
        return False
    params_desc = [param.description for _, param in all_params_schema.items()]
    return all(param_desc and len(param_desc.split()) > 2 for param_desc in params_desc)


def find_requested_enum(state: FormFillerState, step: RequestFunctionParameters) -> list[str]:
    for parameter_name in step.parameters:  # type: ignore
        param_schema = state.function_schemas[step.function].get_parameter_schema(parameter_name)
        assert param_schema is not None
        param_values = param_schema.get_possible_values()
        if param_values:
            return [param[1] for param in param_values]
    return []


def get_step_instruction_params(
    request_param_thought: RequestFunctionParameters, state: FormFillerState
) -> dict[str, Any]:
    if len(request_param_thought.parameters) != 1:
        raise ValueError("Can only generate user message when only one parameter is requested")
    if len(state.function_schemas) > 1:
        raise ValueError("User model only support dialogues about 1 function")
    function_name = request_param_thought.function

    params = {}
    # Function info
    params["function_name"] = function_name
    (schema,) = state.function_schemas.values()

    all_params_schema = schema.with_replaced_refs.parameters
    assert all_params_schema is not None and all_params_schema.properties is not None
    all_params = all_params_schema.properties.keys()
    current_param = request_param_thought.parameters[0]
    filled_params = list(state.function_parameters_filled.get(function_name, {}).keys())
    skipped_params = list(state.function_parameters_skipped.get(function_name, []))
    filled_enum_params = [p for p in filled_params if all_params_schema.properties[p].get_possible_values()]
    future_params = list(set(all_params) - set(filled_params) - set(skipped_params) - set([current_param]))
    future_enum_params = [p for p in future_params if is_nonboolean_enum(all_params_schema.properties[p])]
    params["all_params"] = ", ".join(all_params)
    params["current_param"] = current_param
    params["filled_params"] = ", ".join(filled_params)
    params["skipped_params"] = ", ".join(skipped_params)
    params["future_params"] = ", ".join(future_params)

    # extract technical parameters (email, uri, ip_address, url)
    web_param = extract_specific_parameters(list(all_params), ["email", "uri", "ip_address", "url"])
    if web_param:
        params["web_param"] = np.random.choice(web_param)
    # extract date parameters (date, time)
    date_param = extract_specific_parameters(list(all_params), ["date", "time"])
    if date_param:
        params["date_param"] = date_param

    assert schema.name == function_name
    function_doc = render_function_doc(schema)
    if function_doc:
        params["function_doc"] = function_doc

    doc_description = schema.description
    if doc_description:
        params["documentation_details"] = doc_description

    # Extract useful info from function_name
    # named_entity = get_named_entity(function_name=function_name)
    # params["named_entity"] = named_entity
    params["named_entity"] = function_name  # just put the function name for now

    # Details about the parameter that the agent is currently requesting
    curr_param_schema = all_params_schema.properties[current_param]
    assert curr_param_schema is not None
    params["current_parameter_name"] = curr_param_schema.title or current_param
    curr_value_titles = [pair[1] for pair in curr_param_schema.get_possible_values()]
    if curr_value_titles:
        params["current_parameter_enum_values"] = "[" + ", ".join(curr_value_titles) + "]"

    # Choose a previously filled parameter
    if len(filled_params) or len(skipped_params):
        # Choose a parameter we have already filled
        param: str = np.random.choice(filled_params + skipped_params)
        param_schema = all_params_schema.properties[param]
        params["parameter_name"] = param_schema.title or param
        params["assigned_parameters"] = ", ".join(
            [
                all_params_schema.properties[p].title if all_params_schema.properties[p].title else p.replace("_", " ")
                for p in filled_params + skipped_params
            ]
        )  # type: ignore
        if param in filled_params:
            param_value = state.function_parameters_filled.get(function_name, {}).get(param)
            if param_value:
                params["parameter_value"] = param_value
        elif param in skipped_params:
            params["parameter_value"] = "<SKIPPED>"

    # Choose a previous filled enum parameter
    if len(filled_enum_params):
        # Choose an enum parameter we have already filled
        enum_param = np.random.choice(filled_enum_params)
        enum_param_schema = all_params_schema.properties[enum_param]
        enum_value_titles = [pair[1] for pair in enum_param_schema.get_possible_values()]
        params["enum_parameter_name"] = enum_param_schema.title or enum_param
        params["enum_parameter_values"] = ", ".join(enum_value_titles)
        enum_param_value = state.function_parameters_filled.get(function_name, {}).get(enum_param)
        if enum_param_value:
            params["enum_parameter_value"] = enum_param_value
            other_possible_values = [
                pair[1] for pair in enum_param_schema.get_possible_values() if pair[0] != enum_param_value
            ]
            if other_possible_values:
                params["other_enum_parameter_values"] = ", ".join(other_possible_values)
                params["other_enum_parameter_value"] = np.random.choice(other_possible_values)

    # Choose parameter we can fill in the future
    if len(future_params):
        future_param = np.random.choice(future_params)
        future_param_schema = all_params_schema.properties[future_param]
        params["future_parameter_name"] = future_param_schema.title or future_param
        future_params2 = list(set(future_params) - set([future_param]))
        if future_params2:
            future_param2 = np.random.choice(future_params2)
            future_param_schema2 = all_params_schema.properties[future_param2]
            params["future_parameter_name2"] = future_param_schema2.title or future_param2
            future_params3 = list(set(future_params2) - set([future_param2]))
            if future_params3:
                future_param3 = np.random.choice(future_params3)
                future_param_schema3 = all_params_schema.properties[future_param3]
                params["future_parameter_name3"] = future_param_schema3.title or future_param3

    # Choose an enum parameter we can fill in the future
    if len(future_enum_params):
        future_enum = np.random.choice(future_enum_params)
        future_enum_param_schema = all_params_schema.properties[future_enum]
        future_enum_value_titles = [pair[1] for pair in future_enum_param_schema.get_possible_values()]
        assert future_enum_value_titles
        params["future_enum_parameter_name"] = future_enum_param_schema.title or future_enum
        params["future_enum_parameter_values"] = "- " + "\n- ".join(future_enum_value_titles)

    return params


class NotYesNoEnum(StepFilter):
    def filter(self, state: FormFillerState, step: FormFillerTape) -> bool:
        if not isinstance(step, RequestFunctionParameters):
            raise ValueError()
        yesno_answers = set(["yes", "no", "", "maybe"])
        enum_values = find_requested_enum(state, step)
        return len(enum_values) > 0 and not all([v.lower() in yesno_answers for v in enum_values])


class IsEnumStep(StepFilter):
    def filter(self, state: FormFillerState, step: Step) -> bool:
        if not isinstance(step, RequestFunctionParameters):
            raise ValueError()
        return len(find_requested_enum(state, step)) > 0


class NotYesNoEnumOld(StepFilter):
    def filter(self, state: FormFillerState, step: Step) -> bool:
        if not isinstance(step, RequestFunctionParameters):
            raise ValueError()
        yesno_answers = set(["yes", "no", "", "maybe"])
        enum_values = find_requested_enum(state, step)
        return len(enum_values) > 0 and not all([v.lower() in yesno_answers for v in enum_values])


class HasFilledSlot(StepFilter):
    def filter(self, state: FormFillerState, _: Step) -> bool:
        flat_slots_list = [v for _, params in state.function_parameters_filled.items() for v in params.values()]
        return len(flat_slots_list) > 0


class HasFunction(StepFilter):
    allowed_fuctions: list[str]

    def filter(self, _: FormFillerState, step: Step) -> bool:
        function = None
        if isinstance(step, InspectFunction):
            function = step.result.name  # type: ignore
        return function is not None and function in self.allowed_fuctions


class HasAnyFunction(StepFilter):
    # making sure the user selected a function.
    # will be false at the first dialogue PromptUserForTextMessage("hi how can I help you")
    def filter(self, state: FormFillerState, step: Step) -> bool:
        return bool(state.function_schemas)


class HasNoFunction(StepFilter):
    # making sure the user did not select a function yet.
    # will be true at the first dialogue PromptUserForTextMessage("hi how can I help you")
    def filter(self, state: FormFillerState, step: Step) -> bool:
        return not bool(state.function_schemas)


class ParameterType(StepFilter):
    """
    Parameters can be required/optional and with default values or not.
    This makes 4 different cases:

    [CASE 1] required & no default
        user cannot skip, must give a value

    [CASE 2] required & has default
        user can "skip", agent sets to default value

    [CASE 3] optional & no default
        user can skip, agent does a real skip

    [CASE 4] optional & has default
        same as [CASE 2]: user can "skip", agent sets to default value
    """

    required_no_default: bool = False  # [CASE 1]
    required_with_default: bool = False  # [CASE 2]
    optional_no_default: bool = False  # [CASE 3]
    optional_with_default: bool = False  # [CASE 4]

    def filter(self, state: FormFillerState, step: Step) -> bool:
        """
        Return True if at least 1 parameter satisfies at least 1 of the 'activated' cases
        """
        if isinstance(step, RequestFunctionParameters):
            function_name = step.function
            parameter_names = step.parameters
            function_schema = state.function_schemas[function_name].with_replaced_refs

            for parameter in parameter_names:
                is_optional = function_schema.is_optional_parameter(parameter)
                parameter_schema = function_schema.get_parameter_schema(parameter)
                if parameter_schema is None:
                    raise ValueError(
                        f"Requesting parameter {parameter} but function schema `{function_name}` has no parameter schema?"
                    )
                if self.required_no_default and (not is_optional and not parameter_schema.default):
                    return True
                elif self.required_with_default and (not is_optional and parameter_schema.default):
                    return True
                elif self.optional_no_default and (is_optional and not parameter_schema.default):
                    return True
                elif self.optional_with_default and (is_optional and parameter_schema.default):
                    return True
            return False
        return False


class HasRequiredTemplateParams(StepFilter):
    params: tuple[str, ...]

    def filter(self, state: FormFillerState, step: Step) -> bool:
        try:
            if not isinstance(step, RequestFunctionParameters):
                raise ValueError(f"Expected RequestFunctionParameters, got {step}")
            available_params = get_step_instruction_params(step, state)
            return all([p in available_params for p in self.params])
        except Exception as e:
            logger.error(f"Error in HasRequiredTemplateParams: {e}")
            return False


class AreAllParamDescriptionsOfRequiredLen(StepFilter):
    def filter(self, state: FormFillerState, step: Step) -> bool:
        (schema,) = state.function_schemas.values()
        desc_len = check_all_parameters_have_mininum_description_len(schema)
        return desc_len


class HasRequestedElements(StepFilter):
    requested_elements: tuple[str, ...]
    unwanted_elements: tuple[str, ...]
    delimiters: tuple[str, ...]

    def contains_unwanted_substring(self, param: str) -> bool:
        return any(unwanted in param for unwanted in self.unwanted_elements)

    def contains_requested_substring(self, param: str) -> bool:
        for substring in self.requested_elements:
            for delimiter in self.delimiters:
                # Checking for patterns with leading, enclosing, and trailing delimiters
                if self._matches_pattern(param, substring, delimiter):
                    return True
        return False

    def _matches_pattern(self, param: str, substring: str, delimiter: str) -> bool:
        pattern = f"{delimiter}{substring}{delimiter}"
        return (
            param == substring
            or pattern in param
            or param.startswith(f"{substring}{delimiter}")
            or param.endswith(f"{delimiter}{substring}")
        )

    def filter(self, state: FormFillerState, step: Step) -> bool:
        try:
            if not isinstance(step, RequestFunctionParameters):
                raise TypeError(f"Expected RequestFunctionParameters, got {type(step).__name__}")

            requested_param = step.parameters[0]
            return not self.contains_unwanted_substring(requested_param) and self.contains_requested_substring(
                requested_param
            )
        except TypeError as e:
            logger.error(f"Type error in HasRequestedElements: {e}")
        except IndexError as e:
            logger.error(f"Index error accessing step parameters in HasRequestedElements: {e}")
        return False


def extract_placeholder_keys(format_string: str | list[dict[str, str]]) -> list[str]:
    """
    Extracts all placeholders in a string; format() style
    Example: "Hello, {name}! Your balance is {balance:.2f} dollars." -> ['name', 'balance']
    """
    from string import Formatter

    formatter = Formatter()

    if isinstance(format_string, str):
        keys = [field_name for _, field_name, _, _ in formatter.parse(format_string) if field_name is not None]
        return keys
    elif isinstance(format_string, list):
        keys = []
        for message in format_string:
            for field_name in extract_placeholder_keys(message["content"]):
                keys.append(field_name)
        return keys
