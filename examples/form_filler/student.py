import json
import logging
from typing import Any, Generator

import pydantic

from tapeagents.agent import Agent, Node
from tapeagents.core import LLMOutputParsingFailureAction, Prompt, SetNextNode
from tapeagents.dialog_tape import AssistantStep
from tapeagents.llms import LLM, LLMStream
from tapeagents.utils import sanitize_json_completion

from .error import FormFillerStateError
from .state import compute_form_filler_state, update_form_filler_state
from .steps import (
    AnswerFromFunctionSchema,
    CallFunction,
    Exit,
    FormFillerStep,
    InspectFunction,
    NoAnswerFromFunctionSchema,
    RefuseInexistentFunction,
    RefuseInvalidFunctionParameterSkip,
    RefuseInvalidFunctionParameterValue,
    RefuseToEngage,
    RequestExitConfirmation,
    RequestFunction,
    RequestFunctionCallConfirmation,
    RequestFunctionParameters,
    ResolveFunction,
    UpdateFunctionParameters,
)
from .tape import FormFillerTape, prepare_formfiller_template_variables
from .types import FunctionName

logger = logging.getLogger(__name__)


class StudentAgent(Agent[FormFillerTape]):
    @classmethod
    def create(cls, llm: LLM, templates: dict[str, dict[str, str]]):
        nodes = [
            CheckFunctionCandidatesNode(),
            IntentDiscoveryNode(),
            StudentNode(),
        ]
        return super().create(name="student_formfiller", llms=llm, templates=templates, nodes=nodes)

    def generate_steps(self, tape: FormFillerTape, llm_stream: LLMStream) -> Generator[FormFillerStep, None, None]:
        # compute current tape state
        state = compute_form_filler_state(tape)
        assert not isinstance(
            state, FormFillerStateError
        ), f"Tape state should not be an error.\nTAPE:\n{tape}\nSTATE:\n{state}"
        # try to update state for all predicted steps
        for predicted_step in super().generate_steps(tape, llm_stream):
            assert isinstance(
                predicted_step, FormFillerStep
            ), f"Predicted step should be a FormFillerStep, got {predicted_step}"
            state_or_error = update_form_filler_state(state, predicted_step)
            # stop as soon as there is an error
            if isinstance(state_or_error, FormFillerStateError):
                yield state_or_error  # type: ignore
                return
            # yield the step if state updated successfully
            state = state_or_error
            yield predicted_step


class CheckFunctionCandidatesNode(Node):
    name: str = "check_function_candidates_node"

    def generate_steps(
        self, agent: StudentAgent, tape: FormFillerTape, llm_stream: LLMStream
    ) -> Generator[FormFillerStep, None, None]:
        if not tape.are_function_candidates_listed:
            assert tape.last_user_step is not None, "User step is required to resolve function"
            yield ResolveFunction(query=tape.last_user_step.content)
        else:
            yield SetNextNode(next_node="intent_discovery_node")


class IntentDiscoveryNode(Node):
    name: str = "intent_discovery_node"

    def make_prompt(self, agent: StudentAgent, tape: FormFillerTape) -> Prompt:
        # If no function has been inspected
        if not tape.intent_is_discovered:
            template_variables = self.create_template_variables(tape)
            return Prompt.from_user_message(
                agent.templates["intent_discovery_post_list_template"].format(**template_variables)
            )
        else:
            return Prompt()

    def create_template_variables(self, tape: FormFillerTape) -> dict[str, Any]:
        return {
            "function_search_results": json.dumps(tape.function_candidates_step.model_dump()["candidates"], indent=2)
            if tape.function_candidates_step
            else "< Not available >",
            "last_user_message": tape.last_user_step.content if tape.last_user_step else "< Not available >",
        }

    def generate_steps(
        self, agent: StudentAgent, tape: FormFillerTape, llm_stream: LLMStream
    ) -> Generator[FormFillerStep, None, None]:
        if llm_stream:
            inspect_function_found = False
            predicted_steps = parse_completion(None, llm_stream.get_text())
            for step in predicted_steps:
                if isinstance(step, InspectFunction):
                    inspect_function_found = True
                yield step

            if not inspect_function_found:
                yield SetNextNode(next_node="intent_discovery_node")
        else:
            yield SetNextNode(next_node="student_node")


class StudentNode(Node):
    name: str = "student_node"

    def make_prompt(self, agent: StudentAgent, tape: FormFillerTape) -> Prompt:
        template_variables = prepare_formfiller_template_variables(tape)
        return Prompt.from_user_message(agent.templates["main_template"].format(**template_variables))

    def generate_steps(
        self, agent: StudentAgent, tape: FormFillerTape, llm_stream: LLMStream
    ) -> Generator[FormFillerStep, None, None]:
        assert tape.function_schema_step is not None, "Function schema should already be determined in student node"
        yield from parse_completion(tape.function_schema_step.name, llm_stream.get_text())
        # go back to this node if the agent gets called again
        yield SetNextNode(next_node="student_node")


def parse_completion(function_name: FunctionName | None, completion: str) -> Generator[FormFillerStep, None, None]:
    # first, check if the completion has a valid json format
    try:
        step_dict = json.loads(sanitize_json_completion(completion))
    except Exception as e:
        logger.exception(f"Failed to parse agent output: {completion}\n\nError: {e}")
        yield LLMOutputParsingFailureAction(error=f"Failed to parse agent output.\n\nError: {e}", llm_output=completion)
        return
    try:
        yield from reconstruct_compact_steps(function_name, step_dict)
    except pydantic.ValidationError as e:
        logger.exception(f"Failed to parse step data: {step_dict}\n\nError: {e}")
        yield LLMOutputParsingFailureAction(error=f"Failed to parse agent output.\n\nError: {e}", llm_output=completion)


def reconstruct_compact_steps(
    function_name: FunctionName | None, data: dict[str, Any]
) -> Generator[FormFillerStep, None, None]:
    data.setdefault("thoughts", [])

    for entry in data["thoughts"]:
        match entry:
            # Intent classification
            case "no_form_found":
                yield RefuseInexistentFunction()
            case {"offer_forms": functions}:
                yield RequestFunction(functions=functions)

            # question answering
            case "answer":
                assert isinstance(function_name, str), f"Function must be selected, {data}"
                yield AnswerFromFunctionSchema(function=function_name)
            case "no_answer":
                assert isinstance(function_name, str), f"Function must be selected, {data}"
                yield NoAnswerFromFunctionSchema(function=function_name)

            # unrelated requests
            case "unrelated_request":
                yield RefuseToEngage()

            # Update function parameters"
            case {"update_parameters": updates}:
                assert isinstance(function_name, str), f"Function must be selected, {data}"
                assigns = {}
                skips = []
                for key, value in updates.items():
                    if value == "_skip":
                        skips.append(key)
                    elif value == "_refuse_value":
                        yield RefuseInvalidFunctionParameterValue(
                            function=function_name,
                            parameter=key,
                            parameter_value="unk",
                        )
                    elif value == "_refuse_skip":
                        yield RefuseInvalidFunctionParameterSkip(function=function_name, parameter=key)
                    else:
                        assigns[key] = value
                if assigns or skips:
                    yield UpdateFunctionParameters(function=function_name, assign=assigns, skip=skips)

            # Confirm or request
            case "confirm_exit":
                yield RequestExitConfirmation()
            case "confirm_submit":
                assert isinstance(function_name, str), f"Function must be selected, {data}"
                yield RequestFunctionCallConfirmation(function=function_name)
            case {"next_requested_parameter": parameter}:
                assert isinstance(function_name, str), f"Function must be selected, {data}"
                yield RequestFunctionParameters(function=function_name, parameters=[parameter])
            case _:
                yield LLMOutputParsingFailureAction(error=f"Unexpected entry: {entry}", llm_output=str(data))
                return

    if "action" not in data:
        yield LLMOutputParsingFailureAction(error="No action found in the compact steps", llm_output=str(data))
        return

    match data["action"]:
        case "list_forms":
            yield ResolveFunction(query=None)
        case {"select_form": function_name}:
            assert isinstance(function_name, str), f"Function name must be a string in {data}"
            yield InspectFunction(function=function_name)
        case "exit":
            yield Exit()
        case "submit":
            assert isinstance(function_name, str), "Function name is required for submit"
            yield CallFunction(function=function_name)
        case {"prompt": message}:
            yield AssistantStep(content=message)
        case _:
            yield LLMOutputParsingFailureAction(error=f"Unexpected action: {data['action']}", llm_output=str(data))
