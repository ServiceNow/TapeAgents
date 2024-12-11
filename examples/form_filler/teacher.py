import json
import logging
from typing import Any, Generator

from pydantic import BaseModel

from tapeagents.agent import Agent, Node
from tapeagents.core import Action, LLMOutputParsingFailureAction, Prompt, SetNextNode
from tapeagents.dialog_tape import AssistantStep, UserStep
from tapeagents.llms import LLM, LLMStream

from .error import FormFillerStateError
from .state import compute_form_filler_state, update_form_filler_state
from .steps import (
    ACTION_STEPS,
    I_NOTE_STEPS,
    I_SHOULD_STEPS,
    CallFunction,
    FormFillerStep,
    GatherValuesThought,
    InspectFunction,
    RequestFunctionCallConfirmation,
    ResolveFunction,
    UpdateFunctionParameters,
    VerifyValuesThought,
)
from .tape import FormFillerTape, prepare_formfiller_template_variables
from .utils import render_chat_template, sanitize_json_completion

logger = logging.getLogger(__name__)


class StepWrapper(BaseModel):
    step: FormFillerStep


class TeacherAgent(Agent[FormFillerTape]):
    @classmethod
    def create(cls, llm: LLM, templates: dict[str, dict[str, str]]):
        nodes = [
            RoutingNode(),
            IntentDiscoveryNode(),
            GatherValuesNode(),
            VerifyValuesNode(),
            RetrospectivePlanNode(),
            ForwardPlanNode(),
            GenerationNode(),
            WriteConfirmationNode(),
            CallFunctionNode(),
        ]
        return super().create(name="teacher_formfiller", llms=llm, templates=templates, nodes=nodes)

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


class RoutingNode(Node):
    name: str = "routing_node"

    def generate_steps(
        self, agent: TeacherAgent, tape: FormFillerTape, llm_stream: LLMStream
    ) -> Generator[FormFillerStep, None, None]:
        if not tape.are_function_candidates_listed:
            assert tape.last_user_step is not None, "User step is required to resolve function"
            yield ResolveFunction(query=tape.last_user_step.content)
        else:
            # Check if history has a request confirmation step
            request_confirmation_found = False
            for step in reversed(tape.steps):
                if isinstance(step, RequestFunctionCallConfirmation):
                    request_confirmation_found = True
                    break

            # If request confirmation is found, we need to check if user confirmed or not
            # Otherwise, continue the chain as normal (i.e. gather, verify, plan, etc.)
            if request_confirmation_found:
                assert isinstance(
                    tape.steps[-1], UserStep
                ), "Last step should be a UserStep (either they confirm or they don't) when request confirmation is found"
                yield SetNextNode(next_node="call_function_node")
            else:
                yield SetNextNode(next_node="intent_discovery_node")


class IntentDiscoveryNode(Node):
    name: str = "intent_discovery_node"

    def make_prompt(self, agent: TeacherAgent, tape: FormFillerTape) -> Prompt:
        # If no function has been inspected
        if not tape.intent_is_discovered:
            template_variables = self.create_template_variables(tape)
            return Prompt(messages=render_chat_template(agent.templates["intent_discovery_prompt"], template_variables))
        else:
            return Prompt()

    def create_template_variables(self, tape: FormFillerTape) -> dict[str, Any]:
        template_values = prepare_formfiller_template_variables(tape)
        template_values["function_search_results"] = (
            json.dumps(tape.function_candidates_step.model_dump(exclude={"metadata"})["candidates"], indent=2)
            if tape.function_candidates_step
            else "< Not available >"
        )
        return template_values

    def generate_steps(
        self, agent: TeacherAgent, tape: FormFillerTape, llm_stream: LLMStream
    ) -> Generator[FormFillerStep, None, None]:
        if llm_stream:
            inspect_function_found = False
            predicted_steps = parse_completion(llm_stream.get_text())
            for step in predicted_steps:
                if isinstance(step, InspectFunction):
                    inspect_function_found = True
                yield step
                if isinstance(step, Action):
                    break  # stop yielding steps after the first action step

            if not inspect_function_found:
                yield SetNextNode(next_node="intent_discovery_node")
        else:
            yield SetNextNode(next_node="gather_values_node")


class TeacherNode(Node):
    # placeholders to be replaced in the subclasses
    name: str = "teacher_node"
    template_name: str = "teacher_node_template"
    yield_actions: bool = False  # by default, intermediate nodes should only yield thoughts

    def create_template_variables(self, agent: TeacherAgent, tape: FormFillerTape) -> dict[str, Any]:
        return prepare_formfiller_template_variables(tape)

    def make_prompt(self, agent: TeacherAgent, tape: FormFillerTape) -> Prompt:
        template_variables = self.create_template_variables(agent, tape)
        return Prompt(messages=render_chat_template(agent.templates[self.template_name], template_variables))

    def generate_steps(
        self, agent: TeacherAgent, tape: FormFillerTape, llm_stream: LLMStream
    ) -> Generator[FormFillerStep, None, None]:
        completion = llm_stream.get_text()
        logger.debug(f"teacher agent at node {self.name} --- Prompt: {llm_stream.prompt} --- Completion: {completion}")
        for predicted_step in parse_completion(completion):
            if not self.yield_actions and isinstance(predicted_step, ACTION_STEPS):
                # Actions (but not error) should be skipped in intermediate nodes
                continue
            yield predicted_step
            if isinstance(predicted_step, Action):
                break  # stop yielding steps after the first action step


class GatherValuesNode(TeacherNode):
    name: str = "gather_values_node"
    template_name: str = "gather_values_prompt"


class VerifyValuesNode(TeacherNode):
    name: str = "verify_values_node"
    template_name: str = "verify_values_prompt"

    def create_template_variables(self, agent: TeacherAgent, tape: FormFillerTape) -> dict[str, Any]:
        template_values = super().create_template_variables(agent, tape)

        gather_step = None
        for step in reversed(tape.steps):
            if isinstance(step, GatherValuesThought):
                gather_step = step
                break

        assert gather_step, "GatherValuesThought should be found in previous steps to be able to VerifyValuesNode"
        template_values["extracted_parameters_json"] = json.dumps(gather_step.parameters, indent=2)

        return template_values


class RetrospectivePlanNode(TeacherNode):
    name: str = "retrospective_plan_node"
    template_name: str = "retrospective_plan_prompt"

    def create_template_variables(self, agent: TeacherAgent, tape: FormFillerTape) -> dict[str, Any]:
        template_values = super().create_template_variables(agent, tape)

        verify_step = None
        for step in reversed(tape.steps):
            if isinstance(step, VerifyValuesThought):
                verify_step = step
                break

        assert verify_step, "VerifyValuesThought should be found in previous steps to be able to RetrospectivePlanNode"
        template_values["verified_parameters_json"] = json.dumps(verify_step.parameters, indent=2)
        template_values["retrospective_thoughts_description"] = dict(
            agent.templates["retrospective_thoughts_description"]
        )

        return template_values


class ForwardPlanNode(TeacherNode):
    name: str = "forward_plan_node"
    template_name: str = "forward_plan_prompt"

    def create_template_variables(self, agent: TeacherAgent, tape: FormFillerTape) -> dict[str, Any]:
        template_values = super().create_template_variables(agent, tape)

        planning_steps = []
        for predicted_step in reversed(tape.steps):
            if isinstance(predicted_step, Action):
                break

            if isinstance(predicted_step, (I_SHOULD_STEPS, I_NOTE_STEPS)):
                planning_steps.append(predicted_step.model_dump(exclude={"metadata"}))

        template_values["planning_steps"] = json.dumps(planning_steps, indent=2)
        template_values["request_thoughts_description"] = dict(agent.templates["request_thoughts_description"])

        return template_values

    def generate_steps(
        self, agent: TeacherAgent, tape: FormFillerTape, llm_stream: LLMStream
    ) -> Generator[FormFillerStep, None, None]:
        request_confirmation_found = False
        # yield all predicted steps and check if request confirmation is found
        for step in super().generate_steps(agent, tape, llm_stream):
            if isinstance(step, RequestFunctionCallConfirmation):
                request_confirmation_found = True
            yield step
        # if request confirmation is found, set next node to write confirmation message
        if request_confirmation_found:
            yield SetNextNode(next_node="write_confirmation_node")


class GenerationNode(TeacherNode):
    name: str = "generation_node"
    template_name: str = "generate_prompt"
    yield_actions: bool = True  # this node will produce an Agent message, which is an action

    def create_template_variables(self, agent: TeacherAgent, tape: FormFillerTape) -> dict[str, Any]:
        template_values = super().create_template_variables(agent, tape)

        planning_steps = []
        for predicted_step in reversed(tape.steps):
            if isinstance(predicted_step, Action):
                break

            if isinstance(predicted_step, (I_SHOULD_STEPS, I_NOTE_STEPS)):
                planning_steps.append(predicted_step.model_dump(exclude={"metadata"}))

        template_values["planning_steps"] = json.dumps(planning_steps, indent=2)
        template_values["actions_description"] = dict(agent.templates["actions_description"])

        return template_values

    def generate_steps(
        self, agent: TeacherAgent, tape: FormFillerTape, llm_stream: LLMStream
    ) -> Generator[FormFillerStep, None, None]:
        yield from super().generate_steps(agent, tape, llm_stream)  # yield all predicted steps
        # go back to routing node if the agent gets called again
        yield SetNextNode(next_node="routing_node")


# This is actually the final node in the chain because it generates the confirmation message
# Once the user confirms the message, "CallFunctionNode" will terminate the chain
class WriteConfirmationNode(TeacherNode):
    name: str = "write_confirmation_node"
    template_name: str = "write_confirmation_prompt"
    yield_actions: bool = True  # this node will produce an Agent message, which is an action

    def create_template_variables(self, agent: TeacherAgent, tape: FormFillerTape) -> dict[str, Any]:
        template_values = super().create_template_variables(agent, tape)

        last_update: UpdateFunctionParameters | None = None
        for step in reversed(tape.steps):
            if isinstance(step, UpdateFunctionParameters):
                last_update = step
                break

        assert last_update
        template_values["last_update"] = json.dumps(last_update.model_dump(exclude={"metadata"}), indent=2)

        return template_values

    def generate_steps(
        self, agent: TeacherAgent, tape: FormFillerTape, llm_stream: LLMStream
    ) -> Generator[FormFillerStep, None, None]:
        yield from super().generate_steps(agent, tape, llm_stream)  # yield all predicted steps
        # hard code the confirmation message with summary of assigned parameters
        yield AssistantStep(
            content=f"Please review the information you provided. If you want to proceed, please approve.\n{tape.get_filled_parameters_as_llm_view()}"
        )
        # go back to routing node if the agent gets called again
        yield SetNextNode(next_node="routing_node")


class CallFunctionNode(TeacherNode):
    name: str = "call_function_node"
    template_name: str = "call_function_prompt"
    yield_actions: bool = True  # this node will try to produce a CallFunction Action

    def generate_steps(
        self, agent: TeacherAgent, tape: FormFillerTape, llm_stream: LLMStream
    ) -> Generator[FormFillerStep, None, None]:
        call_function_found = False
        assistant_message_found = False
        for step in super().generate_steps(agent, tape, llm_stream):  # yield all predicted steps
            if isinstance(step, CallFunction):
                call_function_found = True
            if isinstance(step, AssistantStep):
                # this should not happen based on the prompt, but sometimes the llm still generates an assistant message
                # we can either ignore this step and set_next_node to write_confirmation_node to write a proper confirmation message,
                # or we can keep it and set_next_node to routing_node
                # we choose the latter
                assistant_message_found = True
            yield step

        # If there was a call function, nothing else to do, otherwise check if there was an assistant message
        if not call_function_found:
            if assistant_message_found:
                # If there was an assistant message, skip write_confirmation_node and go back to routing node
                yield SetNextNode(next_node="routing_node")
            else:
                # If there was no assistant message, go to write_confirmation_node
                yield SetNextNode(next_node="write_confirmation_node")


def parse_completion(completion: str) -> Generator[FormFillerStep, None, None]:
    # first, check if the completion has a valid json format
    try:
        step_dict = json.loads(sanitize_json_completion(completion))
        if isinstance(step_dict, dict):
            step_dict = [step_dict]
        for step in step_dict:
            yield StepWrapper.model_validate({"step": step}).step
    except Exception as e:
        logger.exception(f"Failed to parse agent output: {completion}\n\nError: {e}")
        yield LLMOutputParsingFailureAction(error=f"Failed to parse agent output.\n\nError: {e}", llm_output=completion)
