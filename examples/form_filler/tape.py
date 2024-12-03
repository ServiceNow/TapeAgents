from datetime import datetime
from typing import Any, Iterator, Tuple
from pydantic import BaseModel, Field, TypeAdapter
import json

from tapeagents.core import Tape, TapeMetadata, Thought, Observation, Action
from tapeagents.dialog_tape import UserStep, AssistantStep

from examples.form_filler.steps import *
from examples.form_filler.schema import FunctionSchema


class FormFillerContext(BaseModel):
    env_spec: str
    date: str


class FormFillerAgentMetadata(TapeMetadata):
    last_user_tape_id: str | None = None


class FormFillerUserMetadata(TapeMetadata):
    last_agent_tape_id: str | None = None
    user_behavior: str = ''
    user_secondary_behavior: str = ''
    other: dict[str, Any] = {}


class FormFillerTape(Tape[FormFillerContext, FormFillerStep]):
    metadata: FormFillerAgentMetadata | FormFillerUserMetadata = Field(default_factory=FormFillerAgentMetadata)

    def get_context_and_predicted_steps(self) -> Tuple[Iterator[FormFillerStep], Iterator[FormFillerStep]]:
        """
        split the tape into 2:
        - context_steps = all the steps before the last user message.
        - predicted_steps = all the steps from the last user message until the end.
        """
        context_steps: list[FormFillerStep] = []
        predicted_steps: list[FormFillerStep] = []
        in_context = False
        for step in reversed(self.steps):
            # switches after the first user message
            if not in_context and isinstance(step, UserStep):
                in_context = True

            if in_context:
                context_steps.append(step)
            else:
                predicted_steps.append(step)

        # put back the steps in order
        return reversed(context_steps), reversed(predicted_steps)
    
    @property
    def predicted_steps(self) -> Iterator[FormFillerStep]:
        return self.get_context_and_predicted_steps()[1]
    
    @property
    def context_steps(self) -> Iterator[FormFillerStep]:
        return self.get_context_and_predicted_steps()[0]
    
    @property
    def last_action(self) -> Action | None:
        for step in reversed(self.steps):
            if isinstance(step, Action):
                return step
        return None

    @property
    def last_user_step(self) -> UserStep | None:
        for step in reversed(self.steps):
            if isinstance(step, UserStep):
                return step
        return None

    @property
    def last_assistant_step(self) -> AssistantStep | None:
        for step in reversed(self.steps):
            if isinstance(step, AssistantStep):
                return step
        return None

    @property
    def predicted_i_should(self) -> bool:
        for step in self.predicted_steps:
            if isinstance(step, I_SHOULD_STEPS):
                return True
        return False

    @property
    def i_should_in_context(self) -> bool:
        for step in self.context_steps:
            if isinstance(step, I_SHOULD_STEPS):
                return True
        return False

    @property
    def predicted_i_note(self) -> bool:
        for step in self.predicted_steps:
            if isinstance(step, I_NOTE_STEPS):
                return True
        return False

    @property
    def i_note_in_context(self) -> bool:
        for step in self.context_steps:
            if isinstance(step, I_NOTE_STEPS):
                return True
        return False

    @property
    def function_candidates_step(self) -> FunctionCandidates | None:
        for step in self.steps:
            if isinstance(step, FunctionCandidates):
                return step
        return None

    @property
    def are_function_candidates_listed(self) -> bool:
        return self.function_candidates_step is not None

    @property
    def function_schema_step(self) -> FunctionSchema | None:
        for step in self.steps:
            if isinstance(step, FunctionSchema):
                return step
        return None
    
    @property
    def intent_is_discovered(self) -> bool:
        return self.function_schema_step is not None

    def get_filled_parameters_as_llm_view(self) -> str:
        assigned_parameters = {}
        skipped_parameters = []
        for step in self.steps:
            if isinstance(step, UpdateFunctionParameters):
                for p_name, p_value in step.assign.items():
                    assigned_parameters[p_name] = p_value
                for p_name in step.skip:
                    skipped_parameters.append(p_name)
        if assigned_parameters:
            assigned_parameters = "\n- " + "\n- ".join([f"{k}={v}" for k, v in assigned_parameters.items()])
        else:
            assigned_parameters = " {}"
        if skipped_parameters:
            skipped_parameters = "\n- " + "\n- ".join(skipped_parameters) if skipped_parameters else ""
        else:
            skipped_parameters = " []"

        return "Assigned Parameters:" + assigned_parameters + "\nSkipped Parameters:" + skipped_parameters

    def render_dialogue_as_text(self) -> str:
        messages = []
        for step in self.steps:
            if isinstance(step, UserStep):
                messages.append(f'User: {step.content}')
            elif isinstance(step, AssistantStep):
                messages.append(f'Agent: {step.content}')
        return '\n'.join(messages)


def from_llmd2_dialog_dict(d: dict) -> FormFillerTape:
    steps = []
    for step in d["steps"]:
        match step["kind"]:
            case "resolve_function_action":
                steps.append(ResolveFunction(query=step.get("query")))
                steps.append(FunctionCandidates(
                    candidates=TypeAdapter(list[FunctionCandidate]).validate_python(step["result"])))
            case "inspect_function_action":
                steps.append(InspectFunction(function=step["function"]))
                steps.append(FunctionSchema.model_validate(step["result"]))
            case "prompt_user_for_text_message_action":
                steps.append(AssistantStep(content=step["prompt"]))
                if "result" in step:
                    steps.append(UserStep(content=step["result"]))
            case "call_function_action":
                steps.append(CallFunction(function=step["function"]))
                if "result" in step:
                    steps.append(FunctionResult(function=step["function"], result=step["result"]))
            case "request_function_thought":
                steps.append(RequestFunction(functions=step.get("functions")))
            case "request_function_parameters_thought":
                steps.append(RequestFunctionParameters(
                    function=step["function"],
                    parameters=step["parameters"],
                ))
            case "request_function_call_confirmation_thought":
                steps.append(RequestFunctionCallConfirmation(function=step["function"]))
            case "request_exit_confirmation_thought":
                steps.append(RequestExitConfirmation())
            case "update_function_parameters_thought":
                steps.append(UpdateFunctionParameters(
                    function=step["function"],
                    assign=step.get("assign", {}),
                    skip=step.get("skip", []),
                ))
            case "gather_values_thought":
                steps.append(GatherValuesThought(
                    function=step["function"],
                    parameters=step.get("parameters", {}),
                ))
            case "verify_values_thought":
                steps.append(VerifyValuesThought(
                    function=step["function"],
                    parameters=step.get("parameters", {}),
                ))
            case "refuse_inexistent_function_parameter_thought":
                steps.append(RefuseInexistentFunctionParameter(function=step["function"]))
            case "refuse_invalid_function_parameter_value_thought":
                steps.append(RefuseInvalidFunctionParameterValue(
                    function=step["function"],
                    parameter=step["parameter"],
                    parameter_value=step.get("parameter_value"),
                ))
            case "refuse_invalid_function_parameter_skip_thought":
                steps.append(RefuseInvalidFunctionParameterSkip(
                    function=step["function"],
                    parameter=step["parameter"],
                ))
            case "refuse_inexistent_function_thought":
                steps.append(RefuseInexistentFunction())
            case "refuse_to_engage_thought":
                steps.append(RefuseToEngage())
            case "answer_from_function_schema_thought":
                steps.append(AnswerFromFunctionSchema(function=step["function"]))
            case "no_answer_in_function_schema_thought":
                steps.append(NoAnswerFromFunctionSchema(function=step["function"]))
            case "exit_action":
                steps.append(Exit())
            case _:
                raise ValueError(f"Unknown step: {step}")

    if d["metadata"]["provenance"] == "user_fork":
        metadata = FormFillerUserMetadata(
            id=d["id"],
            last_agent_tape_id=d["metadata"].get("parent_id"),
            user_behavior=d["metadata"].get("behavior_alias"),
        )
    elif d["metadata"]["provenance"] == "agent_fork":
        metadata = FormFillerAgentMetadata(
            id=d["id"],
            last_user_tape_id=d["metadata"].get("parent_id"),
        )
    context = FormFillerContext(
        env_spec=d.get("before_summary", {}).get("env_spec"),
        date=d.get("before_summary", {}).get("dialogue_date"),
    )
    if not context.date:
        context.date = datetime.now().strftime("%Y-%m-%d")
    return FormFillerTape(metadata=metadata, context=context, steps=steps)


def prepare_formfiller_template_variables(tape: FormFillerTape) -> dict[str, Any]:
    """
    Prepare common template variables from a form filler tape that are mainly used for making prompts
    """
    template_values = {
        "current_date": datetime.now().strftime("%Y-%m-%d"),
        "current_function_schema": tape.function_schema_step.model_dump_json(
            indent=2,
            exclude={"return_value"}
        ) if tape.function_schema_step else "< Not available >",
        "current_parameter": "",
        "filled_parameters": {},
        "skipped_parameters": set(),
        "remaining_parameters": [],
        "dialogue_history": [],
        "dialogue_history_json": [],
        "last_agent_request": "< Not available >",
        "last_user_message": "< Not available >",
    }

    for step in tape.steps:
        if isinstance(step, UserStep):
            template_values["dialogue_history"].append(f"User: {step.content}")
            template_values["last_user_message"] = step.content
        elif isinstance(step, AssistantStep):
            template_values["dialogue_history"].append(f"Agent: {step.content}")
            template_values["last_agent_request"] = step.content
        elif isinstance(step, UpdateFunctionParameters):
            for parameter, value in step.assign.items():
                if parameter in template_values["skipped_parameters"]:
                    template_values["skipped_parameters"].remove(parameter)
                template_values["filled_parameters"][parameter] = value

            for parameter in step.skip:
                if parameter in template_values["filled_parameters"]:
                    del template_values["filled_parameters"][parameter]
                template_values["skipped_parameters"].add(parameter)
        elif isinstance(step, RequestFunctionParameters):
            template_values["current_parameter"] = step.parameters[0]

        if isinstance(step, (Thought, Observation, Action)):
            template_values["dialogue_history_json"].append(step.model_dump())

    if not template_values["current_parameter"]:
        function_schema_step = tape.function_schema_step
        if function_schema_step is not None:
            function_parameters = function_schema_step.parameter_names
            next_parameter_to_request = [
                parameter for parameter in function_parameters 
                if parameter not in template_values["filled_parameters"] and parameter not in template_values["skipped_parameters"]
            ]
            if next_parameter_to_request:
                template_values["current_parameter"] = next_parameter_to_request[0]
                template_values["remaining_parameters"].extend(next_parameter_to_request)

    # set default values for empty skipped and filled parameters
    if not template_values["skipped_parameters"]:
        template_values["skipped_parameters"] = "No parameters skipped for now."
    if not template_values["filled_parameters"]:
        template_values["filled_parameters"] = "No parameters filled for now."
    if not template_values["remaining_parameters"]:
        template_values["remaining_parameters"] = "No parameters remained to fill."

    template_values["dialogue_history_json"] = json.dumps(template_values["dialogue_history_json"], indent=2) or "[]"

    # remove last agent message and last user message from conversation history
    if len(template_values["dialogue_history"]) > 1:
        template_values["dialogue_history"] = template_values["dialogue_history"][:-2]
    template_values["dialogue_history"] = "\n".join(template_values["dialogue_history"]) or "No dialogue history so far."

    return template_values
