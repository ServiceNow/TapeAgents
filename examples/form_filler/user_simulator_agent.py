import json
import logging
from typing import Literal, Type

import numpy as np
import pydantic
from pydantic import BaseModel

from tapeagents.agent import Node, ObservationMaker, TapeMetadata
from tapeagents.core import Action, Error, MakeObservation, Prompt, SetNextNode, StepMetadata, Tape, Thought
from tapeagents.dialog_tape import UserStep
from tapeagents.llms import LLMStream

from .environment import FormFillerEnvironment
from .error import FormFillerStateError
from .state import compute_form_filler_state
from .steps import RequestFunctionParameters
from .tape import FormFillerTape
from .user_simulator_filters import (
    AlwaysTrue,
    extract_placeholder_keys,
    get_step_instruction_params,
)
from .utils import get_step_cls, render_chat_template

logger = logging.getLogger(__name__)


class SampleUserInstructionThought(Thought):
    kind: Literal["sample_user_instruction_thought"] = "sample_user_instruction_thought"  # type: ignore
    instruction_alias: str
    instruction: str


class UserSimulatorError(Action, Error):
    kind: Literal["user_simulator_error"] = "user_simulator_error"  # type: ignore
    error: str


# The user simulator tape wraps the formfiller tape as its context
# It produces (potentially some intermediate steps) and a final step
UserSimulatorTape = Tape[FormFillerTape, SampleUserInstructionThought | MakeObservation[UserStep] | UserSimulatorError]


class InstructionSampler(BaseModel):
    instructions: dict[str, str]
    weights: dict[str, float]

    @pydantic.model_validator(mode="after")
    def validate_weights(self):
        # Verify that weights are positive and keys match
        if not all(w >= 0 for w in self.weights.values()):
            raise ValueError("Weights must be non-negative.")
        if set(self.instructions.keys()) != set(self.weights.keys()):
            raise ValueError("Instruction keys and weights keys must match.")
        return self

    def sample_instruction(self) -> SampleUserInstructionThought:
        import random

        if not self.instructions or not self.weights:
            raise ValueError("Instructions and weights must be provided for sampling.")

        normalized_weights = [self.weights[key] for key in self.instructions.keys()]
        total_weight = sum(normalized_weights)
        if total_weight == 0:
            raise ValueError("Total weight must be greater than zero for sampling.")
        normalized_weights = [w / total_weight for w in normalized_weights]

        sampled_key = random.choices(list(self.instructions.keys()), weights=normalized_weights, k=1)[0]

        sample_user_instruction_thought = SampleUserInstructionThought(
            instruction_alias=sampled_key,
            instruction=self.instructions[sampled_key],
        )

        return sample_user_instruction_thought

    @classmethod
    def from_str(cls, instruction_str: str) -> "InstructionSampler":
        # When given a plain string, use it as a single instruction
        return cls(instructions={"default_instruction": instruction_str}, weights={"default_instruction": 1.0})


class UserSimulatorAgent(ObservationMaker[FormFillerTape, UserSimulatorTape]):
    nodes: list[Node] = []
    behavior_alias: str
    prompt_template: Prompt
    instruction: InstructionSampler
    step_types_chain: list[str] = []
    _step_types_chain: list[Type] = []  # same but the str are converted to types
    step_filters: list = []
    _step_filters: list = []  # same but padded to match the length of step_types_chain

    @classmethod
    def create(cls, instruction: str | InstructionSampler, **kwargs) -> "UserSimulatorAgent":
        if isinstance(instruction, str):
            instruction = InstructionSampler.from_str(instruction)

        nodes = [
            SampleUserInstructionNode(),
            UserSimulatorMainNode(),
        ]
        user_simulator_agent = super().create(name="user_simulator", nodes=nodes, instruction=instruction, **kwargs)
        user_simulator_agent._step_types_chain = [get_step_cls(name) for name in user_simulator_agent.step_types_chain]

        assert len(user_simulator_agent.step_filters) <= len(
            user_simulator_agent._step_types_chain
        ), "Step filters must be less than or equal to the step types chain"

        # Fill step_filters with True to match the length of step_types_chain
        user_simulator_agent._step_filters = user_simulator_agent.step_filters + [
            AlwaysTrue()
            for __ in range(len(user_simulator_agent._step_types_chain) - len(user_simulator_agent.step_filters))
        ]

        return user_simulator_agent

    def make_own_tape(self, tape: FormFillerTape) -> UserSimulatorTape:
        # Hacky stuff to remove things like setnextnode so that filters and placeholder work
        tape_without_setnextnode = tape.model_copy(
            update={"steps": [step for step in tape.steps if not isinstance(step, SetNextNode)]}
        )

        # This just wraps the input formfiller tape into a user simulator tape's context
        user_simulator_tape = UserSimulatorTape(
            metadata=TapeMetadata(
                parent_id=tape.metadata.id,
                author="user_simulator",
            ),
            context=tape_without_setnextnode,  # use input tape as context
            steps=[],
        )
        return user_simulator_tape

    def can_continue(self, tape: FormFillerTape) -> bool:
        # Hacky stuff to remove things like setnextnode so that filters and placeholder work
        tape_without_setnextnode = tape.model_copy(
            update={"steps": [step for step in tape.steps if not isinstance(step, SetNextNode)]}
        )

        # If the tape is too short, we can't continue
        if len(tape_without_setnextnode.steps) < len(self.step_types_chain):
            return False

        # Compute the state
        state = compute_form_filler_state(tape_without_setnextnode)
        if isinstance(state, FormFillerStateError):
            logger.error(
                f"Failed to compute state for tape_without_setnextnode {tape_without_setnextnode.metadata.id}: {state}"
            )
            return False

        # only assess the last steps
        last_steps = tape_without_setnextnode.steps[-len(self._step_types_chain) :]
        offset = len(tape_without_setnextnode.steps) - len(last_steps)

        # Verify step class[k]
        target_step_classes = [step_type.__name__ for step_type in self._step_types_chain]
        actual_step_classes = [step.__class__.__name__ for step in last_steps]

        # Verify step class[k] and step filter[k] match
        for i, (step, step_type, step_filter) in enumerate(zip(last_steps, self._step_types_chain, self._step_filters)):
            actual_index = i + offset
            if not isinstance(step, step_type):
                logger.debug(
                    f"Step {actual_index} of tape_without_setnextnode {tape_without_setnextnode.metadata.id} is not of type {step_type}: "
                )
                logger.debug(f"Expected: {target_step_classes}, got: {actual_step_classes}")
                return False
            if not step_filter.filter(state, step):
                logger.debug(
                    f"Step {actual_index} of tape_without_setnextnode {tape_without_setnextnode.metadata.id} failed the filter {repr(step_filter)}"
                )
                return False

        logger.debug(f"Tape {tape.metadata.id} is continuable")
        return True


class SampleUserInstructionNode(Node):
    name: str = "sample_instruction"

    # No prompt needed
    def generate_steps(self, agent: UserSimulatorAgent, tape: UserSimulatorTape, llm_stream: LLMStream):
        instruction_thought = agent.instruction.sample_instruction()
        yield instruction_thought


class UserSimulatorMainNode(Node):
    name: str = "main_node"

    def make_prompt(self, agent: UserSimulatorAgent, tape: UserSimulatorTape) -> Prompt:
        # Render the prompt template
        assert isinstance(tape.context, FormFillerTape), "UserSimulatorTape must have FormFillerTape context"
        assert isinstance(
            tape.steps[-1], SampleUserInstructionThought
        ), "Last step must be SampleUserInstructionThought"

        # Put the rendered dialogue and instruction into the main prompt template
        rendered_dialogue = tape.context.render_dialogue_as_text()
        instruction = tape.steps[-1].instruction
        prompt_template_with_instruction = render_chat_template(
            agent.prompt_template.messages,
            dict(
                text=rendered_dialogue,
                instruction=instruction,
            ),
        )

        # Now get the placeholders for the instruction
        placeholder_values = {}

        # 1) Get placeholders for function schemas
        # Use a hack to access the envrionemnt directly to get schemas
        assert tape.context.context is not None
        environment = FormFillerEnvironment.from_spec(tape.context.context.env_spec)
        sampled_schema = np.random.choice(list(environment.available_schemas))
        parameters = environment.available_schemas[sampled_schema].with_replaced_refs.model_dump()["parameters"]
        schema_placeholders = {
            "schema_description": environment.available_schemas[sampled_schema].description,
            "schema_parameters": json.dumps(parameters, indent=2),
            "schema_name": environment.available_schemas[sampled_schema].name,
            "all_schema_descriptions": json.dumps(
                dict((schema.name, schema.description) for schema in environment.available_schemas.values()), indent=2
            ),
            "all_schema_names": json.dumps(
                [schema.name for schema in environment.available_schemas.values()], indent=2
            ),
        }
        placeholder_values.update(schema_placeholders)

        # 2) Optionally, get placeholders on RequestFunctionParameters
        # In some cases, the user simulator assumes steps[-2] is RequestFunctionParameters
        # and uses it to generate extra placeholders
        if len(agent._step_types_chain) >= 2 and agent._step_types_chain[-2] == RequestFunctionParameters:
            state = compute_form_filler_state(tape.context)
            assert not isinstance(state, FormFillerStateError), "State should not be an error"

            request_function_parameters_step = tape.context.steps[-2]
            assert isinstance(
                request_function_parameters_step, RequestFunctionParameters
            ), f"Expected RequestFunctionParameters, got {request_function_parameters_step}"

            extra_placeholders = get_step_instruction_params(request_function_parameters_step, state)
            placeholder_values.update(extra_placeholders)

        # FINALLY: replace the placeholders in the prompt template
        rendered_prompt = render_chat_template(prompt_template_with_instruction, placeholder_values)

        remaining_placeholders = extract_placeholder_keys(rendered_prompt)
        assert not remaining_placeholders, f"Remaining placeholders in prompt: {remaining_placeholders}"

        logger.debug(f"Rendered prompt: {rendered_prompt}")
        return Prompt(messages=rendered_prompt)

    def generate_steps(self, agent: UserSimulatorAgent, tape: UserSimulatorTape, llm_stream: LLMStream):
        user_utterance = llm_stream.get_text()
        step = UserStep(
            content=user_utterance,
            metadata=StepMetadata(other={"alias": agent.behavior_alias}),
        )
        yield MakeObservation[UserStep](new_observation=step)
