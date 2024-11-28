import json
import logging
from typing import Any, Generator, Type

from pydantic import Field, TypeAdapter, ValidationError

from .agent import Node
from .core import (
    AgentStep,
    LLMOutput,
    LLMOutputParsingFailureAction,
    Observation,
    PartialStep,
    Prompt,
    SetNextNode,
    Step,
    StopStep,
    Tape,
)
from .llms import LLMStream
from .utils import FatalError, sanitize_json_completion

logger = logging.getLogger(__name__)


class MonoNode(Node):
    """
    A node for simple monolithic agents:
    - Renders the whole tape into a prompt. Trims the tape if needed.
    - Attaches a guidance text to the end of the prompt after rendering the tape.
    - Parses the llm output into provided step classes (class provided in a form of annotated union).
    """

    guidance: str = ""  # guidance text that is attached to the end of the prompt
    system_prompt: str = ""
    steps_prompt: str = ""  # prompt that describes the steps that the agent can take
    agent_step_cls: Any = Field(exclude=True)
    next_node: str = ""

    def make_prompt(self, agent: Any, tape: Tape) -> Prompt:
        cleaned_tape = self.prepare_tape(tape)
        steps_description = self.get_steps_description(tape, agent)
        messages = self.tape_to_messages(cleaned_tape, steps_description)
        if agent.llm.count_tokens(messages) > (agent.llm.context_size - 500):
            cleaned_tape = self.trim_tape(cleaned_tape)
        messages = self.tape_to_messages(cleaned_tape, steps_description)
        return Prompt(messages=messages)

    def prepare_tape(self, tape: Tape) -> Tape:
        steps_without_control_flow = [step for step in tape.steps if not isinstance(step, SetNextNode)]
        return tape.model_copy(update=dict(steps=steps_without_control_flow))

    def make_llm_output(self, agent: Any, tape: Tape, index: int) -> LLMOutput:
        """
        Make output from steps produced by the single llm call (having the same prompt_id), except for SetNextNode steps.
        """
        steps = []
        i = index
        first_prompt_id = tape.steps[i].metadata.prompt_id
        while i < len(tape) and tape.steps[i].metadata.prompt_id == first_prompt_id:
            if not isinstance(tape.steps[i], SetNextNode):
                steps.append(tape.steps[i])
            i += 1

        # if there is only one step, return it as a single dict, not a list
        content = [step.llm_dict() for step in steps] if len(steps) > 1 else steps[0].llm_dict()
        return LLMOutput(role="assistant", content=json.dumps(content, indent=2, ensure_ascii=False))

    def tape_to_messages(self, tape: Tape, steps_description: str) -> list[dict]:
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
        ]
        if steps_description:
            messages.append({"role": "user", "content": steps_description})
        for step in tape:
            role = "assistant" if isinstance(step, AgentStep) else "user"
            messages.append({"role": role, "content": step.llm_view()})
        if self.guidance:
            messages.append({"role": "user", "content": self.guidance})
        return messages

    def get_steps_description(self, tape: Tape, agent: Any) -> str:
        return self.steps_prompt

    def generate_steps(
        self, agent: Any, tape: Tape, llm_stream: LLMStream
    ) -> Generator[Step | PartialStep, None, None]:
        new_steps = []
        try:
            cnt = 0
            for event in llm_stream:
                if event.output:
                    cnt += 1
                    assert event.output.content
                    for step in self.parse_completion(event.output.content, llm_stream.prompt.id):
                        step = self.postprocess_step(tape, new_steps, step)
                        new_steps.append(step)
                        yield step
            if not cnt:
                raise FatalError("No completions!")
        except FatalError:
            raise

        if self.next_node and not isinstance(new_steps[-1], StopStep):
            yield SetNextNode(next_node=self.next_node)

    def postprocess_step(self, tape: Tape, new_steps: list[Step], step: Step) -> Step:
        return step

    def parse_completion(self, llm_output: str, prompt_id: str) -> Generator[Step, None, None]:
        try:
            step_dicts = json.loads(sanitize_json_completion(llm_output))
            if isinstance(step_dicts, dict):
                step_dicts = [step_dicts]
        except Exception as e:
            logger.exception(f"Failed to parse LLM output as json: {llm_output}\n\nError: {e}")
            yield LLMOutputParsingFailureAction(error=f"Failed to parse LLM output as json: {e}", llm_output=llm_output)
            return
        try:
            steps = [TypeAdapter(self.agent_step_cls).validate_python(step_dict) for step_dict in step_dicts]
        except ValidationError as e:
            err_text = ""
            for err in e.errors():
                loc = ".".join([str(loc) for loc in err["loc"]])
                err_text += f"{loc}: {err['msg']}\n"
            logger.exception(f"Failed to validate LLM output: {step_dicts}\n\nErrors:\n{err_text}")
            yield LLMOutputParsingFailureAction(
                error=f"Failed to validate LLM output: {err_text}", llm_output=llm_output
            )
            return
        except Exception as e:
            logger.exception(f"Failed to parse LLM output dict: {step_dicts}\n\nError: {e}")
            yield LLMOutputParsingFailureAction(error=f"Failed to parse LLM output dict: {e}", llm_output=llm_output)
            return
        for step in steps:
            step.metadata.prompt_id = prompt_id
            yield step

    def trim_tape(self, tape: Tape) -> Tape:
        return tape


class ControlFlowNode(Node):
    """
    ControlFlowNode is a Node that selects another node to run based on the tape.

    Methods:
        select_node(tape: Tape) -> int:
            Abstract method to choose the next node based on the tape. Must be implemented in a subclass.
    """

    def generate_steps(
        self, agent: Any, tape: Tape, llm_stream: LLMStream
    ) -> Generator[Step | PartialStep, None, None]:
        yield SetNextNode(next_node=self.select_node(tape))

    def select_node(self, tape: Tape) -> str:
        raise NotImplementedError("Implement this method in the subclass to set the next node according to your logic")


class ObservationControlNode(ControlFlowNode):
    """
    ObservationControlNode is a ControlFlowNode that selects the next node based on the last observation in the tape.
    """

    observation_to_node: dict[Type, str] = {}
    default_node: str = ""  # jump to the last node by default

    def select_node(self, tape: Tape) -> str:
        observations = [step for step in tape.steps if isinstance(step, Observation)]
        last_observation = observations[-1] if observations else None
        return self.observation_to_node.get(type(last_observation), self.default_node)


class FixedStepsNode(Node):
    steps: list[Step]

    def generate_steps(
        self, agent: Any, tape: Tape, llm_stream: LLMStream
    ) -> Generator[Step | PartialStep, None, None]:
        for step in self.steps:
            yield step
