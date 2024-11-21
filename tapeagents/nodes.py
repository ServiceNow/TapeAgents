import json
import logging
from typing import Any, Callable, Generator, Type, Union, get_origin

from pydantic import Field, TypeAdapter, ValidationError

from tapeagents.dialog_tape import AssistantStep, UserStep
from tapeagents.prompting import FORMALIZE_FORMAT, FORMALIZE_GUIDANCE, FORMALIZE_INPUT, FORMALIZE_SYSTEM_PROMPT
from tapeagents.view import TapeViewStack

from .agent import Node
from .core import (
    CONTROL_FLOW_STEPS,
    AgentStep,
    ConditionCheck,
    LLMOutput,
    LLMOutputParsingFailureAction,
    Observation,
    PartialStep,
    Prompt,
    Respond,
    SetNextNode,
    Step,
    StopStep,
    Tape,
)
from .llms import LLMStream
from .utils import FatalError, get_step_schemas_from_union_type, sanitize_json_completion

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MonoNode(Node):
    """
    A node for simple monolithic agents:
    - Renders the whole tape into a prompt. Trims the tape if needed.
    - Attaches a guidance text to the end of the prompt after rendering the tape.
    - Parses the llm output into provided step classes (class provided in a form of annotated union).
    """

    system_prompt: str = ""
    steps_prompt: str = ""  # prompt that describes the steps that the agent can take
    guidance: str = ""  # guidance text that is attached to the end of the prompt
    output_cls: Any = Field(exclude=True, default=None)
    next_node: str = ""

    def make_prompt(self, agent: Any, tape: Tape) -> Prompt:
        clean_tape = self.prepare_tape(tape)
        messages = self.tape_to_messages(clean_tape)
        if agent.llm.count_tokens(messages) > (agent.llm.context_size - 500):
            clean_tape = self.trim_tape(clean_tape)
            messages = self.tape_to_messages(clean_tape)
        return Prompt(messages=messages)

    def prepare_tape(self, tape: Tape) -> Tape:
        return tape

    def tape_to_steps(self, tape: Tape) -> list[Step]:
        tape_view = self.tape_view(tape)
        if tape_view:
            return [UserStep(content=tape_view)]
        else:
            view = TapeViewStack.compute(tape).top
            return view.steps

    def tape_to_messages(self, tape: Tape) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": self.system_prompt}]
        if self.output_cls is not None:
            steps_description = self.get_steps_description(tape)
            messages.append({"role": "user", "content": steps_description})
        for step in self.tape_to_steps(tape):
            if isinstance(step, CONTROL_FLOW_STEPS):
                continue
            if isinstance(step, (AssistantStep, UserStep)):
                step_role = step.kind
                step_content = step.content
            else:
                step_role = "assistant" if isinstance(step, AgentStep) else "user"
                step_content = step.llm_view()
            messages.append({"role": step_role, "content": step_content})
        if self.guidance:
            messages.append({"role": "user", "content": self.guidance})
        return messages

    def tape_view(self, tape: Tape) -> str:
        return ""

    def get_steps_description(self, tape: Tape) -> str:
        text = self.steps_prompt
        if self.output_cls:
            schema = get_step_schemas_from_union_type(self.output_cls)
            text = self.steps_prompt.format(schema=schema)
        return text

    def generate_steps(
        self, agent: Any, tape: Tape, llm_stream: LLMStream
    ) -> Generator[Step | PartialStep, None, None]:
        new_steps = []
        cnt = 0
        for event in llm_stream:
            if event.output:
                cnt += 1
                assert event.output.content
                if self.output_cls is None:
                    yield AssistantStep(content=event.output.content)
                else:
                    try:
                        for step in parse_completion(event.output.content, llm_stream.prompt.id, self.output_cls):
                            step = self.postprocess_step(tape, new_steps, step)
                            new_steps.append(step)
                            yield step
                    except Exception as e:
                        step = SetNextNode(next_node=self.name)  # stay in the same node when parsing fails
                        step = self.postprocess_step(tape, new_steps, step)
                        raise e
        if not cnt:
            raise FatalError("No completions!")

        if self.next_node and not isinstance(new_steps[-1], StopStep):
            yield SetNextNode(next_node=self.next_node)

    def postprocess_step(self, tape: Tape, new_steps: list[Step], step: Step) -> Step:
        return step

    def trim_tape(self, tape: Tape) -> Tape:
        return tape

    def make_llm_output(self, agent: Any, tape: Tape, index: int) -> LLMOutput:
        """
        Make output from steps produced by the single llm call (having the same prompt_id), except for SetNextNode steps.
        """
        steps = []
        i = index
        first_prompt_id = tape.steps[i].metadata.prompt_id
        while i < len(tape) and tape.steps[i].metadata.prompt_id == first_prompt_id:
            if not isinstance(tape.steps[i], CONTROL_FLOW_STEPS):
                steps.append(tape.steps[i])
            i += 1

        # if there is only one step, return it as a single dict, not a list
        content = [step.llm_dict() for step in steps] if len(steps) > 1 else steps[0].llm_dict()
        return LLMOutput(role="assistant", content=json.dumps(content, indent=2, ensure_ascii=False))


def parse_completion(llm_output: str, prompt_id: str, output_cls: Any) -> Generator[Step, None, None]:
    try:
        step_dicts = json.loads(sanitize_json_completion(llm_output))
        if isinstance(step_dicts, dict):
            step_dicts = [step_dicts]
    except Exception as e:
        logger.exception(f"Failed to parse LLM output as json: {llm_output}\n\nError: {e}")
        yield LLMOutputParsingFailureAction(error=f"Failed to parse LLM output as json: {e}", llm_output=llm_output)
        return
    try:
        steps = [TypeAdapter(output_cls).validate_python(step_dict) for step_dict in step_dicts]
    except ValidationError as e:
        err_text = ""
        for err in e.errors():
            loc = ".".join([str(loc) for loc in err["loc"]])
            err_text += f"{loc}: {err['msg']}\n"
        logger.exception(f"Failed to validate LLM output: {step_dicts}\n\nErrors:\n{err_text}")
        yield LLMOutputParsingFailureAction(error=f"Failed to validate LLM output: {err_text}", llm_output=llm_output)
        return
    except Exception as e:
        logger.exception(f"Failed to parse LLM output dict: {step_dicts}\n\nError: {e}")
        yield LLMOutputParsingFailureAction(error=f"Failed to parse LLM output dict: {e}", llm_output=llm_output)
        return
    for step in steps:
        step.metadata.prompt_id = prompt_id
        yield step


class Formalize(Node):
    formalize_prompt: str = FORMALIZE_FORMAT
    output_cls: Any

    def make_prompt(self, agent: Any, tape: Tape) -> Prompt:
        step = tape[-1]
        assert hasattr(step, "content")
        messages = [
            {"role": "system", "content": FORMALIZE_SYSTEM_PROMPT},
            {"role": "user", "content": FORMALIZE_INPUT.format(content=step.content)},  # type: ignore
            {"role": "user", "content": FORMALIZE_GUIDANCE},
            {
                "role": "user",
                "content": self.formalize_prompt.format(
                    schema=get_step_schemas_from_union_type(
                        Union[self.output_cls] if not get_origin(self.output_cls) == Union else self.output_cls
                    ),
                ),
            },
        ]
        return Prompt(messages=messages)

    def generate_steps(
        self, agent: Any, tape: Tape, llm_stream: LLMStream
    ) -> Generator[Step | PartialStep, None, None]:
        step = tape[-1]
        assert hasattr(step, "content")
        for event in llm_stream:
            if event.output:
                assert event.output.content
                formal_step = next(parse_completion(event.output.content, llm_stream.prompt.id, self.output_cls))
                if hasattr(formal_step, "text"):
                    formal_step.text = step.content  # type: ignore
                formal_step.metadata.prompt_id = llm_stream.prompt.id
                yield formal_step
                return
        raise FatalError("No completions!")


class ControlFlowNode(Node):
    """
    ControlFlowNode is a Node that selects another node to run based on the tape.

    Methods:
        select_node(tape: Tape) -> int:
            Abstract method to choose the next node based on the tape. Must be implemented in a subclass.
    """

    next_node: str = ""
    predicate: Callable[[Tape], bool] = lambda tape: True

    def generate_steps(
        self, agent: Any, tape: Tape, llm_stream: LLMStream
    ) -> Generator[Step | PartialStep, None, None]:
        next_node = self.select_node(tape)
        if next_node:
            yield SetNextNode(next_node=next_node)
        else:
            yield ConditionCheck()  # we should put a step into tape to know last executed node

    def select_node(self, tape: Tape) -> str:
        return self.next_node if self.predicate(tape) else ""


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


class ConditionalNode(Node):
    steps: list[Step]
    predicate: Callable[[Tape], bool] = lambda tape: True

    def generate_steps(
        self, agent: Any, tape: Tape, llm_stream: LLMStream
    ) -> Generator[Step | PartialStep, None, None]:
        if self.predicate(tape):
            for step in self.steps:
                yield step
        else:
            yield ConditionCheck()  # we should put a step into tape to know last executed node


class Return(FixedStepsNode):
    steps: list[Step] = [Respond(copy_output=True)]
