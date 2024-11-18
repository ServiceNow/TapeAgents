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
    SetNextNode,
    Step,
    StopStep,
    Tape,
)
from .llms import LLM, LLMStream
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

    def tape_to_messages(self, tape: Tape, steps_description: str) -> list[dict]:
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": steps_description},
        ]
        view = TapeViewStack.compute(tape).top
        for step in view.steps:
            if isinstance(step, CONTROL_FLOW_STEPS):
                continue
            elif isinstance(step, (AssistantStep, UserStep)):
                message = {"role": step.kind, "content": step.content}
            else:
                role = "assistant" if isinstance(step, AgentStep) else "user"
                message = {"role": role, "content": step.llm_view()}
            messages.append(message)
        if self.guidance:
            messages.append({"role": "user", "content": self.guidance})
        return messages

    def get_steps_description(self, tape: Tape, agent: Any) -> str:
        return self.steps_prompt

    def generate_steps(
        self, agent: Any, tape: Tape, llm_stream: LLMStream
    ) -> Generator[Step | PartialStep, None, None]:
        new_steps = []
        cnt = 0
        for event in llm_stream:
            if event.output:
                cnt += 1
                assert event.output.content
                try:
                    for step in parse_completion(event.output.content, llm_stream.prompt.id, self.agent_step_cls):
                        step = self.postprocess_step(tape, new_steps, step)
                        new_steps.append(step)
                        yield step
                except Exception as e:
                    yield SetNextNode(next_node=self.name)  # stay in the same node when parsing fails
                    raise e
        if not cnt:
            raise FatalError("No completions!")

        if self.next_node and not isinstance(new_steps[-1], StopStep):
            yield SetNextNode(next_node=self.next_node)

    def postprocess_step(self, tape: Tape, new_steps: list[Step], step: Step) -> Step:
        return step

    def trim_tape(self, tape: Tape) -> Tape:
        return tape


def parse_completion(llm_output: str, prompt_id: str, agent_step_cls: Any) -> Generator[Step, None, None]:
    try:
        step_dicts = json.loads(sanitize_json_completion(llm_output))
        if isinstance(step_dicts, dict):
            step_dicts = [step_dicts]
    except Exception as e:
        logger.exception(f"Failed to parse LLM output as json: {llm_output}\n\nError: {e}")
        yield LLMOutputParsingFailureAction(error=f"Failed to parse LLM output as json: {e}", llm_output=llm_output)
        return
    try:
        steps = [TypeAdapter(agent_step_cls).validate_python(step_dict) for step_dict in step_dicts]
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


class ThinkingNode(Node):
    """
    Produce plain text thought
    """

    system_prompt: str
    guidance: str
    next_node: str = ""
    output_cls: Any = None

    def make_prompt(self, agent: Any, tape: Tape) -> Prompt:
        messages = self.tape_to_messages(tape)
        return Prompt(messages=messages)

    def tape_to_messages(self, tape: Tape) -> list[dict]:
        messages = [{"role": "system", "content": self.system_prompt}]
        tape_view = self.tape_view(tape)
        if tape_view:
            messages.append({"role": "user", "content": tape_view})
        else:
            view = TapeViewStack.compute(tape).top
            for step in view.steps:
                if isinstance(step, CONTROL_FLOW_STEPS):
                    continue
                elif isinstance(step, (AssistantStep, UserStep)):
                    message = {"role": step.kind, "content": step.content}
                else:
                    role = "assistant" if isinstance(step, AgentStep) else "user"
                    message = {"role": role, "content": step.llm_view()}
                messages.append(message)
        messages.append({"role": "user", "content": self.guidance})
        return messages

    def generate_steps(self, agent: Any, tape: Tape, llm_stream: LLMStream):
        for event in llm_stream:
            if event.output:
                assert event.output.content
                step = AssistantStep(content=event.output.content)
                yield step
                if self.output_cls:
                    formal_step = self.formalize(step, agent.llm)
                    yield formal_step

        if self.next_node:
            yield SetNextNode(next_node=self.next_node)

    def set_subagent_task(self, tape: Tape, new_steps: list[Step]) -> dict:
        steps = tape.steps + new_steps
        return steps[-1].llm_dict()

    def tape_view(self, tape: Tape) -> str:
        return ""

    def formalize(self, step: AssistantStep, llm: LLM) -> Step:
        messages = [
            {"role": "system", "content": FORMALIZE_SYSTEM_PROMPT},
            {"role": "user", "content": FORMALIZE_INPUT.format(content=step.content)},
            {"role": "user", "content": FORMALIZE_GUIDANCE},
            {
                "role": "user",
                "content": FORMALIZE_FORMAT.format(
                    schema=get_step_schemas_from_union_type(
                        Union[self.output_cls] if not get_origin(self.output_cls) == Union else self.output_cls
                    ),
                ),
            },
        ]
        llm_stream = llm.generate(Prompt(messages=messages))
        for event in llm_stream:
            if event.output:
                assert event.output.content
                return next(parse_completion(event.output.content, llm_stream.prompt.id, self.output_cls))
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
