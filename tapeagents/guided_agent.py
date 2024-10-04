import json
import logging
from typing import Any, Generator, Generic

from pydantic import TypeAdapter, ValidationError
from typing_extensions import Self

from .agent import Agent, Node
from .core import (
    AgentResponseParsingFailureAction,
    AgentStep,
    LLMOutput,
    PartialStep,
    Prompt,
    Step,
    Tape,
    TapeType,
)
from .llms import LLM, LLMStream
from .utils import FatalError, sanitize_json_completion

logger = logging.getLogger(__name__)


class GuidanceNode(Node):
    """
    A node for the guided agent.
    Validates that the tape starts with a specific step class.
    Attaches a guidance text to the end of the prompt after rendering the tape.
    Parses the llm output into provided step classes (class provided in a form of annotated union).
    Trims the tape if needed.
    """

    guidance: str
    system_prompt: str = ""
    steps_prompt: str = ""
    agent_step_cls: Any = None
    start_step_cls: Any = None

    def make_prompt(self, agent: Any, tape: Tape) -> Prompt:
        assert isinstance(tape.steps[0], self.start_step_cls)
        cleaned_tape = self.prepare_tape(tape)
        steps_description = self.get_steps_description(tape, agent)
        messages = self.tape_to_messages(cleaned_tape, steps_description)
        if agent.llm.count_tokens(messages) > (agent.llm.context_size - 500):
            cleaned_tape = agent.trim_tape(cleaned_tape)
        messages = self.tape_to_messages(cleaned_tape, steps_description)
        return Prompt(messages=messages)

    def prepare_tape(self, tape: Tape) -> Tape:
        return tape

    def make_llm_output(self, tape: Tape, index: int) -> LLMOutput:
        return LLMOutput(role="assistant", content=tape.steps[index].llm_view())

    def tape_to_messages(self, tape: Tape, steps_description: str) -> list[dict]:
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": steps_description},
        ]
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

    def postprocess_step(self, tape: Tape, new_steps: list[Step], step: Step) -> Step:
        return step

    def parse_completion(self, completion: str, prompt_id: str) -> Generator[Step, None, None]:
        try:
            step_dicts = json.loads(sanitize_json_completion(completion))
            if isinstance(step_dicts, dict):
                step_dicts = [step_dicts]
        except Exception as e:
            logger.exception(f"Failed to parse agent output: {completion}\n\nError: {e}")
            yield AgentResponseParsingFailureAction(error=f"Failed to parse agent output: {completion}\n\nError: {e}")
            return
        try:
            steps = [TypeAdapter(self.agent_step_cls).validate_python(step_dict) for step_dict in step_dicts]
        except ValidationError as e:
            err_text = ""
            for err in e.errors():
                loc = ".".join([str(loc) for loc in err["loc"]])
                err_text += f"{loc}: {err['msg']}\n"
            logger.exception(f"Failed to validate agent output: {step_dicts}\n\nErrors:\n{err_text}")
            yield AgentResponseParsingFailureAction(
                error=f"Failed to validate agent output: {step_dicts}\n\nErrors:\n{err_text}"
            )
            return
        except Exception as e:
            logger.exception(f"Failed to parse agent output dict: {step_dicts}\n\nError: {e}")
            yield AgentResponseParsingFailureAction(
                error=f"Failed to parse agent output dict: {step_dicts}\n\nError: {e}"
            )
            return
        for step in steps:
            step.metadata.prompt_id = prompt_id
            yield step


class GuidedAgent(Agent, Generic[TapeType]):
    """
    Generic agent class which renders all tape steps into prompt and parses the llm completion into a sequence of steps.
    Main features:
    - selects guidance node based on the kind of the last step in the tape.
    - selected node does the following:
        - validates that the tape starts with a specific step class.
        - attaches a guidance prompt text to the end of the prompt after rendering the tape.
        - trims the tape if the total token count exceeds the context size.
    """

    nodes: list[GuidanceNode]  # type: ignore

    def select_node(self, tape: TapeType) -> Node:
        last_kind = tape.steps[-1].kind
        for node in self.nodes:
            if last_kind == node.name:
                return node
        return self.nodes[-1]  # default to the last node

    def delegate(self, tape: TapeType):
        return self

    def trim_tape(self, tape: Tape) -> Tape:
        return tape
