import json
import logging
from typing import Any, Generator, Generic

from pydantic import TypeAdapter, ValidationError

from .agent import Agent
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
from .dialog import AssistantStep, SystemStep, UserStep
from .llms import LLMStream
from .utils import FatalError, sanitize_json_completion

logger = logging.getLogger(__name__)


class GuidedAgent(Agent, Generic[TapeType]):
    """
    Generic agent class which renders all tape steps into prompt and parses the llm completion into a sequence of steps.
    Main features:
    - validates that the tape starts with a specific step class.
    - attaches a guidance prompt text to the end of the prompt after rendering the tape.
    - selects guidance based on the kind of the last step in the tape from the templates dictionary.
    - trims the tape if the total token count exceeds the context size.
    """

    _start_step_cls: Any
    _agent_step_cls: Any
    templates: dict[str, str] = {}
    max_iterations: int = 2

    def delegate(self, tape: TapeType):
        return self

    def get_steps_description(self, tape) -> str:
        return self.templates["allowed_steps"]

    def make_prompt(self, tape: Tape) -> Prompt:
        assert isinstance(tape.steps[0], self._start_step_cls)

        cleaned_tape = self.prepare_tape(tape)
        messages = self.tape_to_messages(cleaned_tape)
        if self.llm.count_tokens(messages) > (self.llm.context_size - 500):
            cleaned_tape = self.trim_tape(cleaned_tape)
        messages = self.tape_to_messages(cleaned_tape)
        return Prompt(messages=messages)

    def make_completion(self, tape: TapeType, index: int) -> LLMOutput:
        return LLMOutput(role="assistant", content=tape.steps[index].llm_view())

    def prepare_tape(self, tape: Tape) -> Tape:
        return tape

    def trim_tape(self, tape: Tape) -> Tape:
        return tape

    def tape_to_messages(self, tape: Tape) -> list[dict]:
        messages: list[dict] = [
            {"role": "system", "content": self.templates["system_prompt"]},
            {"role": "user", "content": self.get_steps_description(tape)},
        ]
        for step in tape:
            role = "assistant" if isinstance(step, AgentStep) else "user"
            messages.append({"role": role, "content": step.llm_view()})
        if tape.steps[-1].kind in self.templates:
            guidance = self.templates[tape.steps[-1].kind]
            messages.append({"role": "user", "content": guidance})
        elif "_default" in self.templates:
            messages.append({"role": "user", "content": self.templates["_default"]})
        return messages

    def generate_steps(self, tape: Tape, llm_stream: LLMStream) -> Generator[Step | PartialStep, None, None]:
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
            steps = [TypeAdapter(self._agent_step_cls).validate_python(step_dict) for step_dict in step_dicts]
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
            step.prompt_id = prompt_id
            yield step
