import json
import logging
from typing import Any, Generator, Generic

from pydantic import TypeAdapter, ValidationError

from .agent import Agent, Node
from .core import (
    AgentResponseParsingFailureAction,
    AgentStep,
    StepMetadata,
    LLMOutput,
    PartialStep,
    Prompt,
    Step,
    Tape,
    TapeType,
)
from .llms import LLMStream
from .utils import FatalError, sanitize_json_completion

logger = logging.getLogger(__name__)


class MonoNode(Node):
    """
    A node for the monolithic agent.
    - Renders the whole tape into a prompt. Trims the tape if needed.
    - Attaches a guidance text to the end of the prompt after rendering the tape.
    - Parses the llm output into provided step classes (class provided in a form of annotated union).
    """

    trigger_step: str | list[str]  # which step kind in the end of the tape triggers this node
    guidance: str  # guidance text that is attached to the end of the prompt
    system_prompt: str = ""
    steps_prompt: str = ""  # prompt that describes the steps that the agent can take
    agent_step_cls: Any = None

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
        if isinstance(tape.steps[index], AgentResponseParsingFailureAction):
            # FIXME: this is a hack to log the completion to train the agent
            return LLMOutput(role="assistant", content=tape.steps[index].metadata.other["completion"])
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
            yield AgentResponseParsingFailureAction(
                error=f"Failed to parse agent output: {completion}\n\nError: {e}",
                metadata=StepMetadata(other={"completion": completion}),
            )
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
                error=f"Failed to validate agent output: {step_dicts}\n\nErrors:\n{err_text}",
                metadata=StepMetadata(other={"completion": completion}),
            )
            return
        except Exception as e:
            logger.exception(f"Failed to parse agent output dict: {step_dicts}\n\nError: {e}")
            yield AgentResponseParsingFailureAction(
                error=f"Failed to parse agent output dict: {step_dicts}\n\nError: {e}",
                metadata=StepMetadata(other={"completion": completion}),
            )
            return
        for step in steps:
            step.metadata.prompt_id = prompt_id
            yield step

    def trim_tape(self, tape: Tape) -> Tape:
        return tape


class MonoAgent(Agent, Generic[TapeType]):
    """
    Monolithic agent which selects the node based on the last step in the tape.
    """

    nodes: list[MonoNode]  # type: ignore

    def select_node(self, tape: TapeType) -> Node:
        last_kind = tape.steps[-1].kind
        for node in self.nodes:
            if (isinstance(node.trigger_step, str) and last_kind == node.trigger_step) or (
                isinstance(node.trigger_step, list) and last_kind in node.trigger_step
            ):
                return node
        return self.nodes[-1]  # default to the last node

    def delegate(self, tape: TapeType):
        """
        Does not support delegation to subagents.
        """
        return self
