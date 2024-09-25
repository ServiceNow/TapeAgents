from __future__ import annotations

import datetime
import json
from typing import Any, Generic, Iterable, Iterator, Literal, TypeAlias, TypeVar
from uuid import uuid4

import litellm
from pydantic import BaseModel, Field, SerializeAsAny
from typing_extensions import Self


class TrainingText(BaseModel):
    """
    Data sample to finetune a language model
    """

    text: str
    n_predicted: int

    @property
    def prompt_str(self) -> str:
        return self.text[: -self.n_predicted]

    @property
    def output_str(self) -> str:
        return self.text[-self.n_predicted :]


class StepMetadata(BaseModel):
    """
    Metadata for the step
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    prompt_id: str = ""
    node: str = ""
    agent: str = ""
    other: dict[str, Any] = Field(default_factory=dict)


class Step(BaseModel):
    metadata: StepMetadata = StepMetadata()

    def llm_dict(self) -> dict[str, Any]:
        """Dump step data only, drop the metadata"""
        return self.model_dump(exclude_none=True, exclude={"metadata"})

    def llm_view(self, indent: int = 2) -> str:
        return json.dumps(self.llm_dict(), indent=indent, ensure_ascii=False)


class PartialStep(BaseModel):
    """Wrap your step as partial step to indicate that it is not finished yet."""

    step: Step


class Observation(Step):
    pass


class Error(Observation):
    pass


class AgentStep(Step):
    pass

    def by_node(self, node_name: str) -> Self:
        """
        Set the node_name that is being run when the step is produced
        """
        self.metadata.node = node_name
        return self


class Thought(AgentStep):
    pass


class Action(AgentStep):
    pass


class AgentResponseParsingFailureAction(Action):
    """
    Action produced automatically when the agent response parsing failed
    """

    kind: Literal["agent_response_parsing_failure_action"] = "agent_response_parsing_failure_action"
    error: str


class FinalStep(Action):
    kind: Literal["final_step"] = "final_step"
    reason: str = ""


class Jump(Thought):
    kind: Literal["jump"] = "jump"
    next_node: int


class Pass(Thought):
    kind: Literal["pass"] = "pass"


StepType = TypeVar("StepType", bound=Action | Observation | Thought)


class TapeMetadata(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    parent_id: str | None = None
    author: str | None = None
    author_tape_id: str | None = None
    n_added_steps: int = 0
    error: Any | None = None
    result: Any = None


ContextType = TypeVar("ContextType")


class Tape(BaseModel, Generic[ContextType, StepType]):
    """
    A sequence of steps produced by agents and environments
    """

    metadata: TapeMetadata = TapeMetadata()
    context: ContextType | None = None
    steps: list[StepType] = []

    def __iter__(self) -> Iterator[StepType]:  # type: ignore
        return iter(self.steps)

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, key: int | slice) -> StepType | Self:
        if isinstance(key, slice):  # cut and erase metadata
            return self.model_copy(update=dict(steps=self.steps[key.start : key.stop], metadata=TapeMetadata()))
        return self.steps[key]

    def __add__(self, tape: Self | Iterable[Step]) -> Self:
        """
        Concatenate two tapes or append list of steps to the tape
        """
        new_steps = tape.steps if isinstance(tape, Tape) else list(tape)
        return self.model_copy(
            update=dict(
                steps=self.steps + new_steps,
                metadata=TapeMetadata(n_added_steps=len(new_steps)),
            )
        )

    def append(self, step: StepType) -> Self:
        """
        Add a step to the tape
        """
        return self.model_copy(update=dict(steps=self.steps + [step], metadata=TapeMetadata(n_added_steps=1)))

    def with_new_id(self) -> Self:
        return self.model_copy(update=dict(metadata=TapeMetadata()))

    def as_prompt_messages(self) -> list[dict]:
        messages = []
        for step in self.steps:
            if isinstance(step, (Pass, Jump)):
                continue
            llm_message = self.step_to_message(step)
            messages.append(llm_message)
        return messages

    def step_to_message(self, step: Step) -> dict[str, str]:
        message = step.llm_dict()
        kind = message.pop("kind", None)
        if kind == "system":
            role = "system"
        elif kind == "tool":
            role = "tool"
        elif isinstance(step, (Thought, Action)):
            role = "assistant"
        elif isinstance(step, Observation):
            role = "user"
        else:
            raise ValueError(f"Cannot convert step type: {step} to role")
        llm_message = {"role": role, **message}
        return llm_message


TapeType = TypeVar("TapeType", bound=Tape)


class Prompt(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    tools: list[dict] | None = None
    messages: list[dict] = []

    @staticmethod
    def from_user_message(content: str) -> Prompt:
        return Prompt(messages=[{"role": "user", "content": content}])

    def __bool__(self) -> bool:
        return bool(self.messages)


LLMMessage: TypeAlias = litellm.utils.Message
LLMOutput: TypeAlias = LLMMessage


class LLMCall(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    prompt: Prompt
    output: LLMOutput
    prompt_length_tokens: int = -1
    output_length_tokens: int = -1
    cached: bool


AnnotatorTape = Tape[TapeType, StepType]
AnnotatorTapeType = TypeVar("AnnotatorTapeType", bound=AnnotatorTape)


class AnnotationWithMetadata(BaseModel):
    annotation: Any
    annotator_tape_id: str


class Episode(BaseModel):
    """
    Auxiliary data structure for tape with annotations attached.
    Humans may want to look at a tape with one-or-many annotations attached to agent's actions.
    This is a data structure to store such tapes. We store the tape id for traceability.
    """

    tape: Tape
    annotator_tapes: dict[int, list[Tape]]
    obs_making_tapes: dict[int, Tape]

    def group_by_step(self) -> Iterator[tuple[Tape | None, Step, list[Tape]]]:
        for i, step in enumerate(self.tape.steps):
            yield (
                self.obs_making_tapes.get(i, None),
                step,
                self.annotator_tapes.get(i, []),
            )


class AgentEvent(BaseModel, Generic[TapeType]):
    """
    Event produced by the agent during the run.
    Can contain a step, a final tape with all new steps, or a partial step when used in streaming mode.
    Fields are mutually exclusive.
    """

    step: SerializeAsAny[Step] | None = None
    partial_step: PartialStep | None = None
    partial_tape: TapeType | None = None
    final_tape: TapeType | None = None


ObservationMakerTapeType = TypeVar("ObservationMakerTapeType", bound=Tape)


class MakeObservation(Action, Generic[StepType]):
    new_observation: StepType
