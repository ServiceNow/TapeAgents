"""
Core data structures for the tape agents framework.
"""

from __future__ import annotations

import datetime
import json
from typing import Any, Generic, Iterable, Iterator, List, Literal, TypeAlias, TypeVar
from uuid import uuid4

import litellm
from pydantic import BaseModel, Field, SerializeAsAny
from typing_extensions import Self


class TrainingText(BaseModel):
    """
    Training text instance used to finetune a language model.

    Attributes:
        text (str): The full text of the training instance.
        n_predicted (int): The number of predicted tokens in the text.
        reward (float): The reward associated with the training instance. Defaults to 0.0.
        logprobs (List[float]): A list of log probabilities of the completion tokens from the assistant model.
        ref_logprobs (List[float]): A list of reference log probabilities of the completion tokens from the reference model.
        group_id (str, optional): ID of the group. It is used by the RL finetuning script to normalize rewards.
        prompt_text (str): Portion of the text that serves as the prompt (i.e., the text excluding the predicted tokens).
        output_text (str): Portion of the text that represents the predicted output (i.e., the last n_predicted tokens).
    """

    text: str
    n_predicted: int
    reward: float = 0.0
    logprobs: List[float] = Field(default_factory=list)
    ref_logprobs: List[float] = Field(default_factory=list)
    input_ids: List[int] = Field(default_factory=list)
    labels: List[int] = Field(default_factory=list)
    group_id: str | None = None

    @property
    def prompt_text(self) -> str:
        return self.text[: -self.n_predicted]

    @property
    def output_text(self) -> str:
        return self.text[-self.n_predicted :]


class StepMetadata(BaseModel):
    """
    StepMetadata is a model that represents metadata for a tape step.

    Attributes:
        id (str): A unique identifier for the step, generated using uuid4.
        prompt_id (str): An identifier for the llm prompt used to produce the step.
        node (str): The node that produced the step.
        agent (str): The agent that produced the step.
        other (dict[str, Any]): A dictionary to store additional metadata related to the step.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    prompt_id: str = ""
    node: str = ""
    agent: str = ""
    llm: str = ""
    other: dict[str, Any] = Field(default_factory=dict)


class Step(BaseModel):
    """
    Base class representing a step in a tape.

    Attributes:
        metadata (StepMetadata): Metadata associated with the step.
        kind (Literal["define_me"]): A placeholder value indicating the kind of step.
                                     This should be overwritten in subclasses.
    """

    metadata: StepMetadata = Field(default_factory=StepMetadata)
    kind: Literal["define_me"] = "define_me"  # This is a placeholder value, it should be overwritten in subclasses

    def llm_dict(self) -> dict[str, Any]:
        """Dumps the step data as dictionary, excluding the metadata."""
        return self.model_dump(exclude_none=True, exclude={"metadata"})

    def llm_view(self, indent: int | None = 2) -> str | list[dict]:
        """Returns a JSON string representation of the step data, excluding the metadata."""
        return json.dumps(self.llm_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def get_kind(cls) -> str:
        """Returns the default value of the 'kind' field."""
        return cls.model_fields["kind"].default


class PartialStep(BaseModel):
    """
    Wrap your step as partial step to indicate that it is not finished yet.

    Attributes:
        step (Step): An instance of the Step class representing the step details.
    """

    step: Step


class Observation(Step):
    """
    Base class representing an observation in a tape.
    """

    def short_view(self) -> str:
        """Returns a short string representation of the observation when the tape needs to be trimmed."""
        return self.llm_view()


class Error(Step):
    """
    Base class representing an error in a tape.
    """

    pass


class AgentStep(Step):
    """
    Base class representing a step produced by an agent.
    """

    pass


class Thought(AgentStep):
    """
    Base class representing an agent's thought in a tape.
    """

    pass


class Action(AgentStep):
    """
    Base class representing an agent's action in a tape.
    """

    pass


class LLMOutputParsingFailureAction(Action, Error):
    """
    Represents an action that is produced automatically when the LLM output parsing fails.

    Attributes:
        kind (Literal["llm_output_parsing_failure_action"]): A constant string indicating the type of action.
        error (str): A description of the error that occurred during parsing.
        llm_output (str): The raw output from the LLM that could not be parsed.
    """

    kind: Literal["llm_output_parsing_failure_action"] = "llm_output_parsing_failure_action"
    error: str
    llm_output: str


class StopStep(Action):
    """
    Action that stops orchestrator loop.
    """

    pass


class FinalStep(StopStep):
    """
    Action that stops orchestrator loop with a reason.

    Attributes:
        kind (Literal["final_step"]): A constant string indicating the type of action.
        reason (str): The reason for stopping the orchestr
    """

    kind: Literal["final_step"] = "final_step"
    reason: str = ""


class SetNextNode(Thought):
    """
    Action that sets the next node to run in the current agent.

    Attributes:
        kind (Literal["set_next_node"]): A constant string indicating the type of action.
        next_node (str): The name of the next node to run.
    """

    kind: Literal["set_next_node"] = "set_next_node"
    next_node: str


class Pass(Thought):
    """
    Action that does nothing.
    """

    kind: Literal["pass"] = "pass"


class Call(Thought):
    """
    Action that calls another agent.

    Attributes:
        kind (Literal["call"]): A constant string indicating the type of action.
        content (str): The content passed with the call to the agent.
        agent_name (str): The name of the agent to call.
    """

    kind: Literal["call"] = "call"
    content: str = ""
    agent_name: str


class Respond(Thought):
    """
    Action that returns a response to the top-level agent after processing the call.

    Attributes:
        kind (Literal["respond"]): A constant string indicating the type of action.
        content (str): The content of the response.
        copy_output (bool): Indicates whether the last step before this one should be copied to the top-level agent.
    """

    content: str = ""
    kind: Literal["respond"] = "respond"
    copy_output: bool = False


StepType = TypeVar("StepType", bound=Action | Observation | Thought)
"""Type variable for step types."""


class TapeMetadata(BaseModel):
    """
    TapeMetadata is a model that represents metadata information for a tape.

    Attributes:
        id (str): A unique identifier for the tape, generated by default using uuid4.
        parent_id (str, optional): An optional identifier for the parent tape.
        author (str, optional): An optional name of the author of the tape.
        author_tape_id (str, optional): An optional identifier for the author's tape.
        n_added_steps (int, optional): The number of steps added to the tape in the last agent run, default is 0.
        error (Any, optional): An optional field to store any errors occured during the last agent run.
        result (Any, optional): Optional field to store the result associated with the tape, default is an empty dictionary.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    parent_id: str | None = None
    author: str | None = None
    author_tape_id: str | None = None
    n_added_steps: int = 0
    error: Any | None = None
    result: Any = {}


ContextType = TypeVar("ContextType")
"""Type variable for context types."""


class Tape(BaseModel, Generic[ContextType, StepType]):
    """
    Tape class represents a sequence of steps produced by agents and environments with associated metadata.

    Supports iteration, slicing, concatenation, and appending of steps.

    Attributes:
        metadata (TapeMetadata): Metadata associated with the tape.
        context (ContextType, optional): Context information for the tape.
        steps (List[StepType]): List of steps in the tape.
    """

    metadata: TapeMetadata = Field(default_factory=TapeMetadata)
    context: ContextType | None = None
    steps: List[StepType] = Field(default_factory=list)

    def __iter__(self) -> Iterator[StepType]:  # type: ignore
        return iter(self.steps)

    def __len__(self) -> int:
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
        Adds a step to the tape and creates new metadata (new tape id, etc...).
        """
        return self.model_copy(update=dict(steps=self.steps + [step], metadata=TapeMetadata()))

    def with_new_id(self) -> Self:
        """
        Creates a copy of the tape with new metadata.
        """
        return self.model_copy(update=dict(metadata=TapeMetadata()))


TapeType = TypeVar("TapeType", bound=Tape)
"""Type variable for tape types."""


class Prompt(BaseModel):
    """
    A class representing a LLM prompt with messages and tools.

    Attributes:
        id (str): A unique identifier for the prompt, generated by default.
        tools (list[dict], optional): A list of tools associated with the prompt, default is None.
        messages (list[dict]): A list of messages in the prompt, default is an empty list.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    tools: list[dict] | None = None
    messages: list[dict] = Field(default_factory=list)
    token_ids: list[int] = Field(default_factory=list)

    @staticmethod
    def from_user_message(content: str) -> Prompt:
        """
        Creates a Prompt instance from a user message.

        Args:
            content (str): The content of the user message.

        Returns:
            Prompt: A Prompt instance with the user message.
        """
        return Prompt(messages=[{"role": "user", "content": content}])

    def __bool__(self) -> bool:
        return bool(self.messages)


LLMOutput: TypeAlias = litellm.utils.Message
"""Type alias for the output of the language model."""


class LLMCall(BaseModel):
    """
    LLMCall stores info about a call to a language model.

    Attributes:
        timestamp (str): The timestamp when the call was made, in ISO 8601 format.
        prompt (Prompt): The input prompt provided to the language model.
        output (LLMOutput): The output generated by the language model.
        prompt_length_tokens (int): The length of the prompt in tokens. Defaults to -1 if not set.
        output_length_tokens (int): The length of the output in tokens. Defaults to -1 if not set.
        cached (bool): Indicates whether the result was retrieved from cache.
    """

    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    prompt: Prompt
    output: LLMOutput
    prompt_length_tokens: int = -1
    output_length_tokens: int = -1
    cached: bool
    llm_info: dict = {}
    cost: float = 0
    logprobs: list[TokenLogprob] = Field(default_factory=list, exclude=True)


class TokenLogprob(BaseModel):
    logprob: float
    token_id: int
    generated: int


AnnotatorTape = Tape[TapeType, StepType]
"""Type alias for annotator tapes."""
AnnotatorTapeType = TypeVar("AnnotatorTapeType", bound=AnnotatorTape)
"""Type variable for annotator tape types."""


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
    Attributes are mutually exclusive.

    Attributes:
        step (SerializeAsAny[Step], optional): A step produced by the agent.
        partial_step (PartialStep, optional): A partial step produced by the agent.
        partial_tape (TapeType, optional): A partial tape produced by the agent.
        final_tape (TapeType, optional): A final tape produced by the agent.
    """

    step: SerializeAsAny[Step] | None = None
    partial_step: PartialStep | None = None
    partial_tape: TapeType | None = None
    final_tape: TapeType | None = None


ObservationMakerTapeType = TypeVar("ObservationMakerTapeType", bound=Tape)


class MakeObservation(Action, Generic[StepType]):
    kind: Literal["make_observation"] = "make_observation"
    new_observation: StepType

    def llm_dict(self) -> dict[str, Any]:
        """Dumps the step data as dictionary, excluding the metadata of the step itself and the metadata of the wrapped step"""
        obj = self.model_dump(exclude_none=True, exclude={"metadata"})
        del obj["new_observation"]["metadata"]
        return obj
