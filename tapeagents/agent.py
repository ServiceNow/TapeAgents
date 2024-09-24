from __future__ import annotations

import json
from abc import abstractmethod
from typing import Any, Callable, Generator, Generic

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny
from typing_extensions import Self

from tapeagents.observe import observe_llm_call
from tapeagents.view import TapeViewStack

from .core import (
    Action,
    AgentEvent,
    AgentStep,
    AnnotatorTapeType,
    Completion,
    LLMCall,
    MakeObservation,
    ObservationMakerTapeType,
    PartialStep,
    Prompt,
    Step,
    Tape,
    TapeMetadata,
    TapeType,
    Thought,
    TrainingText,
)
from .llms import LLM, LLMEvent, LLMStream

DEFAULT = "default"


class AgentStream(Generic[TapeType]):
    def __init__(self, generator: Generator[AgentEvent[TapeType], None, None]):
        self.generator = generator

    def __iter__(self):
        return self.generator

    def __next__(self) -> AgentEvent[TapeType]:
        return next(self.generator)

    def get_final_tape(self) -> TapeType:
        for event in self:
            if event.final_tape:
                return event.final_tape
        raise ValueError("Agent didn't produce final tape")

    def get_steps(self) -> Generator[Step, None, None]:
        for event in self:
            if event.step:
                yield event.step


# TODO: try adding node and agent types
PromptMakerFunction = Callable[[Any, Tape], Prompt]
StepsGeneratorFunction = Callable[[Any, Tape, LLMStream], Generator[Step | PartialStep, None, None]]


class Node(BaseModel):
    """
    A node in the agent's flow.
    The node is correspond to some state of the tape, has a name and contains 2 main functions:
     - one to make a prompt out of the tape
     - one to generate steps out of the received llm completion
    """

    name: str = ""
    make_prompt_func: PromptMakerFunction = Field(default=lambda agent, tape: Prompt(), exclude=True)
    generate_steps_func: StepsGeneratorFunction = Field(
        default=lambda agent, tape, llm_stream: (step for step in ()), exclude=True
    )
    make_completion_func: Callable[[Any, Tape, int], Completion] = Field(
        default=lambda agent, tape, index: Completion(role="assistant", content=tape.steps[index].content), exclude=True
    )

    def make_prompt(self, agent: Any, tape: Tape) -> Prompt:
        return self.make_prompt_func(agent, tape)

    def generate_steps(
        self, agent: Any, tape: Tape, llm_stream: LLMStream
    ) -> Generator[Step | PartialStep, None, None]:
        yield from self.generate_steps_func(agent, tape, llm_stream)

    def make_completion(self, agent: Any, tape: Tape, index: int) -> Completion:
        """"""
        return self.make_completion_func(agent, tape, index)

    def with_prompt(self, make_prompt: PromptMakerFunction) -> Node:
        self.make_prompt_func = make_prompt
        return self

    def with_generate_steps(self, generate_steps: StepsGeneratorFunction) -> Node:
        """
        Set the function that generates steps from the LLM completion
        """
        self.generate_steps_func = generate_steps
        return self

    def with_completion(self, make_completion: Callable[[Any, Tape, int], Completion]) -> Node:
        self.make_completion_func = make_completion
        return self

    def with_fixed_steps(self, steps: list[Step]) -> Node:
        """
        Use fixed steps instead of generating them from the LLM completion for that node
        """
        self.generate_steps_func = lambda agent, tape, llm_stream: (yield from steps)  # type: ignore
        return self


class Agent(BaseModel, Generic[TapeType]):
    name: str = ""
    llms: dict[str, SerializeAsAny[LLM]] = {}
    # can't use list[Agent] because of pydantic bug
    # https://github.com/pydantic/pydantic/issues/9969
    subagents: list[Any] = []
    templates: dict[str, str] = {}
    flow: list[SerializeAsAny[Node]] = Field(
        default_factory=lambda: [],
        description="List of nodes in the agent's flow, order of the list used to determine the prioirty during activation",
    )
    max_iterations: int = 100

    _manager: Any | None = None

    model_config = ConfigDict(extra="forbid")

    def model_post_init(self, __context: Any) -> None:
        if not self.name:
            # by default use the class name without the values for type variables
            # e.g. "Agent" instea of "Agent[Tape[...]]""
            self.name = self.__class__.__name__.split("[")[0]
        names = set()
        for agent in self.subagents:
            names.add(agent.name)
            if isinstance(agent, Agent):
                if agent._manager is not None:
                    raise ValueError("Agent is already a subagent of another agent. Make a copy of your agent.")
                agent._manager = self
            else:
                raise ValueError("Subagents must be instances of Agent")
        if len(names) < len(self.subagents):
            raise ValueError("Subagents have duplicate names")
        return super().model_post_init(__context)

    @property
    def manager(self):
        if self._manager is None:
            raise ValueError("Agent doesn't have a manager")
        return self._manager

    @property
    def llm(self):
        if len(self.llms) > 1:
            raise ValueError("Agent has multiple LLMs. Use llms property to access a specific one.")
        return self.llms[DEFAULT]

    @property
    def template(self):
        if len(self.templates) > 1:
            raise ValueError("Agent has multiple templates. Use templates property to access a specific one.")
        return self.templates[DEFAULT]

    @property
    def full_name(self):
        if self._manager is None:
            return self.name
        return f"{self._manager.full_name}/{self.name}"

    def find_subagent(self, name: str):
        for agent in self.subagents:
            if agent.name == name:
                return agent
        raise ValueError(f"Agent {name} not found")

    def get_subagent_names(self) -> list[str]:
        return [agent.name for agent in self.subagents]

    @classmethod
    def create(
        cls,
        llms: dict[str, LLM] | LLM | None = {},
        subagents: list[Agent[TapeType]] | None = [],
        templates: dict[str, str] | str | None = {},
        **kwargs,
    ) -> Self:
        """The user-friendly way to create an agent that flexible-typed inputs.

        A subclass can override this method and extend its signature.

        """
        if isinstance(llms, LLM):
            llms = {DEFAULT: llms}
        if isinstance(templates, str):
            templates = {DEFAULT: templates}
        if subagents:
            subagents = list(subagents)
        if templates:
            templates = dict(templates)
        extra_kwargs = {
            k: v
            for k, v in {
                "llms": llms,
                "subagents": subagents,
                "templates": templates,
            }.items()
            if v is not None
        }

        return cls(**(extra_kwargs | kwargs))

    def update(self, agent_config: dict[str, Any]) -> Agent[TapeType]:
        """
        Reload the configuration of all llms and subagents while keeping their classes.

        :param obj: the new configuration
        """
        if not set(self.llms.keys()) == set(agent_config["llms"].keys()):
            raise ValueError("Agent has different LLMs than the new configuration.")
        llms = {name: llm.model_validate(agent_config["llms"][name]) for name, llm in self.llms.items()}
        if len(self.subagents) != len(agent_config["subagents"]):
            raise ValueError("Agent has different number of subagents than the new configuration.")
        # recurse into subagents
        subagents = [
            subagent.model_validate(subagent.update(subagent_obj))
            for subagent, subagent_obj in zip(self.subagents, agent_config["subagents"])
        ]
        config_copy = agent_config.copy()
        config_copy["llms"] = llms
        config_copy["subagents"] = subagents
        return type(self).model_validate(config_copy)

    @property
    def signature(self):
        return ""

    def compute_view(self, tape: TapeType) -> TapeViewStack:
        return TapeViewStack.compute(tape)

    def select_node(self, tape: TapeType) -> Node:
        """
        Select the node to run next based on the current state of the tape.

        :param tape: the tape to make the decision on
        :return: the node to run next
        """
        return self.flow[self.compute_view(tape).top.next_node]

    def make_prompt(self, tape: TapeType) -> Prompt:
        """Make the prompt for the next iteration of the agent.

        Can return prompt with no messages, which means the agent should generate next
        steps by following rules, with no help from the LLM. An agent that only delegate
        to subagents may not need to implement this method.
        """
        return self.select_node(tape).make_prompt(self, tape)

    def generate_steps(self, tape: TapeType, llm_stream: LLMStream) -> Generator[Step | PartialStep, None, None]:
        """
        Generate new steps and other events by feeding the prompt to the LLM

        :param tape: the tape to generate steps for
        :param llm_stream: the stream of tokens from the LLM
        :return: a generator of steps (or partial steps) to append to the tape
        """
        yield from self.select_node(tape).generate_steps(self, tape, llm_stream)

    def make_completion(self, tape: TapeType, index: int) -> Completion:
        return self.select_node(tape[:index]).make_completion(self, tape, index)

    def delegate(self, tape: TapeType) -> Agent[TapeType]:
        view = self.compute_view(tape)
        result = self
        for frame in view.stack[1:]:
            result = result.find_subagent(frame.agent_name)
        return result

    def is_agent_step(self, step: Step) -> bool:
        """Check if the step was produced by the agent or by the environment."""
        return isinstance(step, (Action, Thought))

    def should_stop(self, tape: TapeType) -> bool:
        """Check if the agent should stop its turn and wait for observations."""
        return isinstance(tape.steps[-1], Action)

    def run_iteration(
        self, tape: TapeType, llm_stream: LLMStream | None = None
    ) -> Generator[Step | PartialStep, None, None]:
        """Run one iteration of the agent (assuming one call to the underlyng model)

        During an iteration the agent generates steps from a stream of tokens that arises
        from a single LLM call with a single prompt. An agent can do multiple iterations
        before returning the next action (see `run` method).

        This function can also take a given `llm_stream`, which can be useful when the agent
        reuses a tape.

        """
        new_steps = []
        if llm_stream is None:
            prompt = self.make_prompt(tape)
            if len(self.llms) > 1:
                raise NotImplementedError("TODO: implement LLM choice in the prompt")
            llm_stream = self.llm.generate(prompt) if prompt else LLMStream(None, prompt)
        for item in self.generate_steps(tape, llm_stream):
            if isinstance(item, AgentStep):
                item.metadata.prompt_id = llm_stream.prompt.id
                yield item
                new_steps.append(item)
            else:
                yield item

    def run(self, tape: TapeType) -> AgentStream[TapeType]:
        """
        Run the agent on the tape until it produces a stop step, but no more than max_iterations.
        """

        def _run_implementation():
            nonlocal tape
            n_iterations = 0
            input_tape_length = len(tape)
            stop = False
            while n_iterations < self.max_iterations and not stop:
                current_subagent = self.delegate(tape)
                for step in current_subagent.run_iteration(tape):
                    if isinstance(step, PartialStep):
                        yield AgentEvent(partial_step=step)
                    elif isinstance(step, AgentStep):
                        step.metadata.by = current_subagent.full_name
                        tape = tape.append(step)
                        yield AgentEvent(step=step, partial_tape=tape)
                        if self.should_stop(tape):
                            stop = True
                    else:
                        raise ValueError("Agent can only generate steps or partial steps")
                n_iterations += 1
            final_tape = tape
            final_tape.metadata = TapeMetadata(
                n_added_steps=len(tape) - input_tape_length, parent_id=tape.metadata.id, author=self.signature
            )
            yield AgentEvent(final_tape=final_tape)

        return AgentStream(_run_implementation())

    def reuse(self, tape: TapeType) -> tuple[TapeType, list[LLMCall]]:
        """Reuse another agent's tape as one's own.

        Construct completions at each step where a prompt is made. Check that completion
        parsing yield the same steps as in the original tape. Rewrite metadata for all steps.

        """
        reused_steps = []
        llm_calls = []
        i = 0
        while i < len(tape):
            past_tape = tape[:i]
            step = tape.steps[i]
            if self.is_agent_step(step):
                current_agent = self.delegate(past_tape)
                prompt = current_agent.make_prompt(past_tape)
                completion = current_agent.make_completion(tape, i)
                llm_call = LLMCall(prompt=prompt, completion=completion, cached=True)
                observe_llm_call(llm_call)

                # Validate that the reconstructed llm call leads to the same steps as in the given tape
                def _generator():
                    yield LLMEvent(completion=completion)

                new_steps = list(current_agent.run_iteration(past_tape, LLMStream(_generator(), prompt)))
                for j, new_step in enumerate(new_steps):
                    assert isinstance(new_step, Step)
                    old_step = tape.steps[i + j]
                    if type(old_step) is not type(new_step) or not _is_step_data_equal(old_step, new_step):
                        raise TapeReuseFailure(
                            f"Can't reuse tape because regenerated step {i + j} data doesn't match"
                            f"\nold step data: {old_step.llm_dict()}\nnew step data: {new_step.llm_dict()}",
                            partial_tape=past_tape,
                        )
                llm_calls.append(llm_call)
                reused_steps.extend(new_steps)
                i += len(new_steps)
            else:
                reused_steps.append(step)
                i += 1
        reused_tape = tape.model_validate(dict(context=tape.context, metadata=TapeMetadata(), steps=reused_steps))
        return reused_tape, llm_calls

    def make_training_text(self, llm_call: LLMCall) -> TrainingText:
        """Routes the request to make trace to the appropriate agent's LLM."""
        # TODO: support more than 1 LLM
        return self.llm.make_training_text(llm_call.prompt, llm_call.completion)

    def make_training_data(self, tape: TapeType) -> list[TrainingText]:
        _, llm_calls = self.reuse(tape)
        return [self.make_training_text(llm_call) for llm_call in llm_calls]


class TapeReuseFailure(ValueError):
    def __init__(self, msg: str, partial_tape: Tape):
        self.partial_tape = partial_tape
        super().__init__(msg)


def _is_step_data_equal(step1: Step, step2: Step) -> bool:
    """Compare steps ignoring the metadata.

    Some LLM API messages include metadata, like unique tool call ids, that
    we must store and send back to the API. This function excludes this metadata from the comparison.

    Another caveat is that some data, like function call arguments in LLM messages, is not deserialized,
    and hence can be tricky to compare across steps. This function deserializes known fields like that before comparison..

    """

    def just_data(step: Step) -> dict:
        data = step.llm_dict()
        for tc in data.get("tool_calls", []):
            tc.pop("id", None)
            if function := tc.get("function"):
                function["arguments"] = json.loads(function["arguments"])
        data.pop("tool_call_id", None)
        return data

    return just_data(step1) == just_data(step2)


class Annotator(Agent[AnnotatorTapeType], Generic[TapeType, AnnotatorTapeType]):
    @abstractmethod
    def make_own_tape(self, tape: TapeType) -> AnnotatorTapeType:
        pass

    def annotate(self, tape: TapeType) -> AnnotatorTapeType:
        return self.run(self.make_own_tape(tape)).get_final_tape()

    def get_annotation(self, own_tape: AnnotatorTapeType) -> Any:
        return own_tape.steps[-1].annotation


class ObservationMaker(Agent[ObservationMakerTapeType], Generic[TapeType, ObservationMakerTapeType]):
    @abstractmethod
    def make_own_tape(self, tape: TapeType) -> ObservationMakerTapeType:
        pass

    def add_observation(self, tape: TapeType, own_tape: ObservationMakerTapeType) -> TapeType:
        last_step = own_tape.steps[-1]
        if not isinstance(own_tape.steps[-1], MakeObservation):
            raise ValueError("Own tape must end with MakeObservation to add observation to the target tape")
        metadata = TapeMetadata(
            parent_id=tape.metadata.id, author_tape_id=own_tape.metadata.id, author=self.signature, n_added_steps=1
        )
        return tape.append(last_step.new_observation).model_copy(update=dict(metadata=metadata))

    def can_continue(self, _: TapeType) -> bool:
        """Check if this observation maker can continue the given tape."""
        return True

    def continue_tape(self, tape: TapeType) -> TapeType:
        own_tape = self.make_own_tape(tape)
        for event in self.run(own_tape):
            if event.final_tape:
                return self.add_observation(tape, event.final_tape)
        raise ValueError("Observation maker didn't produce final tape")
