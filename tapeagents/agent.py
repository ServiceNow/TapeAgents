"""
Base classes for agents and nodes.
"""

from __future__ import annotations

import json
import logging
from abc import abstractmethod
from typing import Any, Generator, Generic

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny
from typing_extensions import Self

from tapeagents.core import (
    Action,
    AgentEvent,
    AgentStep,
    AnnotatorTapeType,
    LLMOutputParsingFailureAction,
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
from tapeagents.llms import LLM, LLMCall, LLMEvent, LLMOutput, LLMStream, TrainableLLM
from tapeagents.observe import observe_llm_call
from tapeagents.view import TapeViewStack

DEFAULT = "default"

logger = logging.getLogger(__name__)


class AgentStream(Generic[TapeType]):
    """
    A wrapper around a generator that produces AgentEvents, representing the result of an agent run.

    The generator can be iterated over to get the events, or the final tape can be extracted with get_final_tape.
    Support iterable protocol and generator protocol.

    Attributes:
        generator (Generator[AgentEvent[TapeType], None, None]): The generator that produces AgentEvents.
    """

    def __init__(self, generator: Generator[AgentEvent[TapeType], None, None]):
        self.generator = generator

    def __iter__(self):
        return self.generator

    def __next__(self) -> AgentEvent[TapeType]:
        return next(self.generator)

    def get_final_tape(self) -> TapeType:
        """
        Retrieve the final tape from the agent's events.

        Iterates through the events of the agent and returns the final tape
        if it is found. If no final tape is produced by the agent, a ValueError
        is raised.

        Returns:
            TapeType: The final tape produced by the agent.

        Raises:
            ValueError: If the agent did not produce a final tape.
        """
        for event in self:
            if event.final_tape:
                return event.final_tape
        raise ValueError("Agent didn't produce final tape")

    def get_steps(self) -> Generator[Step, None, None]:
        """
        Generator function that yields steps from events.

        Yields:
            Step: The step associated with each event that has a step.
        """
        for event in self:
            if event.step:
                yield event.step


class Node(BaseModel):
    """
    A node in the agent, atomic unit of the agent's behavior.

    The agent chooses which node to run based on the current tape.
    The node has a name and contains 2 main functions:

    - make a prompt out of the tape
    - generate steps out of the received llm output

    Attributes:
        name (str): The name of the node. Defaults to an empty string.
    """

    name: str = ""
    llm: str = DEFAULT

    def model_post_init(self, __context: Any) -> None:
        if not self.name:
            self.name = self.__class__.__name__.split("[")[0]  # class name without type variables

    def make_prompt(self, agent: Any, tape: Tape) -> Prompt:
        """
        Creates a prompt for the given agent and tape.

        Args:
            agent (Any): The agent for which the prompt is being created.
            tape (Tape): The tape associated with the agent.

        Returns:
            Prompt: The generated prompt.
        """
        return Prompt()

    def generate_steps(
        self, agent: Any, tape: Tape, llm_stream: LLMStream
    ) -> Generator[Step | PartialStep, None, None]:
        """
        Generates steps for the given agent, tape, and LLM stream.

        Args:
            agent (Any): The agent for which steps are to be generated.
            tape (Tape): The tape object containing relevant data.
            llm_stream (LLMStream): The LLM stream to be used for generating steps.

        Yields:
            Union[Step, PartialStep]: The generated steps or partial steps.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError("Node must implement generate_steps")

    def make_llm_output(self, agent: Any, tape: Tape, index: int) -> LLMOutput:
        """
        Generates an LLMOutput object for a given agent and tape at a specified index.

        Args:
            agent (Any): The agent for which the LLMOutput is being generated.
            tape (Tape): The tape containing the steps.
            index (int): The index of the step in the tape from which to generate the output.

        Returns:
            LLMOutput: An object containing the role and content for the LLM output.
        """
        return LLMOutput(role="assistant", content=tape.steps[index].content)


class Agent(BaseModel, Generic[TapeType]):
    """
    Base class for agents within the TapeAgents framework.

    An agent is a model that can run on a tape, generating new steps based on the tape's state.
    The agent can have subagents, which are other agents that it manages and can delegate to.
    The agent can also have nodes, which are atomic units of the agent's behavior that it can choose to run based on the tape.

    Attributes:
        name (str): The unique name of the agent.
        llms (dict[str, SerializeAsAny[LLM]]): A dictionary mapping names to LLM instances used by the agent.
        subagents (list[Any]): A list of subagents managed by this agent. Subagents must have unique names.
        templates (dict[str, Any]): A dictionary of templates used for generating prompts.
        nodes (list[SerializeAsAny[Node]]): A list of nodes that define the agent's actions and decision points.
        max_iterations (int): The maximum number of iterations the agent will execute before stopping.
        manager (Agent): Retrieves the manager agent overseeing this agent.
        llm (LLM): Default language model if only one is configured.
        template (Template): Default template if only one is configured.
        full_name (str): Hierarchical name of the agent, including its manager hierarchy.


    Raises:
        ValueError: If configuration inconsistencies are detected:

            - If a subagent is already managed by another agent
            - If any subagent is not an instance of Agent class
            - If there are duplicate names among subagents
            - If there are duplicate names among nodes

    """

    name: str = ""
    llms: dict[str, SerializeAsAny[LLM]] = {}
    # can't use list[Agent] because of pydantic bug
    # https://github.com/pydantic/pydantic/issues/9969
    subagents: list[Any] = Field(
        default_factory=lambda: [],
        description="List of subagents, which are agents that are run by this agent. Subagents must have unique names.",
    )
    templates: dict[str, Any] = {}
    nodes: list[SerializeAsAny[Node]] = Field(
        default_factory=lambda: [],
        description="List of nodes in the agent, order of the list used to determine the priority during activation. Nodes must have unique names.",
    )
    known_actions: list[type[Action]] = Field(default_factory=list)
    tools_description: str = ""
    max_iterations: int = 100
    store_llm_calls: bool = False

    _manager: Any | None = None

    model_config = ConfigDict(extra="forbid")

    def model_post_init(self, __context: Any) -> None:
        if not self.name:
            # by default use the class name without the values for type variables
            # e.g. "Agent" instead of "Agent[Tape[...]]""
            self.name = self.__class__.__name__.split("[")[0]
        names = set()
        for i, agent in enumerate(self.subagents):
            if isinstance(agent, Agent):
                if agent._manager is not None:
                    raise ValueError("Agent is already a subagent of another agent. Make a copy of your agent.")
                agent._manager = self
            else:
                raise ValueError("Subagents must be instances of Agent")
            if agent.name in names:
                raise ValueError(
                    f'Duplicate subagent name "{agent.name}" in subagent {i}, pass a unique name to the subagent during creation'
                )
            names.add(agent.name)
        node_names = set()
        for i, node in enumerate(self.nodes):
            if node.name in node_names:
                raise ValueError(
                    f'Duplicate node name "{node.name}" in node {i}, pass a unique name to the node during creation'
                )
            if self.llms and node.llm not in self.llms:
                raise ValueError(
                    f"Node {node.name} references unknown LLM {node.llm}. Known LLMs: {list(self.llms.keys())}"
                )
            if hasattr(node, "add_known_actions"):
                node.add_known_actions(self.known_actions)
            node_names.add(node.name)
        return super().model_post_init(__context)

    @property
    def manager(self):
        """
        Gets the manager of the agent.

        Returns:
            (Agent): The manager agent instance.

        Raises:
            ValueError: If the agent doesn't have a manager assigned.
        """
        if self._manager is None:
            raise ValueError("Agent doesn't have a manager")
        return self._manager

    @property
    def llm(self):
        """
        Get the default LLM instance associated with the agent.

        Returns:
            (LLM): The default LLM instance if only one LLM is configured.

        Raises:
            ValueError: If multiple LLMs are configured for this agent. In this case, use the `llms`
                       property to access specific LLM instances.
        """
        if len(self.llms) > 1:
            raise ValueError("Agent has multiple LLMs. Use llms property to access a specific one.")
        return self.llms[DEFAULT]

    @property
    def template(self):
        """
        Returns the default template of the agent.

        This property provides access to the default template when the agent has exactly one template.
        If multiple templates exist, it raises a ValueError indicating that specific templates
        should be accessed through the templates property instead.

        Returns:
            (Template): The default template object.

        Raises:
            ValueError: If the agent has more than one template.
            IndexError: If no templates exist (implicitly through list access).
        """
        if len(self.templates) > 1:
            raise ValueError("Agent has multiple templates. Use templates property to access a specific one.")
        return self.templates[DEFAULT]

    @property
    def full_name(self):
        """Returns the full hierarchical name of the agent.

        The full name is constructed by combining the manager's full name (if present)
        with this agent's name, separated by a forward slash. If the agent has no
        manager, returns just the agent's name.

        Returns:
            (str): The full hierarchical name path of the agent. Examples: "agent_name" (no manager), "manager_name/agent_name" (with manager)
        """
        if self._manager is None:
            return self.name
        return f"{self._manager.full_name}/{self.name}"

    def find_subagent(self, name: str):
        """
        Find a subagent by name in the list of subagents.

        Args:
            name (str): The name of the subagent to find.

        Returns:
            (Agent): The found subagent instance.

        Raises:
            ValueError: If no subagent with the given name is found.
        """
        for agent in self.subagents:
            if agent.name == name:
                return agent
        raise ValueError(f"Agent {name} not found")

    def find_node(self, name: str):
        """Find a node by its name in the list of nodes.

        Args:
            name (str): The name of the node to find.

        Returns:
            (Node): The node with the matching name.

        Raises:
            ValueError: If no node with the given name is found.
        """
        for node in self.nodes:
            if node.name == name:
                return node
        raise ValueError(f"Node {name} not found")

    def get_subagent_names(self) -> list[str]:
        """
        Returns a list of names of all subagents.

        Returns:
            list[str]: A list containing the names of all subagents in the agent.
        """
        return [agent.name for agent in self.subagents]

    def clone(self) -> Self:
        """
        Creates a deep copy of the current agent instance.

        This method creates an independent copy of the agent with all its attributes,
        but detaches it from any manager.

        Returns:
            Self: A new instance of the agent with identical attributes but no manager.
        """
        result = self.model_copy(deep=True)
        result._manager = None
        return result

    @classmethod
    def create(
        cls, llms: dict[str, LLM] | LLM | None = None, templates: dict[str, Any] | str | None = None, **kwargs
    ) -> Self:
        """
        Creates an instance of the class with provided LLMs and templates.

        Args:
            llms (Union[Dict[str, LLM], LLM, None]): Language model(s) to use. Can be:

                - A dictionary mapping names to LLM instances
                - A single LLM instance (will be mapped to default name)
                - None (empty dict will be used)
            templates (Union[Dict[str, Any], str, None]): Template(s) to use. Can be:

                - A dictionary mapping names to template configurations
                - A single template string (will be mapped to default name)
                - None (no templates will be used)
            **kwargs (dict, optional): Additional keyword arguments to pass to the class constructor

        Returns:
            Self: A new instance of the class initialized with the provided arguments

        Example:
            ```python
            agent = Agent.create(llm)  # Single LLM
            agent = Agent.create({"gpt": llm1, "claude": llm2})  # Multiple LLMs
            agent = Agent.create(llm, "template")  # LLM with template
            ```
        """
        if isinstance(llms, LLM):
            llms = {DEFAULT: llms}
        if isinstance(templates, str):
            templates = {DEFAULT: templates}
        if templates:
            kwargs["templates"] = templates

        return cls(llms=llms or {}, **kwargs)

    def update(self, agent_config: dict[str, Any]) -> Agent[TapeType]:
        """
        Updates the agent's configuration while preserving instance types.

        This method allows reconfiguration of the agent while maintaining the class types
        of LLMs and subagents. It performs a deep update by recursively applying changes
        to nested components.

        Args:
            agent_config (dict[str, Any]): New configuration dictionary containing LLMs,
                subagents, templates and other agent settings.

        Returns:
            Agent[TapeType]: A new agent instance with updated configuration.

        Raises:
            ValueError: If the new configuration has different LLMs or number of subagents
                than the current agent.

        Note:
            - Only string templates are updated, complex template objects are preserved
            - Node configurations are preserved to avoid potential issues
        """

        if not set(self.llms.keys()) == set(agent_config["llms"].keys()):
            raise ValueError("Agent has different LLMs than the new configuration.")
        if len(self.subagents) != len(agent_config["subagents"]):
            raise ValueError("Agent has different number of subagents than the new configuration.")
        # recurse into subagents
        subagents = [
            subagent.model_validate(subagent.update(subagent_obj))
            for subagent, subagent_obj in zip(self.subagents, agent_config["subagents"])
        ]
        # recurse into llms
        llms = {name: llm.model_validate(agent_config["llms"][name]) for name, llm in self.llms.items()}
        # only update templates are str
        templates = {
            name: (value if isinstance(value, str) else self.templates[name])
            for name, value in agent_config["templates"].items()
        }
        config_copy = agent_config.copy()
        config_copy["llms"] = llms
        config_copy["subagents"] = subagents
        config_copy["templates"] = templates
        # do not update nodes for now to avoid tricky bugs
        config_copy["nodes"] = self.nodes
        return type(self).model_validate(config_copy)

    def compute_view(self, tape: TapeType) -> TapeViewStack:
        """
        Compute the view stack from a given tape.

        Args:
            tape (TapeType): The input tape to process.

        Returns:
            TapeViewStack: A stack of views computed from the input tape.
        """
        return TapeViewStack.compute(tape)

    def select_node(self, tape: TapeType) -> Node:
        """
        Select the next node to execute based on the current state of the tape.

        The selection process follows these rules:
            1. If next_node is explicitly set in the tape view, return that node
            2. If no nodes have been run yet (last_node is None), return the first node
            3. Return the node that follows the last executed node in the list

        Args:
            tape (TapeType): The tape containing execution state and data

        Returns:
            Node: The next node to be executed

        Raises:
            ValueError: If unable to determine the next node to execute (e.g., reached end of list)
        """
        # Select the node to run next based on the current state of the tape.
        view = self.compute_view(tape).top
        if view.next_node:
            logger.debug(f"{self.name}: Next node was set explicitly in the tape: {view.next_node}")
            return self.find_node(view.next_node)

        if not view.last_node:
            logger.debug(f"{self.name}: No nodes have been run yet, select node 0: {self.nodes[0].name}")
            return self.nodes[0]

        # Select the next node that stored after the last node found in the tape
        logger.debug(f"{self.name}: Last node in view: {view.last_node}")
        logger.debug(f"{self.name}: Known nodes: {[node.name for node in self.nodes]}")
        for i, node in enumerate(self.nodes):
            if node.name == view.last_node and i + 1 < len(self.nodes):
                logger.debug(f"{self.name}: Select immediate next node: {self.nodes[i + 1].name}")
                return self.nodes[i + 1]
        raise ValueError("Next node not found")

    def make_prompt(self, tape: TapeType) -> Prompt:
        """
        Makes the prompt for the next iteration of the agent.
        This method generates a prompt by delegating to the selected node's make_prompt method.
        Can return a prompt with no messages, indicating the agent should generate next steps
        by following rules without LLM assistance. Agents that only delegate to subagents may
        not need to implement this method.

        Args:
            tape (TapeType): The tape containing the agent's state and history

        Returns:
            Prompt: A prompt object for the next agent iteration, potentially empty

        Note:
            - Empty prompts signal rule-based generation without LLM
            - Method may be optional for pure delegation agents
        """

        return self.select_node(tape).make_prompt(self, tape)

    def generate_steps(self, tape: TapeType, llm_stream: LLMStream) -> Generator[Step | PartialStep, None, None]:
        """
        Generate steps from the agent by selecting a node and processing its output.

        Args:
            tape (TapeType): The input tape containing the interaction history
            llm_stream (LLMStream): Stream interface for the language model output

        Yields:
            Union[Step, PartialStep]: Union[Step, PartialStep]: The generated steps or partial steps.
        """
        # Generate new steps and other events by feeding the prompt to the LLM
        node = self.select_node(tape)
        for step in node.generate_steps(self, tape, llm_stream):
            if isinstance(step, AgentStep):
                step.metadata.node = node.name
                if node.llm:
                    step.metadata.llm = node.llm
            yield step

    def make_llm_output(self, tape: TapeType, index: int) -> LLMOutput:
        """
        Generates an LLM output based on a tape and step index.

        Args:
            tape (TapeType): The input tape
            index (int): The position in the tape up to which to process.

        Returns:
            LLMOutput: The generated language model output for the tape segment.

        Note:
            This method delegates the actual output generation to the selected node's
            make_llm_output method after selecting the appropriate node based on the
            tape segment up to the given index.
        """
        return self.select_node(tape[:index]).make_llm_output(self, tape, index)

    def delegate(self, tape: TapeType) -> Agent[TapeType]:
        """
        Delegates control to the appropriate subagent based on the current tape state.

        This method recursively traverses the agent hierarchy to find the most specific
        subagent that should handle the current tape state based on views computed from
        the tape.

        Args:
            tape (TapeType): The tape containing the current state to process.

        Returns:
            Agent[TapeType]: The subagent that should handle the current tape state.
        """
        views = self.compute_view(tape)
        subagent = self
        for view in views.stack[1:]:
            subagent = subagent.find_subagent(view.agent_name)
        logger.debug(f"{self.full_name}: Delegating to subagent: {subagent.full_name}")
        return subagent

    def is_agent_step(self, step: Step) -> bool:
        """
        Check if a step was produced by the agent.

        Args:
            step (Step): The step object to check.

        Returns:
            bool: True if the step is an Action or Thought (agent-produced),
                  False otherwise.
        """
        return isinstance(step, (Action, Thought))

    def should_stop(self, tape: TapeType) -> bool:
        """
        Check if the agent should stop its turn and wait for observations.

        Args:
            tape (TapeType): The tape containing the sequence of steps (actions and observations).

        Returns:
            bool: True if the last step in the tape is an Action, indicating the agent should stop and wait for observations.
                 False if the last step is not an Action, indicating the agent can continue.
        """
        return isinstance(tape.steps[-1], Action)

    def run_iteration(
        self, tape: TapeType, llm_stream: LLMStream | None = None
    ) -> Generator[Step | PartialStep, None, None]:
        """
        Run one iteration of the agent (assuming one call to the underlyng model).

        During an iteration the agent generates steps from a stream of tokens that arises
        from a single LLM call with a single prompt. An agent can do multiple iterations
        before returning the next action (see `run` method).

        This function can also take a given `llm_stream`, which can be useful when the agent
        reuses a tape.

        Args:
            tape (TapeType): The tape to run the agent on
            llm_stream (LLMStream): The stream of tokens from the LLM

        Yields:
            Union[Step, PartialStep]: The generated steps or partial

        Raises:
            NotImplementedError: If the agent has multiple LLMs and no LLM stream is provided
        """
        if llm_stream is None:
            prompt = self.make_prompt(tape)
            if len(self.llms) > 1:
                llm = self.llms[self.select_node(tape).llm]
            elif len(self.llms) == 1:
                llm = self.llm
            llm_stream = llm.generate(prompt) if prompt else LLMStream(None, prompt)
        for step in self.generate_steps(tape, llm_stream):
            if isinstance(step, AgentStep):
                step.metadata.prompt_id = llm_stream.prompt.id
            yield step
        if self.store_llm_calls and (llm_call := getattr(llm_stream, "llm_call", None)):
            step.metadata.other["llm_call"] = llm_call

    def run(self, tape: TapeType, max_iterations: int | None = None) -> AgentStream[TapeType]:
        """
        Run the agent on the tape iteratively, delegating to subagents until a stop condition is met.

        This method executes the agent's logic by:
        1. Delegating to appropriate subagents based on the tape state
        2. Processing steps from subagent iterations
        3. Updating the tape with new steps
        4. Checking stop conditions
        5. Tracking metadata about the execution

        Args:
            tape (TapeType): The input tape to process
            max_iterations (int, optional): Maximum number of iterations to run.
                If None, uses self.max_iterations. Defaults to None.

        Returns:
            AgentStream[TapeType]: A stream of AgentEvents containing:

                - partial_step: Intermediate processing steps
                - step: Completed agent steps with updated tape
                - final_tape: Final tape with updated metadata after completion

        Yields:
            AgentEvent: Events indicating the agent's progress including partial steps,
                completed steps with updated tape, and the final result.

        Raises:
            ValueError: If the agent generates anything other than steps or partial steps.
        """
        if max_iterations is None:
            max_iterations = self.max_iterations

        def _run_implementation():
            nonlocal tape
            n_iterations = 0
            input_tape_length = len(tape)
            input_tape_id = tape.metadata.id
            stop = False
            original_metadata = tape.metadata
            while n_iterations < max_iterations and not stop:
                current_subagent = self.delegate(tape)
                for step in current_subagent.run_iteration(tape):
                    if isinstance(step, PartialStep):
                        yield AgentEvent(partial_step=step)
                    elif isinstance(step, AgentStep):
                        step.metadata.agent = current_subagent.full_name
                        tape = tape.append(step)
                        yield AgentEvent(step=step, partial_tape=tape)
                        if self.should_stop(tape):
                            stop = True
                    else:
                        raise ValueError("Agent can only generate steps or partial steps")
                n_iterations += 1
            updated_metadata = original_metadata.model_validate(
                dict(
                    parent_id=input_tape_id,
                    author=self.name,
                    n_added_steps=len(tape) - input_tape_length,
                )
            )
            final_tape = tape.model_copy(update=dict(metadata=updated_metadata))
            yield AgentEvent(final_tape=final_tape)

        return AgentStream(_run_implementation())

    def run_batch(self: Agent[TapeType], tapes: list[TapeType]) -> list[Tape]:
        """Run agent in parallel on tapes using batched LLM calls.

        This is faster than running agents in thread and having the LLM server batch the calls.

        """
        if len(self.llms) > 1:
            raise NotImplementedError("For run_agent_batch the agent must have only one LLM for now")
        if not isinstance(self.llm, TrainableLLM):
            raise NotImplementedError("For run_agent_batch the LLM must be TrainableLLM")
        original_tapes = list(tapes)
        n_iterations = 0
        active_indices = set(range(len(tapes)))
        while n_iterations < self.max_iterations:
            prompts = []
            current_subagents = [self.delegate(tapes[i]) for i in active_indices]
            prompts = [subagent.make_prompt(tape) for subagent, tape in zip(current_subagents, tapes)]
            llm_calls = self.llm.batch_generate(prompts)
            for i in active_indices:
                # Run the equivalent of agent.run_iteration
                llm_stream = LLMStream(
                    (LLMEvent(output=output) for output in (llm_calls[i].output,)), llm_calls[i].prompt
                )
                for step in self.generate_steps(tapes[i], llm_stream):
                    step.metadata.agent = current_subagents[i].full_name
                    if isinstance(step, AgentStep):
                        step.metadata.prompt_id = llm_calls[i].prompt.id
                    tapes[i] = tapes[i].append(step)
                    if self.should_stop(tapes[i]):
                        active_indices.remove(i)
                if self.store_llm_calls:
                    step.metadata.other["llm_call"] = llm_calls[i]
            n_iterations += 1
        for i in range(len(tapes)):
            updated_metadata = original_tapes[i].metadata.model_validate(
                dict(
                    parent_id=original_tapes[i].metadata.id,
                    author=self.name,
                    n_added_steps=len(tapes[i]) - len(original_tapes[i]),
                )
            )
            tapes[i] = tapes[i].model_copy(update=dict(metadata=updated_metadata))
        return tapes

    def reuse(self, tape: TapeType) -> tuple[TapeType, list[LLMCall]]:
        """
        Reuse another agent's tape as one's own.

        Construct LLM outputs at each step where a prompt is made. Check that output
        parsing yield the same steps as in the original tape. Rewrite metadata for all steps.

        Args:
            tape (TapeType): The tape to reuse

        Returns:
            tuple[TapeType, list[LLMCall]]: The reused tape and a list of LLM calls made during the reuse

        Raises:
            TapeReuseFailure: If the regenerated steps don't match the original tape.
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
                if not prompt:
                    reused_steps.append(step)
                    i += 1
                    continue
                output = current_agent.make_llm_output(tape, i)
                llm_call = LLMCall(prompt=prompt, output=output, cached=True)
                observe_llm_call(llm_call)

                # Validate that the reconstructed llm call leads to the same steps as in the given tape
                def _generator():
                    yield LLMEvent(output=output)

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

    def get_node_runs(self, tape: TapeType) -> list[tuple[Node, int]]:
        """
        Parse the tape and identify the indices where each node began its execution.

        This method identifies transition points in the tape where different nodes started
        producing output by tracking changes in prompt IDs.

        Args:
            tape (TapeType): The sequence of tape steps to analyze.

        Returns:
            list[tuple[Node, int]]: List of tuples containing (node, index) pairs where:

                - node: The Node object that produced the tape fragment
                - index: The starting index in the tape where this node began execution
        """
        last_prompt_id = None
        result = []
        for index, step in enumerate(tape):
            if (prompt_id := step.metadata.prompt_id) and prompt_id != last_prompt_id:
                node = self.find_node(step.metadata.node)
                result.append((node, index))
            last_prompt_id = prompt_id
        return result

    def make_training_text(self, llm_call: LLMCall) -> TrainingText:
        """
        Routes the request to make training text to the agent's LLM.

        Args:
            llm_call (LLMCall): Object containing prompt and output from an LLM call.

        Returns:
            TrainingText: The training text generated from the prompt and output.

        Note:
            Currently only supports one LLM. Future versions will support multiple LLMs.
        """
        # TODO: support more than 1 LLM
        return self.llm.make_training_text(llm_call.prompt, llm_call.output)

    def make_training_data(self, tape: TapeType) -> list[TrainingText]:
        """
        Generates training data from a tape by converting LLM calls into training texts.

        Args:
            tape (TapeType): A tape containing recorded LLM interactions.

        Returns:
            list[TrainingText]: A list of training text objects created from the LLM calls.

        Notes:
            This method first reuses the tape to extract LLM calls, then converts each call
            into a training text format using make_training_text().
        """
        _, llm_calls = self.reuse(tape)
        return [self.make_training_text(llm_call) for llm_call in llm_calls]


class TapeReuseFailure(ValueError):
    """Exception raised when tape reuse operation fails.

    This exception is raised when an attempt to reuse a tape encounters an error,
    providing access to the succesfully reused part of the tape

    Args:
        msg (str): Description of why the tape reuse failed
        partial_tape (Tape): The incomplete/partial tape that was being constructed

    Attributes:
        partial_tape (Tape): The incomplete tape at the point of failure
    """

    def __init__(self, msg: str, partial_tape: Tape):
        self.partial_tape = partial_tape
        super().__init__(msg)


def _is_step_data_equal(step1: Step, step2: Step) -> bool:
    """Compare steps ignoring the metadata.

    Some LLM API messages include metadata, like unique tool call ids, that
    we must store and send back to the API. This function excludes this metadata from the comparison.

    Another caveat is that some data, like function call arguments in LLM messages, is not deserialized,
    and hence can be tricky to compare across steps. This function deserializes known fields like that before comparison..

    Args:
        step1 (Step): The first step to compare
        step2 (Step): The second step to compare

    Returns:
        bool: True if the data in the steps is equal, False otherwise
    """

    def just_data(step: Step) -> dict:
        if isinstance(step, LLMOutputParsingFailureAction):
            return {}

        data = step.llm_dict()
        for tc in data.get("tool_calls", []):
            tc.pop("id", None)
            if function := tc.get("function"):
                function["arguments"] = json.loads(function["arguments"])
        data.pop("tool_call_id", None)
        return data

    return just_data(step1) == just_data(step2)


class Annotator(Agent[AnnotatorTapeType], Generic[TapeType, AnnotatorTapeType]):
    """
    Annotator is the base class for agents that produce annotations for the tape of another agent.
    It annotates the tape by converting it into its own tape and then producing an annotation step appended to the converted tape.
    """

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
            parent_id=tape.metadata.id, author_tape_id=own_tape.metadata.id, author=self.name, n_added_steps=1
        )
        return tape.append(last_step.new_observation).model_copy(update=dict(metadata=metadata))

    def can_continue(self, _: TapeType) -> bool:
        return True

    def continue_tape(self, tape: TapeType) -> TapeType:
        own_tape = self.make_own_tape(tape)
        for event in self.run(own_tape):
            if event.final_tape:
                return self.add_observation(tape, event.final_tape)
        raise ValueError("Observation maker didn't produce final tape")
