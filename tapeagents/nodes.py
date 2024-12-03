"""
Nodes are the building blocks of a TapeAgent, representing atomic units of the agent's behavior.
"""

import json
import logging
from typing import Any, Callable, Generator, Type, Union, get_origin

from pydantic import Field, TypeAdapter, ValidationError

from tapeagents.dialog_tape import AssistantStep, UserStep
from tapeagents.prompting import FORMALIZE_FORMAT, FORMALIZE_GUIDANCE, FORMALIZE_INPUT, FORMALIZE_SYSTEM_PROMPT
from tapeagents.view import Call, Respond, TapeViewStack

from .agent import Agent, Node
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
from .llms import LLMStream
from .utils import FatalError, get_step_schemas_from_union_type, sanitize_json_completion

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MonoNode(Node):
    """
    A node for simple monolithic agents that handles simple prompt generation, and universal LLM output parsing.

    This node performs the following functions:

    - Renders the entire tape into a prompt, trimming if needed
    - Attaches guidance text to the end of the prompt after rendering the tape
    - Parses the LLM output into provided step classes (class provided as annotated union)

    Attributes:
        guidance (str): Guidance text attached to the end of the prompt
        system_prompt (str): System prompt used in message construction
        steps_prompt (str): Prompt describing the steps the agent can take
        agent_step_cls (Any): Class used for step validation, excluded from model
        next_node (str): Identifier for the next node in sequence

    Example:
        ```python
        node = MonoNode(
            guidance="Please respond with next action",
            system_prompt="You are a helpful assistant",
            steps_prompt="Available steps: think, act, finish",
            agent_step_cls=AgentStep
        )
        ```
    """

    system_prompt: str = ""
    steps_prompt: str = ""  # prompt that describes the steps that the agent can take
    guidance: str = ""  # guidance text that is attached to the end of the prompt
    output_cls: Any = Field(exclude=True, default=None)
    next_node: str = ""

    def make_prompt(self, agent: Any, tape: Tape) -> Prompt:
        """Create a prompt from tape interactions.

        This method constructs a prompt by processing the tape content and agent steps description
        into a format suitable for LLM consumption. It includes token count checks and tape trimming
        if needed to fit within context size limits.

        Args:
            agent (Any): The agent object containing LLM configuration.
            tape (Tape): The tape object containing interaction history.

        Returns:
            Prompt: A Prompt object containing formatted messages for LLM consumption.

        Note:
            The method performs the following steps:

            1. Cleans the tape content
            2. Gets steps description
            3. Converts tape to messages
            4. Checks token count and trims if needed
            5. Reconstructs messages if trimming occurred
        """
        clean_tape = self.prepare_tape(tape)
        messages = self.tape_to_messages(clean_tape)
        if agent.llm.count_tokens(messages) > (agent.llm.context_size - 500):
            clean_tape = self.trim_tape(clean_tape)
            messages = self.tape_to_messages(clean_tape)
        return Prompt(messages=messages)

    def prepare_tape(self, tape: Tape) -> Tape:
        """Override this method if you want to e.g. shorten content in some steps of the tape."""
        return tape

    def trim_tape(self, tape: Tape) -> Tape:
        """Override this method if you want to trim the tape to reduce the prompt size."""
        return tape

    def tape_to_steps(self, tape: Tape) -> list[Step]:
        """Override this method if you want to make prompt messages from different steps."""
        return TapeViewStack.compute(tape).top.steps

    def tape_to_messages(self, tape: Tape) -> list[dict]:
        """
        Converts a Tape object and steps description into a list of messages for LLM conversation.

        Args:
            tape (Tape): A Tape object containing conversation steps.

        Returns:
            list[dict]: A list of dictionaries representing the conversation messages.
                       Each dictionary contains 'role' and 'content' keys.
                       Roles can be 'system', 'user', or 'assistant'.
                       The system prompt is always the first message.
                       If the output_cls is set, steps description is added as a user message.
                       Messages from tape are added with roles based on step type.
                       If guidance exists, it's added as the final user message.
        """
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
                agent_own_step = isinstance(step, AgentStep) and step.metadata.agent == agent.full_name
                step_role = "assistant" if agent_own_step else "user"
                step_content = step.llm_view()
            messages.append({"role": step_role, "content": step_content})
        if self.guidance:
            messages.append({"role": "user", "content": self.guidance})
        return messages

    def get_steps_description(self, tape: Tape) -> str:
        """
        Get the steps description for the agent's task.

        This method returns the predefined steps prompt which describes
        the schemas of steps that the agent should follow.

        Args:
            tape (Tape): The tape object containing the context and state information.

        Returns:
            str: The steps prompt describing the sequence of actions.
        """
        text = self.steps_prompt
        if self.output_cls:
            schema = get_step_schemas_from_union_type(self.output_cls)
            text = self.steps_prompt.format(schema=schema)
        return text

    def generate_steps(
        self, agent: Any, tape: Tape, llm_stream: LLMStream
    ) -> Generator[Step | PartialStep, None, None]:
        """
        Generates a sequence of steps based on the LLM stream output.

        This method processes the output from a language model stream and converts it into a series of steps.
        It handles the parsing of completions and post-processing of steps.

        Args:
            agent (Any): The agent instance that will execute the steps.
            tape (Tape): The tape object containing the execution context and history.
            llm_stream (LLMStream): The stream of language model outputs to process.

        Yields:
            Union[Step, PartialStep]: Individual steps generated from the LLM stream output.

        Raises:
            FatalError: If no completions are generated from the LLM stream.

        Note:
            - If the node has a next_node defined and the final step is not a StopStep,
              it will yield a SetNextNode step to continue the execution flow.
        """
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

    def make_llm_output(self, agent: Any, tape: Tape, index: int) -> LLMOutput:
        """
        Creates an LLMOutput from a sequence of steps in the tape that share the same prompt_id.

        Args:
            agent (Any): The agent instance associated with the output.
            tape (Tape): The tape containing the sequence of steps.
            index (int): The starting index in the tape to process steps from.

        Returns:
            LLMOutput: An output object containing:

                - role: Set to "assistant"
                - content: JSON string of step data, formatted as either: a single dictionary
                if there is only one step, or a list of dictionaries

        Note:
            - Only processes steps with matching prompt_id from the starting index
            - Excludes SetNextNode steps from the output
            - JSON content is formatted with indentation
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

    def update(self, node_config: dict[str, Any]) -> Node:
        node_config["output_cls"] = self.output_cls
        return super().update(node_config)


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
    A node that controls the flow of execution by selecting the next node based on tape content.

    This abstract class provides a framework for implementing control flow logic in a node.
    It determines which node should be executed next based on the current state of the tape.

    Example:
        ```python
        class MyControlFlow(ControlFlowNode):
            def select_node(self, tape):
                if isinstance(tape[-1], SuccessObservation):
                    return 'node_a'
                return 'node_b'
        ```
    """

    next_node: str = ""
    predicate: Callable[[Tape], bool] = lambda tape: True

    def generate_steps(
        self, agent: Any, tape: Tape, llm_stream: LLMStream
    ) -> Generator[Step | PartialStep, None, None]:
        """
        Generates steps that moves the execution to the next node based on the tape content.

        Args:
            agent (Any): The agent instance executing the node
            tape (Tape): The tape object containing the context and state
            llm_stream (LLMStream): Stream for language model interaction

        Yields:
            step (SetNextNode): A step indicating which node should be executed next
        """
        next_node = self.select_node(tape)
        if next_node:
            yield SetNextNode(next_node=next_node)
        else:
            yield ConditionCheck()  # we should put a step into tape to know last executed node

    def select_node(self, tape: Tape) -> str:
        return self.next_node if self.predicate(tape) else ""


class ObservationControlNode(ControlFlowNode):
    """
    A control flow node that selects the next node based on the last observation in the tape.

    This node examines the last observation in the tape and uses it to determine which node
    to execute next based on a mapping of observation types to node names.

    Attributes:
        observation_to_node (dict[Type, str]): Mapping of observation types to destination node names
        default_node (str): Default node to jump to if no matching observation type is found

    Example:
        ```python
        node = ObservationControlNode(
            observation_to_node={
                SuccessObservation: "success_node",
                ErrorObservation: "error_node"
            },
            default_node="fallback_node"
        )
        ```
    """

    observation_to_node: dict[Type, str] = {}
    default_node: str = ""  # jump to the last node by default

    def select_node(self, tape: Tape) -> str:
        """
        Selects the next node based on the type of the last observation in the tape.

        Returns default_node if no observations exist or no matching type is found.

        Args:
            tape (Tape): The tape object containing the context and state

        Returns:
            str: The name of the next node to execute
        """
        observations = [step for step in tape.steps if isinstance(step, Observation)]
        last_observation = observations[-1] if observations else None
        return self.observation_to_node.get(type(last_observation), self.default_node)


class FixedStepsNode(Node):
    """A Node that generates a fixed sequence of predefined steps.

    This node simply yields a sequence of steps that were provided during initialization,
    without any dynamic generation or modification.

    Attributes:
        steps (list[Step]): A list of Step objects to be yielded in sequence.

    Example:
        ```python
        fixed_node = FixedStepsNode(steps=[
            AssistantStep(text="Hello"),
            SetNextNode(next_node="node_a")
        ])
        ```
    """

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
    steps: list[Respond] = [Respond(copy_output=True)]  # type: ignore


class CallSubagent(Node):
    """
    Node that calls a subagent with inputs from the current tape view.
    """

    agent: Agent
    inputs: tuple[str | int, ...] = Field(
        default_factory=tuple,
        description="Names of the subagents which outputs are required for the current subagent to run",
    )

    def model_post_init(self, __context: Any) -> None:
        self.name = f"{self.agent.name}Node"

    def generate_steps(self, _: Any, tape: Tape, llm_stream: LLMStream):
        view = TapeViewStack.compute(tape)
        yield Call(agent_name=self.agent.name)
        for input_ in self.inputs:
            yield view.top.get_output(input_).model_copy(deep=True)


class RespondIfNotRootNode(Node):
    def generate_steps(self, _: Any, tape: Tape, llm_stream: LLMStream):
        view = TapeViewStack.compute(tape)
        if len(view.stack) > 1:
            yield Respond(copy_output=True)
        return
