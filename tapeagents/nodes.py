"""
Nodes are the building blocks of a TapeAgent, representing atomic units of the agent's behavior.
"""

import json
import logging
import re
from datetime import datetime
from typing import Annotated, Any, Callable, Generator, Type, Union

from litellm import ChatCompletionMessageToolCall
from pydantic import Field, TypeAdapter, ValidationError
from termcolor import colored

from tapeagents.agent import Agent, Node
from tapeagents.core import (
    AgentStep,
    ControlFlow,
    LLMOutputParsingFailureAction,
    Observation,
    PartialStep,
    Prompt,
    SetNextNode,
    Step,
    StopStep,
    Tape,
)
from tapeagents.environment import CodeBlock
from tapeagents.llms import LLMOutput, LLMStream
from tapeagents.steps import REASON_TO_USE_KEY, BranchStep, ReasoningThought
from tapeagents.tool_calling import FunctionCall, ToolCallAction, ToolSpec, as_openai_tool
from tapeagents.tools.code_executor import PythonCodeAction
from tapeagents.utils import FatalError, class_for_name, response_format, sanitize_json_completion, step_schema_json
from tapeagents.view import Call, Respond, TapeViewStack

logger = logging.getLogger(__name__)


class StandardNode(Node):
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
        steps (Any): Class used for step (or steps) validation, excluded from model
        next_node (str): Identifier for the next node in sequence

    Example:
        ```python
        node = StandardNode(
            guidance="Please respond with next action",
            system_prompt="You are a helpful assistant",
            steps_prompt="Available steps: think, act, finish",
            steps=AgentStep
        )
        ```
    """

    guidance: str = ""  # guidance text that is attached to the end of the prompt
    system_prompt: str = ""
    steps_prompt: str = ""  # use {allowed_steps} to insert steps schema
    steps: type[Step] | list[type[Step] | str] | str = Field(exclude=True, default_factory=list)
    use_known_actions: bool = False
    next_node: str = ""
    trim_obs_except_last_n: int = 2
    use_function_calls: bool = False
    allow_code_blocks: bool = False
    structured_output: bool = False
    _steps_type: Any = None
    _steps: list[Type[Step] | ToolSpec] = None  # type: ignore

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)

    def prepare_step_types(self, agent: Agent) -> list[Type[Step] | ToolSpec]:
        step_classes_or_str = self.steps if isinstance(self.steps, list) else [self.steps]
        steps = [a for a in agent.known_actions if not isinstance(a, ToolSpec)] if self.use_known_actions else []
        steps += [class_for_name(step) if isinstance(step, str) else step for step in step_classes_or_str]
        if self.allow_code_blocks:
            steps.remove(PythonCodeAction)

        if self.structured_output:
            assert len(steps) == 1, "Structured output requires exactly one output step class"

        if steps and not self.use_function_calls:
            self._steps_type = Annotated[Union[tuple(steps)], Field(discriminator="kind")]
        if self.use_known_actions:
            steps += [a for a in agent.known_actions if isinstance(a, ToolSpec)]
        return steps

    def make_prompt(self, agent: Agent, tape: Tape) -> Prompt:
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
        self._steps = self.prepare_step_types(agent)
        steps = self.get_steps(tape, agent)
        steps_description = self.get_steps_description(agent)
        # cleaned_tape = self.prepare_tape(tape)
        # messages = self.tape_to_messages(cleaned_tape, steps_description)
        messages = self.steps_to_messages(steps, steps_description)
        if agent.llms[self.llm].count_tokens(messages) > (agent.llms[self.llm].context_size - 500):
            old_trim = self.trim_obs_except_last_n
            self.trim_obs_except_last_n = 1
            messages = self.steps_to_messages(steps, steps_description)
            self.trim_obs_except_last_n = old_trim
        format = response_format(self._steps[0]) if self.structured_output else None
        tools = (
            [as_openai_tool(s).model_dump() for s in self._steps] if self.use_function_calls and not format else None
        )
        prompt = Prompt(messages=messages, tools=tools, response_format=format)
        logger.debug(colored(f"PROMPT tools:\n{prompt.tools}", "red"))
        for i, m in enumerate(prompt.messages):
            logger.debug(colored(f"PROMPT M{i + 1}, {m['role']}:\n{m['content']}", "red"))
        return prompt

    def get_steps(self, tape: Tape, agent: Agent) -> list[Step]:
        """
        Prepares tape by filtering out control flow steps.

        This method creates a new tape instance with only non-control flow steps,
        specifically excluding SetNextNode instances.

        Args:
            tape (Tape): The input tape containing a sequence of steps.

        Returns:
            Tape: A new tape instance containing only non-control flow steps.
        """
        steps = agent.compute_view(tape).top.steps
        steps_without_control_flow = [step for step in steps if not isinstance(step, ControlFlow)]
        return steps_without_control_flow

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
            if not isinstance(tape.steps[i], SetNextNode):
                steps.append(tape.steps[i])
            i += 1

        # if there is only one step, return it as a single dict, not a list
        content = [step.llm_dict() for step in steps] if len(steps) > 1 else steps[0].llm_dict()
        return LLMOutput(role="assistant", content=json.dumps(content, indent=2, ensure_ascii=False))

    def steps_to_messages(self, steps: list[Step], steps_description: str) -> list[dict]:
        """
        Converts a Tape object and steps description into a list of messages for LLM conversation.

        Args:
            tape (Tape): A Tape object containing conversation steps.
            steps_description (str): A description of the conversation steps.

        Returns:
            list[dict]: A list of dictionaries representing the conversation messages.
                       Each dictionary contains 'role' and 'content' keys.
                       Roles can be 'system', 'user', or 'assistant'.
                       The system prompt is always the first message.
                       If steps_description is provided, it's added as a user message.
                       Messages from tape are added with roles based on step type.
                       If guidance exists, it's added as the final user message.
        """
        messages: list[dict] = []
        if self.system_prompt:
            system_prompt = self.system_prompt.format(date=datetime.now().strftime("%Y-%m-%d"))
            messages.append({"role": "system", "content": system_prompt})
        if steps_description:
            messages.append({"role": "user", "content": steps_description})
        shorts = 0
        shorts_chars = 0
        longs = 0
        longs_chars = 0
        n_observations = len([step for step in steps if isinstance(step, Observation)])
        n_short = n_observations - self.trim_obs_except_last_n
        obs_number = 0
        for i, step in enumerate(steps):
            role = "assistant" if isinstance(step, AgentStep) else "user"
            if isinstance(step, Observation):
                if obs_number < n_short:
                    view = step.short_view()
                    shorts += 1
                    shorts_chars += len(view)
                else:
                    view = step.llm_view()
                    longs += 1
                    longs_chars += len(view)
                obs_number += 1
            else:
                view = step.llm_view()
            messages.append({"role": role, "content": view})
        if self.guidance:
            messages.append({"role": "user", "content": self.guidance})
        logger.debug(
            f"Rendered short observations: {shorts} ({shorts_chars} chars), long observations: {longs} ({longs_chars} chars)"
        )
        return messages

    def get_steps_description(self, agent: Agent) -> str:
        """
        Get the steps description for the agent's task.

        This method returns the predefined steps prompt which describes the sequence of actions
        or steps that the agent should follow.

        Args:
            tape (Tape): The tape object containing the context and state information.
            agent (Agent): The agent object that will execute the steps.

        Returns:
            str: The steps prompt describing the sequence of actions.
        """
        allowed_steps = ""
        if self._steps_type and not self.use_function_calls:
            allowed_steps = agent.llms[self.llm].get_step_schema(self._steps_type)
        return self.steps_prompt.format(allowed_steps=allowed_steps, tools_description=agent.tools_description)

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
        for event in llm_stream:
            if event.chunk:
                yield PartialStep(step=ReasoningThought(reasoning=event.chunk))
            if not event.output:
                continue
            if event.output.content:
                logger.debug(colored(f"LLM output:\n{event.output.content}", "cyan"))
                new_steps += list(self.parse_completion(event.output.content))
            if event.output.tool_calls:
                logger.debug(colored(f"LLM tool calls:\n{event.output.tool_calls}", "cyan"))
                new_steps += [self.tool_call_to_step(agent, tool_call) for tool_call in event.output.tool_calls]
            for i, step in enumerate(new_steps):
                yield self.postprocess_step(tape, new_steps[:i], step)
                if isinstance(step, LLMOutputParsingFailureAction):
                    yield SetNextNode(next_node=self.name)  # loop to the same node to retry
                    break
            if not new_steps:
                logger.warning(f"Empty llm output?\n{event.output}")
        if not new_steps:
            raise FatalError("No completions!")
        if (
            self.next_node
            and not isinstance(new_steps[-1], StopStep)
            and not any(isinstance(step, SetNextNode) for step in new_steps)
        ):
            yield SetNextNode(next_node=self.next_node)

    def tool_call_to_step(self, agent: Agent, tool_call: ChatCompletionMessageToolCall) -> Step:
        name = tool_call.function.name or "None"
        tool_to_cls = {
            as_openai_tool(s).function.name: (ToolCallAction if isinstance(s, ToolSpec) else s) for s in self._steps
        }
        step_cls = tool_to_cls.get(name)
        if step_cls is None:
            step = LLMOutputParsingFailureAction(
                error=f"Unknown tool call: {name}", llm_output=tool_call.model_dump_json(indent=2)
            )
        elif step_cls == ToolCallAction:
            try:
                args = json.loads(tool_call.function.arguments)
                step = ToolCallAction(
                    id=tool_call.id,
                    function=FunctionCall(name=name, arguments=args),
                    reason_to_use=args[REASON_TO_USE_KEY] if REASON_TO_USE_KEY in args else "",
                )
            except json.JSONDecodeError:
                step = LLMOutputParsingFailureAction(
                    error=f"Failed to parse tool call arguments: {tool_call.function.arguments}",
                    llm_output=tool_call.model_dump_json(indent=2),
                )
        else:
            args = tool_call.function.arguments
            step = step_cls.model_validate_json(args) if args else step_cls()
        return step

    def postprocess_step(self, tape: Tape, new_steps: list[Step], step: Step) -> Step:
        """
        Post-processes a step after its generation.

        By default returns the step unchanged.

        Args:
            tape (Tape): The tape
            new_steps (list[Step]): List of new steps that were generated during the current iteration
            step (Step): The step that was just generated

        Returns:
            Step: The processed step, by default returns the original step unmodified
        """
        return step

    def parse_completion(self, llm_output: str) -> Generator[Step, None, None]:
        """Parse LLM completion output into a sequence of agent steps.

        This method processes the LLM output string by parsing it as JSON and validating it against
        the agent step class schema. It handles both single step and multi-step outputs.

        Args:
            llm_output (str): The raw output string from the LLM to be parsed

        Yields:
            Step: Individual validated agent steps
            LLMOutputParsingFailureAction: Error information if parsing or validation fails

        Note:
            All parsing errors are handled internally and yielded as
            LLMOutputParsingFailureAction objects.
        """
        try:
            step_dicts = json.loads(sanitize_json_completion(llm_output))
            if isinstance(step_dicts, dict):
                step_dicts = [step_dicts]
        except Exception as e:
            if self.allow_code_blocks and "```" in llm_output:
                logger.info("Parsing code blocks from LLM output")
                for code_block in self.extract_code_blocks(llm_output):
                    if isinstance(code_block, str):
                        yield ReasoningThought(reasoning=code_block)
                    elif code_block.language and code_block.language != "python":
                        yield LLMOutputParsingFailureAction(
                            error=f"Unsupported code block language: {code_block.language}", llm_output=llm_output
                        )
                    else:
                        yield PythonCodeAction(name="code.py", code=code_block.code, input_files=[])
            elif self._steps_type:
                logger.exception(f"Failed to parse LLM output as json: {llm_output}\n\nError: {e}")
                yield LLMOutputParsingFailureAction(
                    error=f"Failed to parse LLM output as json: {e}", llm_output=llm_output
                )
            else:
                yield ReasoningThought(reasoning=llm_output)
            return

        try:
            if len(self._steps) == 1:
                steps = [self._steps[0].model_validate(step_dict) for step_dict in step_dicts]
            else:
                steps = [TypeAdapter(self._steps_type).validate_python(step_dict) for step_dict in step_dicts]
        except ValidationError as e:
            err_text = ""
            for err in e.errors():
                loc = ".".join([str(loc) for loc in err["loc"]])
                err_text += f"{loc}: {err['msg']}\n"
            logger.exception(f"Failed to validate LLM output: {step_dicts}\n\nErrors:\n{err_text}")
            yield LLMOutputParsingFailureAction(
                error=f"Failed to validate LLM output: {err_text}", llm_output=llm_output
            )
            return
        except Exception as e:
            logger.exception(f"Failed to parse LLM output dict: {step_dicts}\n\nError: {e}")
            yield LLMOutputParsingFailureAction(error=f"Failed to parse LLM output dict: {e}", llm_output=llm_output)
            return
        for step in steps:
            yield step

    def extract_code_blocks(self, text: str) -> list[CodeBlock | str]:
        """Extract code blocks and plain text from a string."""
        results = []
        pattern = r"```(.*?)\n(.*?)```"
        last_end = 0
        for match in re.finditer(pattern, text, re.DOTALL):
            # Add text before this code block
            start = match.start()
            if start > last_end:
                results.append(text[last_end:start])

            # Extract language and code
            language = match.group(1).strip()
            code = match.group(2)
            results.append(CodeBlock(code=code, language=language))

            last_end = match.end()

        # Add remaining text
        if last_end < len(text):
            results.append(text[last_end:])

        return [r for r in results if r]  # Filter out empty strings

    def trim_tape(self, tape: Tape) -> Tape:
        """
        Trims the tape by removing unnecessary positions.

        Args:
            tape (Tape): The tape object to be trimmed.

        Returns:
            Tape: The trimmed tape object.

        Note:
            Currently this is a placeholder method that returns the tape unchanged.
        """
        return tape


class AsStep(StandardNode):
    format_prompt: str = """The JSON object should match the following schema:

{schema}

Do not reproduce the schema when producing the step, use it as a reference!
DO NOT OUTPUT ANYTHING BESIDES THE JSON! DO NOT PLACE ANY COMMENTS INSIDE THE JSON. It will break the system that processes the output."""
    guidance: str = "Convert the following paragraph into a structured JSON object:"

    def make_prompt(self, agent: Agent, tape: Tape) -> Prompt:
        self._steps = self.prepare_step_types(agent)
        last_reasoning_step_pos = [i for i, step in enumerate(tape.steps) if isinstance(step, ReasoningThought)][-1]
        text = tape[last_reasoning_step_pos].reasoning  # type: ignore
        errors_after = [
            step
            for step in tape.steps[last_reasoning_step_pos + 1 :]
            if isinstance(step, LLMOutputParsingFailureAction)
        ]
        step_cls = self._steps[0]
        msg = f"{self.guidance}\n\n{text}"
        messages = [{"role": "user", "content": msg}]
        if not self.structured_output:
            messages.append({"role": "user", "content": self.format_prompt.format(schema=step_schema_json(step_cls))})
        if errors_after:
            msg = f"Our previous attempt resulted in failure:\n\n{errors_after[-1]}"
            messages.append({"role": "user", "content": msg})
        return Prompt(messages=messages, response_format=response_format(step_cls) if self.structured_output else None)


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
        if next_node is None:
            yield BranchStep()
        else:
            yield SetNextNode(next_node=next_node)

    def select_node(self, tape: Tape) -> str:
        """
        Select the next node based on the provided tape.

        This method should be implemented in a subclass to define the logic for selecting the next node.

        Args:
            tape (Tape): The tape object containing the necessary information for node selection.

        Returns:
            str: The identifier of the next node.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError("Implement this method in the subclass to set the next node according to your logic")


class IfLastStep(ControlFlowNode):
    next_node: str
    step_class: type[Step]

    def select_node(self, tape: Tape) -> str:
        return self.next_node if isinstance(tape[-1], self.step_class) else None


class If(ControlFlowNode):
    predicate: Callable[[Tape], bool]
    next_node: str

    def select_node(self, tape: Tape) -> str:
        return self.next_node if self.predicate(tape) else None


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


class Return(FixedStepsNode):
    steps: list[Step] = [Respond(copy_output=True)]


class GoTo(Node):
    next_node: str

    def generate_steps(
        self, agent: Any, tape: Tape, llm_stream: LLMStream
    ) -> Generator[Step | PartialStep, None, None]:
        yield SetNextNode(next_node=self.next_node)


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
