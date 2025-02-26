"""
Optimizable llm functions.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Generator, Type

from pydantic import BaseModel

from tapeagents.agent import Agent, Node
from tapeagents.core import Prompt, Step, Tape
from tapeagents.dialog_tape import (
    AssistantStep,
    AssistantThought,
)
from tapeagents.llms import LLMStream
from tapeagents.tool_calling import FunctionCall, ToolCall, ToolCalls

logger = logging.getLogger(__name__)

LLM_FUNCTION_TEMPLATE = """{function_desc}
{partial_demos}
---

Follow the following format.

{format_desc}
{demos}
---

{input_prompt}"""


class Variable(BaseModel):
    """Base class for all LLM function inputs and outputs"""

    name: str
    prefix: str = ""
    desc: str = ""
    separator: str = " "

    def get_prefix(self):
        return self.prefix or f"{self.name.title()}:"

    def get_desc(self):
        return self.desc or f"${{{self.name}}}"

    def render(self, value: Step):
        return value.content  # type: ignore


class Input(Variable):
    """Describes an input to an LLM-based function."""

    pass


class Output(Variable):
    """Describes an output of an LLM-based function."""

    def parse(self, text: str) -> Step:
        raise NotImplementedError()


class AssistantOutput(Output):
    """Describes an output of an LLM-based function."""

    def parse(self, text: str) -> Step:
        return AssistantStep(content=text)


class ThoughtOutput(Output):
    def parse(self, text: str):
        return AssistantThought(content=text)


class ToolCallOutput(Output):
    tool_name: str
    arg_name: str

    def parse(self, text: str):
        tc = ToolCall(function=FunctionCall(name=self.tool_name, arguments={self.arg_name: text}))
        return ToolCalls(tool_calls=[tc])

    def render(self, value: ToolCalls):
        return value.tool_calls[0].function.arguments[self.arg_name]


class ReasoningOutput(ThoughtOutput):
    @classmethod
    def for_output(cls, output_name: str):
        return cls(
            name="reasoning",
            prefix="Reasoning: Let's think step by step in order to",
            desc=f"""${{produce the {output_name}}}. We ...""",
        )


class LLMFunctionTemplate(BaseModel):
    desc: str
    inputs: list[Input]
    outputs: list[Output]
    partial_demos: list[dict] = []
    demos: list[dict] = []

    def make_prompt(self, input_values: list) -> Prompt:
        def render_demo(demo: dict) -> str:
            lines = []
            for input in self.inputs:
                if input.name in demo:
                    lines.append(f"{input.get_prefix()}{input.separator}{input.render(demo[input.name])}")
            for output in self.outputs:
                if output.name in demo:
                    lines.append(f"{output.get_prefix()}{output.separator}{output.render(demo[output.name])}")
            return "\n".join(lines)

        # PARTIAL DEMOS
        blocks = [render_demo(demo) for demo in self.partial_demos]
        partial_demos = "\n---\n\n" + "\n\n".join(blocks) + "\n" if blocks else ""

        # FORMAT DESCRIPTION
        lines = []
        for input in self.inputs:
            lines.append(f"{input.get_prefix()} {input.get_desc()}")
        for output in self.outputs:
            lines.append(f"{output.get_prefix()} {output.get_desc()}")
        format_desc = "\n".join(lines)

        # DEMOS
        blocks = [render_demo(demo) for demo in self.demos]
        demos = "\n---\n\n" + "\n\n---\n\n".join(blocks) + "\n" if blocks else ""

        # INPUT PROMPT
        lines = []
        for input, value in zip(self.inputs, input_values):
            lines.append(f"{input.get_prefix()}{input.separator}{input.render(value)}")
        lines.append(self.outputs[0].get_prefix())
        input_prompt = "\n".join(lines)

        text = LLM_FUNCTION_TEMPLATE.format(
            function_desc=self.desc,
            partial_demos=partial_demos,
            format_desc=format_desc,
            demos=demos,
            input_prompt=input_prompt,
        )
        return Prompt.from_user_message(text)

    def generate_steps(self, agent: Agent, tape: Tape, llm_stream: LLMStream) -> Generator[Step]:
        # TODO: streaming
        output_values = []
        output_text = llm_stream.get_text()
        for i, output in enumerate(self.outputs):
            if not isinstance(output, (ThoughtOutput, ToolCallOutput, AssistantOutput)):
                raise NotImplementedError(f"Output type {output} not implemented")
            # Find the variable output.name in the llm output
            values = re.split(f"{output.name}:", output_text, flags=re.IGNORECASE)
            if len(values) == 2:
                # Variable output.name found in llm output
                value = values[1].split("\n")[0].strip()  # heuristic for the end of the argument
                if isinstance(output, ReasoningOutput):
                    # if present, remove the prefix from the reasoning output
                    values = re.split(output.prefix, output_text, flags=re.IGNORECASE)
                    if len(values) == 2:
                        value = values[1].split("\n")[0].strip()  # heuristic for the end of the argument
                    else:
                        logger.debug(f"No prefix to remove from reasoning output key {output.name}")
            elif len(values) == 1:
                # Variable output.name not found in llm output
                # Set value to empty string to keep positional order
                value = ""
                if i == 0:
                    # For the first output, llm output might not repeat prefix
                    values = re.split(r"^[A-z]+:", output_text, flags=re.IGNORECASE)
                    if len(values) == 1:
                        # llm output doesn't start with another variable output
                        # we assume the first section of the llm output is the value
                        value = output_text.split("\n")[0].strip()  # heuristic for the end of the output
                        logger.debug(
                            f"Assuming value for the output key '{output.name}' is the first llm output section: '{value}'"
                        )
                    else:
                        # llm output started with another variable output
                        # Not throwing an error here because the first (reasoning) output might not be present be the second (query/answer) output is
                        logger.error(
                            f"Could not assume value for the output key '{output.name}' from the first llm output section: '{values[0]}'"
                        )
                else:
                    raise ValueError(f"Could not find output key '{output.name}' in output_text: '{output_text}'")
            else:
                raise ValueError(
                    f"Found multiple instances of output key `{output.name}` in output_text: {output_text}"
                )
            output_values.append(value)

        if len(output_values) != len(self.outputs):
            logger.warning(f"Could not find all outputs in output_text: {output_text}\nOutput found: {output_values}")

        for output, value in zip(self.outputs, output_values):
            yield output.parse(value)


class KindRef(BaseModel):
    """Refer to the input by the step kind. Refers the last step with the given kind."""

    kind: str


def by_step(step_class: Type) -> KindRef:
    return KindRef(kind=step_class.get_kind())


class NodeRef(BaseModel):
    name: str


def by_node(node: Node) -> NodeRef:
    return NodeRef(name=node.name)


class LLMFunctionNode(Node):
    template_name: str
    input_refs: list[int | NodeRef | KindRef | Step] = []

    def get_function(self, agent):
        template = agent.templates[self.template_name]
        if not isinstance(template, LLMFunctionTemplate):
            raise ValueError(f"Template {self.template_name} is not an LLMFunctionTemplate")
        return template

    def extract_inputs(self, tape: Tape, index: int | None = None):
        if index is None:
            index = len(tape) + 1

        inputs = []
        for ref in self.input_refs:
            resolved = False
            steps = tape.steps[:index]
            if isinstance(ref, int):
                inputs.append(steps[ref])
            elif isinstance(ref, NodeRef):
                for step in reversed(steps):
                    if step.metadata.node == ref.name:
                        inputs.append(step)
                        resolved = True
                        break
                if not resolved:
                    raise ValueError(f"Node {ref.name} not found in tape")
            elif isinstance(ref, KindRef):
                for step in reversed(steps):
                    if step.kind == ref.kind:
                        inputs.append(step)
                        resolved = True
                        break
                if not resolved:
                    raise ValueError(f"Step with kind {ref.kind} not found in tape")
            elif isinstance(ref, Step):
                inputs.append(ref)
            else:
                raise ValueError(f"Invalid input reference {ref}")
        return inputs

    def make_prompt(self, agent, tape: Tape):
        return self.get_function(agent).make_prompt(self.extract_inputs(tape))

    def generate_steps(self, agent: Any, tape: Tape, llm_stream: LLMStream):
        yield from self.get_function(agent).generate_steps(agent, tape, llm_stream)

    def extract_demo(self, agent: Agent, tape: Tape, index: int):
        func = self.get_function(agent)
        input_values = self.extract_inputs(tape, index)
        output_values = tape.steps[index : index + len(func.outputs)]
        demo = {}
        for input, value in zip(func.inputs, input_values):
            demo[input.name] = value
        for output, value in zip(func.outputs, output_values):
            demo[output.name] = value
        return demo
