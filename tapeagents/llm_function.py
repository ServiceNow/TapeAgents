"""
Optimizable llm functions.
"""

from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel

from tapeagents.agent import Agent, Node
from tapeagents.core import Prompt, Step, Tape
from tapeagents.dialog_tape import (
    AssistantStep,
    AssistantThought,
    FunctionCall,
    ToolCall,
    ToolCalls,
)
from tapeagents.llms import LLMStream

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


class RationaleOutput(ThoughtOutput):
    @classmethod
    def for_output(cls, output_name: str):
        return cls(
            name="rationale",
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

    def generate_steps(self, agent, tape: Tape, llm_stream: LLMStream):
        # TODO: streaming
        # TODO: more robust parsing that doesn't rely on ':' and '\n'
        output_text = llm_stream.get_text()
        output_lines = output_text.split("\n")[-len(self.outputs) :]
        output_values = [output_lines[0]] + [line.split(":")[1].strip() for line in output_lines[1:]]
        for i, output in enumerate(self.outputs):
            if len(self.outputs) > len(output_values):
                yield output.parse("LLM SKIPPED OUTPUT")
                output_values.insert(i, "LLM SKIPPED OUTPUT")
            else:
                yield output.parse(output_values[i])


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
