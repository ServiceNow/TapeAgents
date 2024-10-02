import json

from pathlib import Path
from typing import Any, Generator, Type, cast
from typing_extensions import Self
from pydantic import BaseModel, Field

from tapeagents.agent import Agent, Node
from tapeagents.core import PartialStep, Pass, Prompt, Step, Tape
from tapeagents.dialog_tape import AssistantStep, AssistantThought, DialogStep, DialogTape, ToolCalls, ToolResult, UserStep
from tapeagents.environment import ToolEnvironment
from tapeagents.llm_function_template import LLM_FUNCTION_TEMPLATE
from tapeagents.llms import LLMStream, LiteLLM
from tapeagents.observe import retrieve_tape_llm_calls
from tapeagents.runtime import main_loop
from tapeagents.utils import diff_strings


class InputStep(BaseModel):
    name: str
    prefix: str = ""
    desc: str = ""
    separator: str = " "
    
    def get_prefix(self):
        return self.prefix or f"{self.name.title()}:"
    
    def get_desc(self):
        return self.desc or f"${{{self.name}}}"
    
    def render(self, value: Step):
        return value.content # type: ignore
   
    
class OutputStep(BaseModel):
    name: str = ""
    prefix: str = ""
    desc: str
    separator: str = " "
    # TODO: add a step registry to make this serializable
    step_class: Type = AssistantStep
    
    def get_prefix(self):
        return self.prefix or f"{self.name.title()}:"
    
    def render(self, value: Step):
        return value.content # type: ignore
    
    def parse(self, text: str) -> Step:
        return self.step_class.model_validate({"content": text})
    

class LLMFunctionTemplate(BaseModel):
    desc: str
    inputs: list[InputStep]
    outputs: list[OutputStep]
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
            lines.append(f"{output.get_prefix()} {output.desc}")
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
            input_prompt=input_prompt
        )
        return Prompt.from_user_message(text)
    
    def generate_steps(self, agent, tape: Tape, llm_stream: LLMStream):
        # TODO: streaming
        # TODO: more robust parsing that doesn't rely on ':' and '\n'
        # TODO: configure output step class
        output_text = llm_stream.get_text()
        prompt_text = llm_stream.prompt.messages[0]["content"]
        text = prompt_text + "\n" + output_text
        output_lines = text.split("\n")[-len(self.outputs):]
        output_values = [output_lines[0]] + [line.split(":")[1].strip() for line in output_lines[1:]]
        for output, value in zip(self.outputs, output_values):
            yield output.parse(value)
        
        
class KindRef(BaseModel):
    """Refer to the input by the step kind. Refers the last step with the given kind."""
    kind: str
    
    @staticmethod
    def to(step_class: Type) -> KindRef:
        return KindRef(kind=step_class.get_kind())
    
    
class NodeRef(BaseModel):
    name: str
    
        
class LLMFunctionNode(Node):
    template_name: str
    input_refs: list[int | NodeRef | KindRef | Step] = []
    
    def get_template(self, agent):
        template = agent.templates[self.template_name]
        if not isinstance(template, LLMFunctionTemplate):
            raise ValueError(f"Template {self.template_name} is not an LLMFunctionTemplate")
        return template
    
    def make_prompt(self, agent, tape: Tape):
        inputs = []
        for ref in self.input_refs:
            resolved = False
            if isinstance(ref, int):
                inputs.append(tape.steps[ref])
            elif isinstance(ref, NodeRef):  
                for step in reversed(tape.steps):
                    if step.metadata.node == ref.name:
                        inputs.append(step)
                        resolved = True
                        break
                if not resolved:
                    raise ValueError(f"Node {ref.name} not found in tape")
            elif isinstance(ref, KindRef):
                for step in reversed(tape.steps):
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
        return self.get_template(agent).make_prompt(inputs)
        
    def generate_steps(self, agent: Any, tape: Tape, llm_stream: LLMStream):
        yield from self.get_template(agent).generate_steps(agent, tape, llm_stream)
    