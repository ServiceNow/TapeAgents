from pathlib import Path
from typing import Any
from pydantic import BaseModel

from tapeagents.core import Tape
from tapeagents.dialog_tape import DialogTape, UserStep
from tapeagents.llm_function_template import LLM_FUNCTION_TEMPLATE
from tapeagents.utils import diff_strings


class InputStep(BaseModel):
    name: str
    prefix: str = ""
    desc: str = ""
    # TODO: support more ways of extracting inputs from tapes, e.g. by the node name and the step kind
    offset: int
    
    def get_prefix(self):
        return self.prefix or f"{self.name.title()}:"
    
    def render(self, tape: Tape):
        return tape.steps[self.offset].content
    
class OutputStep(BaseModel):
    name: str = ""
    prefix: str = ""
    desc: str    
    
    def get_prefix(self):
        return self.prefix or f"{self.name.title()}:"


class LLMFunctionTemplate(BaseModel):
    desc: str
    inputs: list[InputStep]
    outputs: list[OutputStep]
    demos: list[Any] = []
    
    def render(self, tape: Tape):
        lines = []
        for input in self.inputs:
            lines.append(f"{input.get_prefix()} ${{{input.name}}}")
        for output in self.outputs:
            lines.append(f"{output.get_prefix()} {output.desc}")
        format_desc = "\n".join(lines)
        
        lines = []
        for input in self.inputs:
            lines.append(f"{input.get_prefix()} {input.render(tape)}")
        lines.append(self.outputs[0].get_prefix())
        input_prompt = "\n".join(lines)
        
        return LLM_FUNCTION_TEMPLATE.format(
            function_desc=self.desc, format_desc=format_desc, input_prompt=input_prompt
        )
    
res_path = Path(__file__).parent.resolve() / ".." / "tests" / "res"
    
def test_dspy_qa_prompt():
    func = LLMFunctionTemplate(
        desc="Answer questions with short factoid answers.",
        inputs=[
            InputStep(name="question", offset=-1)
        ],
        outputs=[
            OutputStep(name="answer", desc="often between 1 and 5 words")
        ]
    )    
    
    start_tape = DialogTape(steps=[
        UserStep(content="What is the nationality of the chef and restaurateur featured in Restaurant: Impossible?"),
    ])
    render = func.render(start_tape)
    with open(res_path / "llm_function" / "qa.txt", "r") as f:
        gold = f.read()    
    if render != gold:
        print(diff_strings(render, gold))
        assert False
        
        
def test_dspy_cot_prompt():
    func = LLMFunctionTemplate(
        desc="Answer questions with short factoid answers.",
        inputs=[
            InputStep(name="question", offset=-1)
        ],
        outputs=[
            OutputStep(prefix="Reasoning: Let's think step by step in order to", desc="${produce the answer}. We ..."),
            OutputStep(name="answer", desc="often between 1 and 5 words")
        ]
    )    
    
    start_tape = DialogTape(steps=[
        UserStep(content="What is the nationality of the chef and restaurateur featured in Restaurant: Impossible?"),
    ])
    render = func.render(start_tape)
    with open(res_path / "llm_function" / "cot.txt", "r") as f:
        gold = f.read()    
    if render != gold:
        print(diff_strings(render, gold))
        assert False    
    
    
if __name__ == "__main__":
    test_dspy_qa_prompt()
    test_dspy_cot_prompt()