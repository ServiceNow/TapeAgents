from pathlib import Path
from typing import Any
from pydantic import BaseModel

from tapeagents.core import Tape
from tapeagents.dialog_tape import DialogTape, UserStep
from tapeagents.llm_function_template import LLM_FUNCTION_TEMPLATE
from tapeagents.utils import diff_strings


class FunctionInput(BaseModel):
    name: str
    desc: str = ""
    # TODO: support more ways of extracting inputs from tapes, e.g. by the node name and the step kind
    offset: int
    
    def extract(self, tape: Tape):
        return tape.steps[self.offset]
    
class FunctionOutput(BaseModel):
    name: str
    desc: str    


class LLMFunctionTemplate(BaseModel):
    desc: str
    inputs: list[FunctionInput]
    output: FunctionOutput
    demos: list[Any] = []
    
    def render(self, tape: Tape):
        lines = []
        for input in self.inputs:
            # Question: ${question}

            lines.append(f"{input.name.title()}: ${{{input.name}}}")
        
        format_desc = "\n".join(lines)
        
        
        return LLM_FUNCTION_TEMPLATE.format(
            function_desc=self.desc, format_desc=format_desc, input_prompt=""
        )
    
res_path = Path(__file__).parent.resolve() / ".." / "tests" / "res"
    
def test_dspy_qa_prompt():
    func = LLMFunctionTemplate(
        desc="Answer questions with short factoid answers.",
        inputs=[
            FunctionInput(name="question", offset=-1)
        ],
        output=FunctionOutput(name="answer", desc="often between 1 and 5 words")
    )    
    
    start_tape = DialogTape(steps=[
        UserStep(content="What is the nationality of the chef and restaurateur featured in Restaurant: Impossible?"),
    ])
    with open(res_path / "llm_function" / "qa.txt", "r") as f:
        gold = f.read()
    print(diff_strings(func.render(start_tape), gold))
    
if __name__ == "__main__":
    test_dspy_qa_prompt()