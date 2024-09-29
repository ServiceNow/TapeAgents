from pathlib import Path
from typing import Any
from pydantic import BaseModel

from tapeagents.agent import Agent, Node
from tapeagents.core import Prompt, Tape
from tapeagents.dialog_tape import AssistantThought, DialogTape, UserStep
from tapeagents.llm_function_template import LLM_FUNCTION_TEMPLATE
from tapeagents.llms import LLMStream, LiteLLM
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

# TODO: better decompose the roles between LLMFunctionNode and LLMFunctionTemplate
class LLMFunctionTemplate(BaseModel):
    desc: str
    inputs: list[InputStep]
    outputs: list[OutputStep]
    demos: list[Any] = []
    
    def make_prompt(self, input_values: list) -> Prompt:
        lines = []
        for input in self.inputs:
            lines.append(f"{input.get_prefix()} ${{{input.name}}}")
        for output in self.outputs:
            lines.append(f"{output.get_prefix()} {output.desc}")
        format_desc = "\n".join(lines)
        
        lines = []
        for input, value in zip(self.inputs, input_values):
            lines.append(f"{input.get_prefix()} {value.content}")
        lines.append(self.outputs[0].get_prefix())
        input_prompt = "\n".join(lines)
        
        text = LLM_FUNCTION_TEMPLATE.format(
            function_desc=self.desc, format_desc=format_desc, input_prompt=input_prompt
        )
        return Prompt.from_user_message(text)
        
class LLMFunctionNode(Node):
    template_name: str
    input_offsets: list[int]
    
    def get_template(self, agent):
        template = agent.templates[self.template_name]
        if not isinstance(template, LLMFunctionTemplate):
            raise ValueError(f"Template {self.template_name} is not an LLMFunctionTemplate")
        return template
    
    def make_prompt(self, agent, tape: Tape):
        return self.get_template(agent).make_prompt(
            [tape.steps[offset] for offset in self.input_offsets]
        )
    
    def generate_steps(self, agent, tape: Tape, llm_stream: LLMStream):
        # TODO: streaming
        # TODO: more robust parsing that doesn't rely on ':' and '\n'
        # TODO: configure output step class
        template = self.get_template(agent)
        output_text = llm_stream.get_text()
        prompt_text = llm_stream.prompt.messages[0]["content"]
        text = prompt_text + "\n" + output_text
        output_lines = text.split("\n")[-len(template.outputs):]
        output_values = [output_lines[0]] + [line.split(":")[1].strip() for line in output_lines[1:]]
        for value in output_values:
            yield AssistantThought(content=value)
        
## TESTS        
    
TEST_INPUT_STEP = UserStep(content="What is the nationality of the chef and restaurateur featured in Restaurant: Impossible?")    
    
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

    render = func.make_prompt([TEST_INPUT_STEP]).messages[0]["content"]
    with open(res_path / "llm_function" / "qa.txt", "r") as f:
        gold = f.read()    
    if render != gold:
        print(diff_strings(render, gold))
        assert False
        
    llm = LiteLLM(model_name="gpt-3.5-turbo", parameters={"temperature": 0.})
    agent = Agent.create(
        llms=llm,        
        # TODO: change templates signature everywhere
        templates={"qa": func},
        nodes=[LLMFunctionNode(template_name="qa", input_offsets=[-1])],
    )
    final_tape = agent.run(DialogTape(steps=[TEST_INPUT_STEP]), max_iterations=1).get_final_tape()
    print(final_tape.model_dump_json(indent=2))
        
        
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
    
    input_step = UserStep(content="What is the nationality of the chef and restaurateur featured in Restaurant: Impossible?")
    render = func.make_prompt([input_step]).messages[0]["content"]
    with open(res_path / "llm_function" / "cot.txt", "r") as f:
        gold = f.read()    
    if render != gold:
        print(diff_strings(render, gold))
        assert False    
        
    llm = LiteLLM(model_name="gpt-3.5-turbo", parameters={"temperature": 0.})
    agent = Agent.create(
        llms=llm,        
        # TODO: change templates signature everywhere
        templates={"cot": func},
        nodes=[LLMFunctionNode(template_name="cot", input_offsets=[-1])],
    )
    final_tape = agent.run(DialogTape(steps=[TEST_INPUT_STEP]), max_iterations=1).get_final_tape()
    print(final_tape.model_dump_json(indent=2))
    
    
if __name__ == "__main__":
    test_dspy_qa_prompt()
    test_dspy_cot_prompt()