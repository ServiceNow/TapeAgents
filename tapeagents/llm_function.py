import json

from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field

from tapeagents.agent import Agent, Node
from tapeagents.core import Prompt, Step, Tape
from tapeagents.dialog_tape import AssistantThought, DialogStep, DialogTape, ToolCalls, UserStep
from tapeagents.environment import ToolEnvironment
from tapeagents.llm_function_template import LLM_FUNCTION_TEMPLATE
from tapeagents.llms import LLMStream, LiteLLM
from tapeagents.observe import retrieve_tape_llm_calls
from tapeagents.utils import diff_strings


class InputStep(BaseModel):
    name: str
    prefix: str = ""
    desc: str = ""
    
    def get_prefix(self):
        return self.prefix or f"{self.name.title()}:"
   
    
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
    partial_demos: list[Tape] = Field(
        default=[], description="Tape that contain key outputs but may lack intermediate steps"
    )
    demos: list[Tape] = Field(
        default=[], description="Tapes that contain all input and all output steps"
    )
    
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
    
    
def test_fewshot_prompt():
    with open(res_path / "llm_function" / "rag_demos.json") as f:
        demos = json.load(f)
        
    
        
    
def test_basic_rag_agent():
    import dspy 
    def retrieve(query: str) -> str:
        """Retrieve Wikipedia abstracts"""
        results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query)
        texts = [r['text'] for r in results[:3]]
        return "\n".join(f"[{i + 1}] «{t}»" for i, t in enumerate(texts))
    env = ToolEnvironment(tools=[retrieve])
    
    tc = {
        "function": {
            "name": "retrieve",
            "arguments": json.dumps({"query": "What castle did David Gregory inherit?"})
        }
    }
    steps = [
        UserStep(content="What castle did David Gregory inherit?"),
        ToolCalls.from_dicts([tc])
    ]
    start_tape = DialogTape(steps=steps)
    tape_with_context = env.react(start_tape)
    # print(tape.model_dump_json(indent=2)) 
    
    func = LLMFunctionTemplate(
        desc="Answer questions with short factoid answers.",
        inputs=[
            InputStep(name="question"),
            InputStep(name="context")
        ],
        outputs=[
            OutputStep(prefix="Reasoning: Let's think step by step in order to", desc="${produce the answer}. We ..."),
            OutputStep(name="answer", desc="often between 1 and 5 words")
        ]
    )  
    
    agent = Agent.create(
        llms=LiteLLM(model_name="gpt-3.5-turbo", parameters={"temperature": 0.}),        
        # TODO: change templates signature everywhere
        templates={"rag": func},
        nodes=[LLMFunctionNode(template_name="rag", input_offsets=[-3, -1])],
    )
    final_tape = agent.run(tape_with_context, max_iterations=1).get_final_tape()
    calls = retrieve_tape_llm_calls(final_tape)
    print(list(calls.values())[0].prompt.messages[0]["content"])
    print(list(calls.values())[0].output.content)
    
    
if __name__ == "__main__":
    # test_dspy_qa_prompt()
    # test_dspy_cot_prompt()
    test_basic_rag_agent()