import json

from pathlib import Path
from typing import Any, cast
from pydantic import BaseModel, Field

from tapeagents.agent import Agent, Node
from tapeagents.core import Pass, Prompt, Step, Tape
from tapeagents.dialog_tape import AssistantThought, DialogStep, DialogTape, ToolCalls, ToolResult, UserStep
from tapeagents.environment import ToolEnvironment
from tapeagents.llm_function_template import LLM_FUNCTION_TEMPLATE
from tapeagents.llms import LLMStream, LiteLLM
from tapeagents.observe import retrieve_tape_llm_calls
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
   
    
class OutputStep(BaseModel):
    name: str = ""
    prefix: str = ""
    desc: str
    separator: str = " "
    
    def get_prefix(self):
        return self.prefix or f"{self.name.title()}:"
    

# TODO: better decompose the roles between LLMFunctionNode and LLMFunctionTemplate
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
                    lines.append(f"{input.get_prefix()}{input.separator}{demo[input.name]}")
            for output in self.outputs:
                if output.name in demo:
                    lines.append(f"{output.get_prefix()}{output.separator}{demo[output.name]}")
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
            lines.append(f"{input.get_prefix()}{input.separator}{value.content}")
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
        
## EXAMPLE 

def render_contexts(contexts: list[str]) -> str:
    return "\n".join(f"[{i + 1}] «{t}»" for i, t in enumerate(contexts))
        
## TESTS        
    
TEST_INPUT_STEP1 = UserStep(content="What is the nationality of the chef and restaurateur featured in Restaurant: Impossible?")    
TEST_INPUT_STEP2 = UserStep(content="What castle did David Gregory inherit?")
TEST_CONTEXT_STEP2 = ToolResult(content=
"""[1] «David Gregory (physician) | David Gregory (20 December 1625 – 1720) was a Scottish physician and inventor. His surname is sometimes spelt as Gregorie, the original Scottish spelling. He inherited Kinnairdy Castle in 1664. Three of his twenty-nine children became mathematics professors. He is credited with inventing a military cannon that Isaac Newton described as "being destructive to the human species". Copies and details of the model no longer exist. Gregory's use of a barometer to predict farming-related weather conditions led him to be accused of witchcraft by Presbyterian ministers from Aberdeen, although he was never convicted.»
[2] «Gregory Tarchaneiotes | Gregory Tarchaneiotes (Greek: Γρηγόριος Ταρχανειώτης , Italian: "Gregorio Tracanioto" or "Tracamoto" ) was a "protospatharius" and the long-reigning catepan of Italy from 998 to 1006. In December 999, and again on February 2, 1002, he reinstituted and confirmed the possessions of the abbey and monks of Monte Cassino in Ascoli. In 1004, he fortified and expanded the castle of Dragonara on the Fortore. He gave it three circular towers and one square one. He also strengthened Lucera.»
[3] «David Gregory (mathematician) | David Gregory (originally spelt Gregorie) FRS (? 1659 – 10 October 1708) was a Scottish mathematician and astronomer. He was professor of mathematics at the University of Edinburgh, Savilian Professor of Astronomy at the University of Oxford, and a commentator on Isaac Newton's "Principia".»""",
tool_call_id=""
)
    
res_path = Path(__file__).parent.resolve() / ".." / "tests" / "res"
    
def test_dspy_qa_prompt():
    func = LLMFunctionTemplate(
        desc="Answer questions with short factoid answers.",
        inputs=[
            InputStep(name="question")
        ],
        outputs=[
            OutputStep(name="answer", desc="often between 1 and 5 words")
        ]
    )    

    render = func.make_prompt([TEST_INPUT_STEP1]).messages[0]["content"]
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
    final_tape = agent.run(DialogTape(steps=[TEST_INPUT_STEP1]), max_iterations=1).get_final_tape()
        
        
def test_dspy_cot_prompt():
    func = LLMFunctionTemplate(
        desc="Answer questions with short factoid answers.",
        inputs=[
            InputStep(name="question")
        ],
        outputs=[
            OutputStep(prefix="Reasoning: Let's think step by step in order to", desc="${produce the answer}. We ..."),
            OutputStep(name="answer", desc="often between 1 and 5 words")
        ]
    )
    
    render = func.make_prompt([TEST_INPUT_STEP1]).messages[0]["content"]
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
    final_tape = agent.run(DialogTape(steps=[TEST_INPUT_STEP1]), max_iterations=1).get_final_tape()
    
    
def test_fewshot_prompt():
    with open(res_path / "llm_function" / "rag_demos.json") as f:
        demos_json = json.load(f)
    partial_demos = []
    demos = [] 
    for demo in demos_json:
        if demo.get("augmented"):
            demo["context"] = render_contexts(demo["context"])
            demos.append(demo)
        else:
            partial_demos.append(demo)
        
    func = LLMFunctionTemplate(
        desc="Answer questions with short factoid answers.",
        inputs=[
            InputStep(name="context", desc="may contain relevant facts", separator="\n"),
            InputStep(name="question"),
        ],
        outputs=[
            OutputStep(name="rationale", prefix="Reasoning: Let's think step by step in order to", desc="${produce the answer}. We ..."),
            OutputStep(name="answer", desc="often between 1 and 5 words")
        ],
        demos=demos,
        partial_demos=partial_demos
    )        
        
    render = func.make_prompt([TEST_CONTEXT_STEP2, TEST_INPUT_STEP2]).messages[0]["content"]
    with open(res_path / "llm_function" / "rag.txt", "r") as f:
        gold = f.read()    
    def remove_empty_lines(text: str) -> str:
        return "\n".join(filter(lambda x: x.strip(), text.split("\n")))
    render = remove_empty_lines(render)
    gold = remove_empty_lines(gold)
    if render != gold:
        print(diff_strings(render, gold))
        assert False
        
    
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

    
if __name__ == "__main__":
    test_dspy_qa_prompt()
    test_dspy_cot_prompt()
    test_fewshot_prompt()
    test_basic_rag_agent()