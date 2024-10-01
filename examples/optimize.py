
import json
import pathlib
import dspy
from dspy.datasets import HotPotQA

from tapeagents.agent import Agent, Node
from tapeagents.core import Tape
from tapeagents.dialog_tape import AssistantThought, DialogTape, ToolCalls, UserStep
from tapeagents.environment import ToolEnvironment
from tapeagents.llm_function import InputStep, LLMFunctionNode, LLMFunctionTemplate, OutputStep
from tapeagents.llms import LLMStream, LiteLLM
from tapeagents.runtime import main_loop


res_dir = pathlib.Path(__file__).parent.resolve() / "res"


def render_contexts(contexts: list[str]) -> str:
    return "\n".join(f"[{i + 1}] «{t}»" for i, t in enumerate(contexts))


def retrieve(query: str) -> str:
    """Retrieve Wikipedia abstracts"""
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query)
    texts = [r['text'] for r in results[:3]]
    return "\n".join(f"[{i + 1}] «{t}»" for i, t in enumerate(texts))


def make_rag_function_template() -> LLMFunctionTemplate:
    with open(res_dir / "llm_function_rag_demos.json") as f:
        demos_json = json.load(f)
    partial_demos = []
    demos = [] 
    for demo in demos_json:
        if demo.get("augmented"):
            demo["context"] = render_contexts(demo["context"])
            demos.append(demo)
        else:
            partial_demos.append(demo)
        
    return LLMFunctionTemplate(
        desc="Answer questions with short factoid answers.",
        inputs=[
            InputStep(name="context", desc="may contain relevant facts", separator="\n"),
            InputStep(name="question"),
        ],
        outputs=[
            OutputStep(
                name="rationale", 
                prefix="Reasoning: Let's think step by step in order to", 
                desc="${produce the answer}. We ...",
                _step_class=AssistantThought
            ),
            OutputStep(name="answer", desc="often between 1 and 5 words")
        ],
        demos=demos,
        partial_demos=partial_demos
    )        


def make_rag_agent_and_env() -> tuple[Agent, ToolEnvironment]:
    env = ToolEnvironment(tools=[retrieve])
    
    class RetrieveNode(Node):
        def generate_steps(self, agent, tape: Tape, llm_stream: LLMStream):
            assert isinstance(question := tape.steps[-1], UserStep)
            tc = {
                "function": {
                    "name": "retrieve",
                    "arguments": json.dumps({"query": question.content})
                }
            }
            yield ToolCalls.from_dicts([tc])
    agent = Agent.create(
        llms=LiteLLM(model_name="gpt-3.5-turbo", parameters={"temperature": 0.}),        
        # TODO: change templates signature everywhere
        templates={"rag": make_rag_function_template()},
        nodes=[
            RetrieveNode(),
            LLMFunctionNode(template_name="rag", input_offsets=[-3, -1])
        ],
    )
    return agent, env


def run_ran_agent():
    agent, env = make_rag_agent_and_env()
    start_tape = DialogTape(steps=[UserStep(content="At My Window was released by which American singer-songwriter?")])
    final_tape = main_loop(agent, start_tape, env).get_final_tape()
    print(final_tape.model_dump_json(indent=2))

    
if __name__ == "__main__":
    run_ran_agent()