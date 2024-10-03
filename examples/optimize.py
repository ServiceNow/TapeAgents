import logging
import os

import hydra
from omegaconf import DictConfig

from tapeagents.io import load_tapes, save_tapes
from tapeagents.observe import retrieve_all_llm_calls, retrieve_tape_llm_calls
from tapeagents.rendering import PrettyRenderer
from tapeagents.studio import Studio
from tapeagents.tape_browser import TapeBrowser

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

import json
import pathlib
import dspy
import dsp.utils
import dspy.evaluate
from dsp.utils import deduplicate
from dspy.datasets import HotPotQA
import tqdm

from tapeagents.agent import Agent, Node
from tapeagents.core import Tape
from tapeagents.dialog_tape import AssistantStep, AssistantThought, DialogTape, FunctionCall, ToolCall, ToolCalls, ToolResult, UserStep
from tapeagents.environment import ToolEnvironment
from tapeagents.llm_function import InputStep, KindRef, LLMFunctionNode, LLMFunctionTemplate, NodeRef, OutputStep, RationaleStep
from tapeagents.llms import LLMStream, LiteLLM
from tapeagents.runtime import main_loop
from tapeagents.batch import batch_main_loop


res_dir = pathlib.Path(__file__).parent.resolve() / "res"


def render_contexts(contexts: list[str]) -> str:
    return "\n".join(f"[{i + 1}] «{t}»" for i, t in enumerate(contexts))


def retrieve(query: str) -> list[str]:
    """Retrieve Wikipedia abstracts"""
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query)
    return [r['text'] for r in results[:3]]


def load_few_shot_demos() -> tuple[list, list]:
    with open(res_dir / "llm_function_rag_demos.json") as f:
        demos_json = json.load(f)
    partial_demos = []
    demos = [] 
    for demo in demos_json:
        if demo.get("augmented"):
            demo = {
                "question": UserStep(content=demo["question"]),
                "context": ToolResult(content=demo["context"], tool_call_id=""),
                "rationale": AssistantThought(content=demo["rationale"]),
                "answer": AssistantStep(content=demo["answer"]),
            }            
            demos.append(demo)
        else:
            demo = {
                "question": UserStep(content=demo["question"]),
                "answer": AssistantStep(content=demo["answer"]),
            }
            partial_demos.append(demo)
    return partial_demos, demos

class ContextInput(InputStep):
    def render(self, step: ToolResult):
        return render_contexts(step.content)


def make_answer_template() -> LLMFunctionTemplate:
    partial_demos, demos = load_few_shot_demos()
    return LLMFunctionTemplate(
        desc="Answer questions with short factoid answers.",
        inputs=[
            ContextInput(name="context", desc="may contain relevant facts", separator="\n"),
            InputStep(name="question"),
        ],
        outputs=[
            RationaleStep.for_output("answer"),
            OutputStep(name="answer", desc="often between 1 and 5 words")
        ],
        demos=demos,
        partial_demos=partial_demos
    )        
    
    
def make_query_template() -> LLMFunctionTemplate:
    class RetrieveOutputStep(OutputStep):
        def parse(self, text: str):
            tc = ToolCall(function=FunctionCall(name="retrieve", arguments={"query": text}))
            return ToolCalls(tool_calls=[tc])
    return LLMFunctionTemplate(
        desc="Write a simple search query that will help answer a complex question.",
        inputs=[
            ContextInput(name="context", desc="may contain relevant facts", separator="\n"),
            InputStep(name="question"),
        ],
        outputs=[
            RationaleStep.for_output("query"),
            RetrieveOutputStep(name="query", desc="a search query")
        ]
    )       


def make_env() -> ToolEnvironment:
    return ToolEnvironment(tools=[retrieve])


def make_rag_agent(cfg: DictConfig) -> Agent:
    class RetrieveNode(Node):
        def generate_steps(self, agent, tape: Tape, llm_stream: LLMStream):
            assert isinstance(question := tape.steps[-1], UserStep)
            tc = {
                "function": {
                    "name": "retrieve",
                    "arguments": {"query": question.content}
                }
            }
            yield ToolCalls.from_dicts([tc])
    agent = Agent.create(
        llms=LiteLLM(model_name="gpt-3.5-turbo", parameters={"temperature": 0.}),        
        # TODO: change templates signature everywhere
        templates={"rag": make_answer_template()},
        nodes=[
            RetrieveNode(),
            LLMFunctionNode(template_name="rag", input_refs=[-1, -3])
        ],
    )
    if not cfg.rag.partial_demos:
        agent.templates["rag"].partial_demos = []   
    if not cfg.rag.demos:
        agent.templates["rag"].demos = []
    return agent


def make_agentic_rag_agent(cfg: DictConfig) -> Agent:
    templates = {
        "answer": make_answer_template(),
    }
    for i in range(cfg.agentic_rag.max_hops):
        templates[f"query{i}"] = make_query_template()
    
    class Deduplicate(Node):
        def generate_steps(self, agent, tape: Tape, llm_stream: LLMStream):
            contexts = []
            for step in tape:
                if isinstance(step, ToolResult):
                    contexts.extend(step.content)
            yield AssistantThought(content=deduplicate(contexts))            
            
    nodes = []
    for i in range(cfg.agentic_rag.max_hops):
        context_ref = KindRef.to(ToolResult) if i > 0 else AssistantThought(content=[])
        nodes.append(
            LLMFunctionNode(
                name=f"query{i}",
                template_name=f"query{i}",
                input_refs=[context_ref, KindRef.to(UserStep)],
            )
        )
        nodes.append(Deduplicate(name=f"deduplicate{i}"))
    nodes.append(
        LLMFunctionNode(
            template_name="answer", 
            input_refs=[NodeRef(name=f"deduplicate{i}"), KindRef.to(UserStep)]
        )
    )
    
    agent = Agent.create(
        llms=LiteLLM(model_name="gpt-3.5-turbo", parameters={"temperature": 0.}),
        templates=templates,
        nodes=nodes
    )
    return agent


def studio_few_shot_rag(cfg: DictConfig):
    agent = make_rag_agent(cfg)
    env = make_env()
    start_tape = DialogTape(steps=[UserStep(content="At My Window was released by which American singer-songwriter?")])
    Studio(agent, start_tape, PrettyRenderer(), env).launch()
    
    
def studio_agentic_rag(cfg: DictConfig):
    agent = make_agentic_rag_agent(cfg)
    env = make_env()
    start_tape = DialogTape(steps=[UserStep(content="How many storeys are in the castle that David Gregory inherited?")])
    Studio(agent, start_tape, PrettyRenderer(), env).launch()
    
    
def compute_retrieval_accuracy(examples: list, tapes: list[Tape]):
    n_correct = 0
    for example, tape in zip(examples, tapes):
        gold_titles = set(map(dspy.evaluate.normalize_text, example['gold_titles']))
        # TODO: just retrieve the last set of contexts by index, keep it simple
        for step in tape:
            if isinstance(step, ToolResult):
                found_titles = [c.split(' | ')[0] for c in step.content]
                found_titles = set(map(dspy.evaluate.normalize_text, found_titles))
        ok = gold_titles.issubset(found_titles)
        # print(gold_titles, found_titles, ok)
        n_correct += int(ok)
    return n_correct / len(examples)
    
    
def compute_answer_exact_match(examples: list, tapes: list[Tape]):
    n_correct = 0
    for example, final_tape in zip(examples, tapes):
        if isinstance(answer := final_tape.steps[-1], AssistantStep):
            ok = dspy.evaluate.answer_exact_match(example, dspy.primitives.Example({'answer': answer.content}))
            n_correct += int(ok)
    return n_correct / len(examples)
    
def get_dataset(cfg: DictConfig):    
    logger.info("Loading data ...")
    dataset = HotPotQA(
        train_seed=1, train_size=20, eval_seed=2023, 
        dev_size=cfg.dataset.dev_size, test_size=0
    )    
    logger.info("Data loaded")
    return dataset
    
def evaluate_few_shot_rag(cfg: DictConfig):
    agent = make_rag_agent(cfg)
    env = make_env()
    dataset = get_dataset(cfg)
    
    start_tapes = [DialogTape(steps=[UserStep(content=example["question"])]) for example in dataset.dev]
    final_tapes = []
    with save_tapes("few_shot_rag_tapes.yaml") as saver:
        for tape in tqdm.tqdm(batch_main_loop(agent, start_tapes, env)):
            final_tapes.append(tape)
            saver.save(tape)
        
    retrieval_accuracy = compute_retrieval_accuracy(dataset.dev, final_tapes)
    answer_accuracy = compute_answer_exact_match(dataset.dev, final_tapes)
    print(f"Retrieval accuracy: {retrieval_accuracy:.2f}")
    print(f"Answer accuracy: {answer_accuracy:.2f}")
       
        
def browse_tapes(): 
    tape_loader = lambda path: load_tapes(DialogTape, path)    
    browser = TapeBrowser(tape_loader, ".", PrettyRenderer())
    browser.launch()
    
    
@hydra.main(version_base=None, config_path="../conf/tapeagent", config_name="hotpot_qa")
def main(cfg: DictConfig):
    print(f"Running in {os.getcwd()}")
    match cfg.target:
        case "studio_few_shot_rag":
            studio_few_shot_rag(cfg)
        case "studio_agentic_rag":
            studio_agentic_rag(cfg)
        case "evaluate_fewshot_rag":
            evaluate_few_shot_rag(cfg)
        case "browse_tapes":
            browse_tapes()
        case _:
            raise ValueError(f"Unknown target {cfg.target}")
    
if __name__ == "__main__":
    main()