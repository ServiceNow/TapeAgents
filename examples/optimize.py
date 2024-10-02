import logging
import os

import hydra
from omegaconf import DictConfig

from tapeagents.io import load_tapes, save_tapes
from tapeagents.observe import retrieve_all_llm_calls, retrieve_tape_llm_calls
from tapeagents.rendering import PrettyRenderer
from tapeagents.tape_browser import TapeBrowser

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

import json
import pathlib
import dspy
from dspy.datasets import HotPotQA
from dspy.evaluate.metrics import answer_exact_match
import tqdm

from tapeagents.agent import Agent, Node
from tapeagents.core import Tape
from tapeagents.dialog_tape import AssistantStep, AssistantThought, DialogTape, ToolCalls, UserStep
from tapeagents.environment import ToolEnvironment
from tapeagents.llm_function import InputStep, LLMFunctionNode, LLMFunctionTemplate, OutputStep
from tapeagents.llms import LLMStream, LiteLLM
from tapeagents.runtime import main_loop
from tapeagents.batch import batch_main_loop


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


def make_rag_agent_and_env(cfg: DictConfig) -> tuple[Agent, ToolEnvironment]:
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
            LLMFunctionNode(template_name="rag", input_offsets=[-1, -3])
        ],
    )
    if not cfg.rag.partial_demos:
        agent.templates["rag"].partial_demos = []   
    if not cfg.rag.demos:
        agent.templates["rag"].demos = []
    return agent, env


def run_few_shot_rag_agent(cfg: DictConfig):
    agent, env = make_rag_agent_and_env(cfg)
    start_tape = DialogTape(steps=[UserStep(content="At My Window was released by which American singer-songwriter?")])
    final_tape = main_loop(agent, start_tape, env).get_final_tape()
    print(final_tape.model_dump_json(indent=2))
    
    
def evaluate_few_shot_rag(cfg: DictConfig):
    agent, env = make_rag_agent_and_env(cfg)
    logger.info("Loading data")
    dataset = HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=cfg.dataset.dev_size, test_size=0)
    logger.info("Data loaded")
    
    start_tapes = [DialogTape(steps=[UserStep(content=example["question"])]) for example in dataset.dev]
    final_tapes = []
    with save_tapes(pathlib.Path("few_shot_rag_tapes.yaml")) as saver:
        for tape in tqdm.tqdm(batch_main_loop(agent, start_tapes, env)):
            final_tapes.append(tape)
            saver.save(tape)
        
    n_correct = 0
    for example, final_tape in zip(dataset.dev, final_tapes):
        if isinstance(answer := final_tape.steps[-1], AssistantStep):
            ok = answer_exact_match(example, dspy.primitives.Example({'answer': answer.content}))
            # print(example.answer, answer.content, ok)
            n_correct += int(ok)
    print(f"Accuracy: {n_correct / len(dataset.dev):.2%}")
       
        
def browse_tapes(): 
    tape_loader = lambda path: load_tapes(DialogTape, path)    
    browser = TapeBrowser(tape_loader, ".", PrettyRenderer())
    browser.launch()
    
    
@hydra.main(version_base=None, config_path="../conf/tapeagent", config_name="hotpot_qa")
def main(cfg: DictConfig):
    print(f"Running in {os.getcwd()}")
    match cfg.target:
        case "evaluate_fewshot_rag":
            evaluate_few_shot_rag(cfg)
        case "browse_tapes":
            browse_tapes()
        case _:
            raise ValueError(f"Unknown target {cfg.target}")
    
if __name__ == "__main__":
    main()