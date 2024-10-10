import logging
import os

import hydra
from omegaconf import DictConfig

from tapeagents.io import stream_yaml_tapes
from tapeagents.optimize import add_demos
from tapeagents.rendering import PrettyRenderer
from tapeagents.studio import Studio
from tapeagents.tape_browser import TapeBrowser

from .func_templates import make_answer_template, make_query_template
from .load_demos import load_agentic_rag_demos, load_rag_demos

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

import json
import pathlib
import random

import dspy
import dspy.evaluate
import tqdm
from dsp.utils import deduplicate
from dspy.datasets import HotPotQA

from tapeagents.agent import Agent, Node
from tapeagents.batch import batch_main_loop
from tapeagents.core import Tape
from tapeagents.dialog_tape import (
    AssistantStep,
    AssistantThought,
    DialogTape,
    FunctionCall,
    ToolCall,
    ToolCalls,
    ToolResult,
    UserStep,
)
from tapeagents.environment import ToolEnvironment
from tapeagents.llm_function import KindRef, LLMFunctionNode, NodeRef, by_node, by_step
from tapeagents.llms import LiteLLM, LLMStream
from tapeagents.runtime import main_loop

res_dir = pathlib.Path(__file__).parent.parent.resolve() / "res"


def make_env(n_paragraphs: int = 3) -> ToolEnvironment:
    def retrieve(query: str) -> list[str]:
        """Retrieve Wikipedia abstracts"""
        results = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")(query)
        return [r["text"] for r in results[:n_paragraphs]]

    return ToolEnvironment(tools=[retrieve])


def make_llm(cfg: DictConfig) -> LiteLLM:
    parameters = {
        "temperature": 0.0,
        "max_tokens": 150,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "n": 1,
    }
    return LiteLLM(model_name="gpt-3.5-turbo", parameters=parameters, use_cache=cfg.llm_cache)


def make_rag_agent(cfg: DictConfig) -> Agent:
    class RetrieveNode(Node):
        def generate_steps(self, agent, tape: Tape, llm_stream: LLMStream):
            query = tape.steps[-1].content
            tc = ToolCall(function=FunctionCall(name="retrieve", arguments={"query": query}))
            yield ToolCalls(tool_calls=[tc])

    agent = Agent.create(
        llms=make_llm(cfg),
        templates={"rag": make_answer_template()},
        nodes=[RetrieveNode(), LLMFunctionNode(template_name="rag", input_refs=[-1, -3])],
    )
    if cfg.load_demos:
        partial_demos, demos = load_rag_demos()
        agent.templates["rag"].demos = demos
        agent.templates["rag"].partial_demos = partial_demos
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
        context_ref = by_step(ToolResult) if i > 0 else AssistantThought(content=[])
        nodes.append(
            LLMFunctionNode(
                name=f"query{i}",
                template_name=f"query{i}",
                input_refs=[context_ref, by_step(UserStep)],
            )
        )
        nodes.append(Deduplicate(name=f"deduplicate{i}"))
    nodes.append(LLMFunctionNode(template_name="answer", input_refs=[by_node(nodes[-1]), by_step(UserStep)]))

    agent = Agent.create(llms=make_llm(cfg), templates=templates, nodes=nodes)

    if cfg.load_demos:
        all_demos = load_agentic_rag_demos()
        for template_name, (partial_demos, demos) in all_demos.items():
            agent.templates[template_name].demos = demos
            agent.templates[template_name].partial_demos = partial_demos

    return agent


def optimize_agent(agent: Agent, cfg: DictConfig):
    # Step 1: run agent on the training set
    dataset = get_dataset(cfg)
    env = make_env(cfg.optimize.n_paragraphs)
    start_tapes = [DialogTape(steps=[UserStep(content=example["question"])]) for example in dataset.train]
    final_tapes = list(tqdm.tqdm(batch_main_loop(agent, start_tapes, env)))
    # Step 2: filter out good tapes
    good_tapes = [t for example, t in zip(dataset.train, final_tapes) if is_good_tape(example, t)]
    logger.info(f"{len(good_tapes)} good tapes out of {len(final_tapes)}")
    # Save all tapes for observability
    with stream_yaml_tapes("good_training_tapes.yaml") as saver:
        for tape in good_tapes:
            saver.save(tape)
    with stream_yaml_tapes("bad_training_tapes.yaml") as saver:
        for tape in final_tapes:
            if tape not in good_tapes:
                saver.save(tape)
    better_agent = add_demos(agent, good_tapes, cfg.optimize.max_n_demos, seed=cfg.seed)
    return better_agent


def make_agent(cfg: DictConfig) -> Agent:
    agent = make_rag_agent(cfg) if cfg.agent == "rag" else make_agentic_rag_agent(cfg)
    if cfg.optimize.do:
        agent = optimize_agent(agent, cfg)
    return agent


def is_good_tape(example: dspy.primitives.Example, tape, trace=None):
    pred = dspy.primitives.Example({"answer": tape.steps[-1].content, "context": tape.steps[-3].content})
    tape.metadata.result["groundruth_answer"] = example.answer
    if not dspy.evaluate.answer_exact_match(example, pred):
        tape.metadata.result["reason"] = "bad answer"
        return False
    if not dspy.evaluate.answer_passage_match(example, pred):
        tape.metadata.result["reason"] = "answer not in context"
        return False
    queries = [example.question]
    queries += [step.tool_calls[0].function.arguments["query"] for step in tape if isinstance(step, ToolCalls)]
    if max([len(q) for q in queries]) > 100:
        tape.metadata.result["reason"] = "long query"
        return False
    if any(
        dspy.evaluate.answer_exact_match_str(queries[idx], queries[:idx], frac=0.8) for idx in range(2, len(queries))
    ):
        tape.metadata.result = "repeated query"
        return False
    tape.metadata.result["reason"] = "good tape"
    return True


def compute_retrieval_accuracy(examples: list, tapes: list[Tape]):
    n_correct = 0
    for example, tape in zip(examples, tapes):
        gold_titles = set(map(dspy.evaluate.normalize_text, example["gold_titles"]))
        # TODO: just retrieve the last set of contexts by index, keep it simple
        context_step = tape.steps[-3]
        found_titles = [c.split(" | ")[0] for c in context_step.content]
        found_titles = set(map(dspy.evaluate.normalize_text, found_titles))
        ok = gold_titles.issubset(found_titles)
        tape.metadata.result["retrieval_accurate"] = ok
        n_correct += int(ok)
    return n_correct / len(examples)


def compute_answer_exact_match(examples: list, tapes: list[Tape]):
    n_correct = 0
    for example, tape in zip(examples, tapes):
        tape.metadata.result["groundruth_answer"] = example.answer
        if isinstance(answer := tape.steps[-1], AssistantStep):
            ok = dspy.evaluate.answer_exact_match(example, dspy.primitives.Example({"answer": answer.content}))
            tape.metadata.result["answer_accurate"] = ok
            n_correct += int(ok)
    return n_correct / len(examples)


_dataset = None


def get_dataset(cfg: DictConfig):
    logger.info("Loading data ...")
    global _dataset
    if _dataset is None:
        _dataset = HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=cfg.dataset.dev_size, test_size=0)
    logger.info("Data loaded")
    return _dataset


def batch_run_and_save(agent: Agent, env: ToolEnvironment, dataset, save_tapes_path: str):
    start_tapes = [DialogTape(steps=[UserStep(content=example["question"])]) for example in dataset.dev]
    final_tapes = []
    with stream_yaml_tapes(save_tapes_path) as saver:
        for tape in tqdm.tqdm(batch_main_loop(agent, start_tapes, env)):
            final_tapes.append(tape)
            saver.save(tape)
    return final_tapes


def run(cfg: DictConfig):
    agent = make_agent(cfg)
    env = make_env()
    start_tape = DialogTape(steps=[UserStep(content=cfg.question)])
    final_tape = main_loop(agent, start_tape, env).get_final_tape()
    final_tape_str = final_tape.model_dump_json(indent=2)
    with open("tape.json", "w") as f:
        print(final_tape_str, file=f)
    print(final_tape_str)


def studio(cfg: DictConfig):
    agent = make_agent(cfg)
    env = make_env()
    start_tape = DialogTape(steps=[UserStep(content=cfg.question)])
    Studio(agent, start_tape, PrettyRenderer(), env).launch()


def evaluate(cfg: DictConfig):
    agent = make_agent(cfg)
    env = make_env()
    dataset = get_dataset(cfg)
    tapes_save_path = f"dev_tapes_{cfg.dataset.dev_size}.yaml"
    final_tapes = batch_run_and_save(agent, env, dataset, tapes_save_path)
    retrieval_accuracy = compute_retrieval_accuracy(dataset.dev, final_tapes)
    answer_accuracy = compute_answer_exact_match(dataset.dev, final_tapes)
    print(f"Retrieval accuracy: {retrieval_accuracy:.2f}")
    print(f"Answer accuracy: {answer_accuracy:.2f}")
    metrics_save_path = f"metrics_{cfg.dataset.dev_size}.json"
    with open(metrics_save_path, "w") as f:
        json.dump({"retrieval_accuracy": retrieval_accuracy, "answer_accuracy": answer_accuracy}, f, indent=2)


def browse():
    browser = TapeBrowser(DialogTape, ".", PrettyRenderer())
    browser.launch()


@hydra.main(version_base=None, config_path="../../conf", config_name="hotpot_qa")
def main(cfg: DictConfig):
    print(f"Running in {os.getcwd()}")
    match cfg.target:
        case "run":
            run(cfg)
        case "studio":
            studio(cfg)
        case "evaluate":
            evaluate(cfg)
        case "browse":
            browse()
        case _:
            raise ValueError(f"Unknown target {cfg.target}")


if __name__ == "__main__":
    main()
