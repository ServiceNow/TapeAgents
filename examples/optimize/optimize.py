import json
import logging
import os
import random
from typing import Callable

import dspy
import dspy.evaluate
import hydra
import tqdm
from dsp.utils import deduplicate
from dspy.datasets import HotPotQA
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

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
from tapeagents.io import stream_yaml_tapes
from tapeagents.llm_function import LLMFunctionNode, by_node, by_step
from tapeagents.llms import LiteLLM, LLMStream
from tapeagents.orchestrator import main_loop
from tapeagents.renderers.camera_ready_renderer import CameraReadyRenderer
from tapeagents.renderers.pretty import PrettyRenderer
from tapeagents.studio import Studio
from tapeagents.tape_browser import TapeBrowser

from .func_templates import make_answer_template, make_query_template
from .load_demos import load_agentic_rag_demos, load_rag_demos

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


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
        "n": 1,
    }

    llm = LiteLLM(model_name=cfg.llm_name, parameters=parameters, use_cache=cfg.llm_cache)
    return llm


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


def add_demos(agent: Agent, tapes: list[Tape], max_n_demos: int, seed: int = 1) -> Agent:
    """Extract demos for function templates from the given tapes.

    When there is too many demos, select random ones.

    """
    demos = {template_name: [] for template_name in agent.templates}
    for tape in tapes:
        for node, index in agent.get_node_runs(tape):
            if isinstance(node, LLMFunctionNode):
                demos[node.template_name].append(node.extract_demo(agent, tape, index))
    rng = random.Random(seed)
    agent_copy = agent.model_copy(deep=True)
    for template_name, template in agent_copy.templates.items():
        k = min(max_n_demos, len(demos[template_name]))
        template.demos = rng.sample(demos[template_name], k)  # random selection of demos
    return agent_copy


def run_agent(agent: Agent, dataset: list, cfg: DictConfig) -> tuple[list[Tape], list[Tape]]:
    env = make_env(cfg.optimize.n_paragraphs)
    start_tapes = [DialogTape(steps=[UserStep(content=example["question"])]) for example in dataset]
    final_tapes = list(tqdm.tqdm(batch_main_loop(agent, start_tapes, env)))
    return final_tapes


def optimize_agent(agent: Agent, cfg: DictConfig) -> Agent:
    # Step 1: Run agent on the training set
    dataset = get_dataset(cfg)
    final_tapes = run_agent(agent, dataset.train, cfg)
    # Step 2: filter out good tapes
    good_tapes = [t for example, t in zip(dataset.train, final_tapes) if is_good_tape(example, t)]
    bad_tapes = [t for t in final_tapes if t not in good_tapes]
    logger.info(f"{len(good_tapes)} good tapes out of {len(final_tapes)}")
    # Save all tapes for observability
    with stream_yaml_tapes("good_training_tapes.yaml") as saver:
        for tape in good_tapes:
            saver.save(tape)
    with stream_yaml_tapes("bad_training_tapes.yaml") as saver:
        for tape in bad_tapes:
            saver.save(tape)
    # Step 3: Optimize agent from the good tapes
    better_agent = optimize_demos(agent, good_tapes, dataset.dev, cfg, compute_weighted_accuracy)
    return better_agent


def compute_weighted_accuracy(
    dataset: list, tapes: list[Tape], run_name: str = "", w_retrieval: float = 0.5, w_answer: float = 0.5
) -> float:
    """Compute the weighted average of the retrival and answer accuracy"""
    retrieval_accuracy = compute_retrieval_accuracy(dataset, tapes)
    answer_accuracy = compute_answer_exact_match(dataset, tapes)
    mean_accuracy = retrieval_accuracy * w_retrieval + answer_accuracy * w_answer
    logger.info(
        f"Retrieval accuracy: {retrieval_accuracy:.3f} | Answer accuracy: {answer_accuracy:.3f} | Mean Accuracy: {mean_accuracy:.3f} "
    )
    metrics_save_path = f"metrics_{len(dataset)}{f'_{run_name}' if run_name else ''}.json"
    metrics = {
        "mean_accuracy": mean_accuracy,
        "retrieval_accuracy": retrieval_accuracy,
        "answer_accuracy": answer_accuracy,
    }
    with open(metrics_save_path, "w") as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )
    return mean_accuracy


def optimize_demos(
    agent: Agent,
    good_tapes: list[Tape],
    val_dataset: list,
    cfg: DictConfig,
    metric_fn: Callable[[list, list[Tape]], float],
) -> Agent:
    """Try N times to `add_demos` (see above), measure val set performance, and keep the best agent"""
    best_agent = agent
    best_metric = 0

    for i in range(cfg.optimize.max_optimize_tries):
        # Add demos to the agent with a different seed for each attempt
        new_agent = add_demos(best_agent, good_tapes, cfg.optimize.max_n_demos, seed=cfg.seed + i)
        # Run agent on the validation set to get metric to optimize
        final_tapes = run_agent(new_agent, val_dataset, cfg)
        metric = metric_fn(val_dataset, final_tapes, run_name=f"validation_{i}")
        if metric > best_metric:
            best_metric = metric
            best_agent = new_agent
    return best_agent


def make_agent(cfg: DictConfig) -> Agent:
    agent = make_rag_agent(cfg) if cfg.agent == "rag" else make_agentic_rag_agent(cfg)
    if cfg.optimize.do:
        agent = optimize_agent(agent, cfg)
    return agent


def is_good_tape(example: dspy.primitives.Example, tape, trace=None):
    if len(tape.steps) < 3:  # Error in the tape
        logger.error(tape.metadata.error)
        return False
    pred = dspy.primitives.Example({"answer": str(tape.steps[-1].content).strip(), "context": tape.steps[-3].content})
    tape.metadata.result["groundruth_answer"] = example.answer
    if not dspy.evaluate.answer_exact_match(example, pred):
        tape.metadata.result["reason"] = "bad answer"
        return False
    if not dspy.evaluate.answer_passage_match(example, pred):
        tape.metadata.result["reason"] = "answer not in context"
        return False
    queries = [example.question]
    queries += [step.tool_calls[0].function.arguments["query"] for step in tape if isinstance(step, ToolCalls)]
    if max([len(q) for q in queries]) > 200:
        tape.metadata.result["reason"] = "long query"
        return False
    if any(
        dspy.evaluate.answer_exact_match_str(queries[idx], queries[:idx], frac=0.8) for idx in range(2, len(queries))
    ):
        tape.metadata.result["reason"] = "repeated query"
        return False
    tape.metadata.result["reason"] = "good tape"
    return True


def compute_retrieval_accuracy(examples: list, tapes: list[Tape]):
    n_correct = 0
    for example, tape in zip(examples, tapes):
        gold_titles = set(map(dspy.evaluate.normalize_text, example["gold_titles"]))
        # TODO: just retrieve the last set of contexts by index, keep it simple
        if len(tape.steps) < 3:  # Error in the tape
            tape.metadata.result["retrieval_accurate"] = False
            continue
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
            ok = dspy.evaluate.answer_exact_match(
                example, dspy.primitives.Example({"answer": str(answer.content).strip()})
            )
            tape.metadata.result["answer_accurate"] = ok
            n_correct += int(ok)
    return n_correct / len(examples)


_dataset = None


def get_dataset(cfg: DictConfig) -> HotPotQA:
    logger.info("Loading data ...")
    global _dataset
    if _dataset is None:
        _dataset = HotPotQA(
            train_seed=1,
            train_size=cfg.dataset.train_size,
            eval_seed=2023,
            dev_size=cfg.dataset.dev_size,
            test_size=cfg.dataset.test_size,
        )
    logger.info("Data loaded")
    return _dataset


def batch_run_and_save(agent: Agent, env: ToolEnvironment, dataset: list, save_tapes_path: str):
    start_tapes = [DialogTape(steps=[UserStep(content=example["question"])]) for example in dataset]
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
    tapes_save_path = f"test_tapes_{cfg.dataset.test_size}.yaml"
    final_tapes = batch_run_and_save(agent, env, dataset.test, tapes_save_path)
    compute_weighted_accuracy(dataset.test, final_tapes, run_name="test")


def browse():
    run_dir = HydraConfig.get().runtime.output_dir
    browser = TapeBrowser(DialogTape, run_dir, CameraReadyRenderer())
    browser.launch()


@hydra.main(version_base=None, config_path="../../conf", config_name="optimize_hotpotqa")
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
