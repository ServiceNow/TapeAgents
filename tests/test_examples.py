import contextlib
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))

from examples import data_science, tape_improver
from examples.delegate import ExampleTape, FindIrregularVerbs
from examples.delegate_stack import ExampleTape as ExampleTapeStack
from examples.delegate_stack import Linguist, make_analyze_text_chain
from examples.gaia_agent.agent import GaiaAgent
from examples.gaia_agent.environment import GaiaEnvironment
from examples.gaia_agent.eval import load_results
from examples.gaia_agent.tape import GaiaTape
from examples.llama_agent import LLAMAChatBot
from tapeagents.config import DB_DEFAULT_FILENAME
from tapeagents.core import AgentStep, TrainingText
from tapeagents.dialog_tape import DialogTape
from tapeagents.environment import EmptyEnvironment
from tapeagents.llms import TrainableLLM, LLM, ReplayLLM
from tapeagents.observe import LLMCall, init_sqlite_if_not_exists, retrieve_tape_llm_calls
from tapeagents.runtime import replay_tape, replay_tapes
from tapeagents.team import TeamTape

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

res_path = Path(__file__).parent.resolve() / "res"


def mock_llm(run_dir: str) -> LLM:
    llama = TrainableLLM(
        base_url="https://api.together.xyz",
        model_name="meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
        tokenizer_name="meta-llama/Meta-Llama-3-70B-Instruct",
        parameters=dict(temperature=0.7, max_tokens=512),
    )
    return ReplayLLM.from_llm(llama, run_dir)


@contextlib.contextmanager
def set_sqlite_db_dir(dir: str):
    old_path = os.environ.get("TAPEAGENTS_SQLITE_DB")
    new_path = f"{dir}/{DB_DEFAULT_FILENAME}"
    os.environ["TAPEAGENTS_SQLITE_DB"] = new_path
    init_sqlite_if_not_exists(only_once=False)
    try:
        yield
    finally:
        if old_path is not None:
            os.environ["TAPEAGENTS_SQLITE_DB"] = old_path
        else:
            del os.environ["TAPEAGENTS_SQLITE_DB"]


def load_tape_dict(run_dir: str, fname: str = "tape.json") -> dict:
    tape_fpath = os.path.join(run_dir, fname)
    with open(tape_fpath, "r") as f:
        tape_dict = json.load(f)
    return tape_dict


def load_traces(run_dir: str, fname: str = "traces.json") -> list[TrainingText]:
    traces_fpath = os.path.join(run_dir, fname)
    with open(traces_fpath, "r") as f:
        traces = [TrainingText.model_validate(t) for t in json.load(f)]
    return traces


def test_llama_agent():
    run_dir = str(res_path / "llama_agent")
    llm = mock_llm(run_dir)
    agent = LLAMAChatBot.create(llm)
    tape = DialogTape.model_validate(load_tape_dict(run_dir))

    assert replay_tape(agent, tape, reuse_observations=True)


def test_llama_agent_traces():
    run_dir = f"{res_path}/llama_agent"
    llm = mock_llm(run_dir)
    agent = LLAMAChatBot.create(llm)
    tape = DialogTape.model_validate(load_tape_dict(run_dir))
    orig_traces = load_traces(run_dir)

    with set_sqlite_db_dir(run_dir):
        traces = agent.make_training_data(tape)
        assert len(traces) == len(orig_traces), f"Expected {len(orig_traces)} traces, got {len(traces)}"
        for trace, orig_trace in zip(traces, orig_traces):
            assert trace.prompt_text == orig_trace.prompt_text
            assert trace.output_text == orig_trace.output_text


def test_llama_agent_tape_reuse():
    data_dir = f"{res_path}/llama_agent"
    llm = mock_llm(data_dir)
    agent = LLAMAChatBot.create(llm)
    tape = DialogTape.model_validate(load_tape_dict(data_dir))
    orig_traces = load_traces(data_dir)

    with tempfile.TemporaryDirectory() as run_dir:
        with set_sqlite_db_dir(run_dir):
            reused_tape, _ = agent.reuse(tape)
            for reused_step, step in zip(reused_tape, tape):
                if isinstance(step, AgentStep):
                    assert isinstance(reused_step, AgentStep)
                    assert reused_step.metadata.prompt_id != step.metadata.prompt_id
            traces_from_logs = [
                agent.make_training_text(llm_call) for llm_call in retrieve_tape_llm_calls(reused_tape).values()
            ]
            direct_traces = agent.make_training_data(tape)
            for traces in [traces_from_logs, direct_traces]:
                assert len(traces) == len(orig_traces), f"Expected {len(orig_traces)} traces, got {len(traces)}"
                for trace, orig_trace in zip(traces, orig_traces):
                    assert trace.prompt_text == orig_trace.prompt_text
                    assert trace.output_text == orig_trace.output_text


def test_gaia_agent():
    run_dir = str(res_path / "gaia_agent")
    results = load_results(os.path.join(run_dir, "results.json"))

    llm = ReplayLLM(llm_calls=[LLMCall.model_validate(p) for p in results.prompts], model_name=results.model)
    env = GaiaEnvironment(only_cached_webpages=True, safe_calculator=False)
    env.browser.set_web_cache(results.web_cache)
    agent = GaiaAgent(llms={"default": llm}, short_steps=True)

    tapes = [GaiaTape.model_validate(tape) for tape in results.tapes]
    logger.info(f"Validate {len(tapes)} tapes")

    fails = replay_tapes(agent, tapes, env)
    # two expected failures due to changed parsing exception format
    assert fails == 2, f"{fails} failed tapes, expected 2"


def test_delegate():
    run_dir = str(res_path / "delegate")
    llm = mock_llm(run_dir)
    agent = FindIrregularVerbs.create(llm)

    start_tape = ExampleTape.model_validate(load_tape_dict(run_dir, fname="start_tape.json"))
    tape = ExampleTape.model_validate(load_tape_dict(run_dir))

    assert replay_tape(agent, tape, start_tape=start_tape, reuse_observations=True)


def test_delegate_stack():
    run_dir = str(res_path / "delegate_stack")
    llm = mock_llm(run_dir)
    agent1 = Linguist.create(llm)
    agent2 = make_analyze_text_chain(llm)

    start_tape = ExampleTapeStack.model_validate(load_tape_dict(run_dir, fname="start_tape.json"))
    tape1 = ExampleTapeStack.model_validate(load_tape_dict(run_dir, fname="tape1.json"))
    tape2 = ExampleTapeStack.model_validate(load_tape_dict(run_dir, fname="tape2.json"))

    assert replay_tape(agent1, tape1, start_tape=start_tape, reuse_observations=True)
    assert replay_tape(agent2, tape2, start_tape=start_tape, reuse_observations=True)


def test_data_science():
    run_dir = f"{res_path}/data_science"
    llm = mock_llm(run_dir)
    agent, start_tape, env = data_science.make_world(llm, EmptyEnvironment())
    final_tape = TeamTape.model_validate(load_tape_dict(run_dir, "final_tape.json"))
    assert replay_tape(agent, final_tape, start_tape=start_tape, env=env, reuse_observations=True)


def test_tape_improver():
    run_dir = f"{res_path}/tape_improver"
    llm = mock_llm(run_dir)
    agent, _, improver_tape = tape_improver.make_world(llm)
    final_tape = tape_improver.CodeImproverTape.model_validate(load_tape_dict(run_dir, "final_tape.json"))
    assert replay_tape(agent, final_tape, start_tape=improver_tape, reuse_observations=True)


if __name__ == "__main__":
    test_llama_agent()
    test_llama_agent_traces()
    test_gaia_agent()
    test_delegate()
    test_delegate_stack()
    test_data_science()
    test_tape_improver()
