import contextlib
import gzip
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import yaml
from make_test_data import run_test_in_tmp_dir
from omegaconf import DictConfig

from examples.gsm8k_tuning.finetune_student import get_training_samples_from_tapes
from examples.rl_gsm8k.orchestrate_rl import CoTMathAgent, RLMathTape, extract_tape_training_samples
from tapeagents.finetune.data import load_samples
from tapeagents.io import load_tapes
from tapeagents.observe import retrieve_all_llm_calls

sys.path.append(str(Path(__file__).parent.parent.resolve()))  # allow to import from examples

from examples.data_science import data_science
from examples.delegate import ExampleTape, FindIrregularVerbs
from examples.delegate_stack import (
    ExampleTape as ExampleTapeStack,
)
from examples.delegate_stack import (
    Linguist,
    make_analyze_text_chain,
)
from examples.gaia_agent.agent import GaiaAgent
from examples.gaia_agent.environment import GaiaEnvironment
from examples.gaia_agent.tape import GaiaTape
from examples.gsm8k_tuning.math_agent import MathAgent, MathTape
from examples.llama_agent import LLAMAChatBot
from examples.optimize.optimize import make_agentic_rag_agent, make_env
from examples.tape_improver import tape_improver
from examples.workarena.agent import WorkArenaAgent
from examples.workarena.steps import WorkArenaTape
from tapeagents.config import DB_DEFAULT_FILENAME
from tapeagents.core import AgentStep, TrainingText
from tapeagents.dialog_tape import DialogTape
from tapeagents.environment import EmptyEnvironment
from tapeagents.llms import LLM, ReplayLLM, TrainableLLM
from tapeagents.observe import init_sqlite_if_not_exists, retrieve_tape_llm_calls
from tapeagents.orchestrator import replay_tape, replay_tapes
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

    with tempfile.TemporaryDirectory() as tmp_dir:
        with set_sqlite_db_dir(tmp_dir):
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
            reused_tape, llm_calls = agent.reuse(tape)
            for reused_step, step in zip(reused_tape, tape):
                if isinstance(step, AgentStep):
                    assert isinstance(reused_step, AgentStep)
                    assert reused_step.metadata.prompt_id != step.metadata.prompt_id
    retrieve_tape_llm_calls(reused_tape)
    traces_from_logs = [agent.make_training_text(llm_call) for llm_call in llm_calls]
    direct_traces = agent.make_training_data(tape)
    assert len(traces_from_logs) == len(
        orig_traces
    ), f"Expected {len(orig_traces)} traces from logs, got {len(traces_from_logs)}"
    for trace, orig_trace in zip(traces_from_logs, orig_traces):
        assert trace.prompt_text == orig_trace.prompt_text
        assert trace.output_text == orig_trace.output_text

    assert len(direct_traces) == len(
        orig_traces
    ), f"Expected {len(orig_traces)} direct traces, got {len(direct_traces)}"
    for trace, orig_trace in zip(direct_traces, orig_traces):
        assert trace.prompt_text == orig_trace.prompt_text
        assert trace.output_text == orig_trace.output_text


def test_gaia_agent():
    run_dir = str(res_path / "gaia_agent")
    db_file = f"{run_dir}/tapedata.sqlite"
    with gzip.open(f"{run_dir}/tapedata.sqlite.gz", "rb") as f_in:
        with open(db_file, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    try:
        llm = mock_llm(run_dir)
        env = GaiaEnvironment(only_cached_webpages=True, attachment_dir=f"{run_dir}/attachments")
        env.browser.set_web_cache(f"{run_dir}/web_cache.jsonl")
        agent = GaiaAgent.create(llm)
        tapes = load_tapes(GaiaTape, os.path.join(run_dir, "tapes"), file_extension=".json")
        logger.info(f"Validate {len(tapes)} tapes")
        fails = replay_tapes(agent, tapes, env, reuse_observations=True)
        assert fails == 0, f"{fails} failed tapes"
    finally:
        if os.path.exists(db_file):
            os.remove(db_file)


def test_workarena_agent():
    run_dir = str(res_path / "workarena" / "guided")
    llm = mock_llm(run_dir)
    agent = WorkArenaAgent.create(llm)
    tapes = load_tapes(WorkArenaTape, os.path.join(run_dir, "tapes"), file_extension=".json")
    logger.info(f"Validate {len(tapes)} tapes")
    fails = replay_tapes(agent, tapes, reuse_observations=True)
    assert fails == 0, f"{fails} failed tapes"


def test_delegate():
    run_dir = str(res_path / "delegate")
    llm = mock_llm(run_dir)
    agent = FindIrregularVerbs.create(llm)

    start_tape = ExampleTape.model_validate(load_tape_dict(run_dir, fname="start_tape.json"))
    tape = ExampleTape.model_validate(load_tape_dict(run_dir))

    replay_success = replay_tape(agent, tape, start_tape=start_tape, reuse_observations=True)
    assert replay_success, "Failed to replay tape"


def test_delegate_stack():
    run_dir = str(res_path / "delegate_stack")
    llm = mock_llm(run_dir)
    agent1 = Linguist.create(llm)
    agent2 = make_analyze_text_chain(llm)

    start_tape = ExampleTapeStack.model_validate(load_tape_dict(run_dir, fname="start_tape.json"))
    tape1 = ExampleTapeStack.model_validate(load_tape_dict(run_dir, fname="tape1.json"))
    tape2 = ExampleTapeStack.model_validate(load_tape_dict(run_dir, fname="tape2.json"))

    replay_success = replay_tape(agent1, tape1, start_tape=start_tape, reuse_observations=True)
    assert replay_success, "Failed to replay tape"
    replay_success = replay_tape(agent2, tape2, start_tape=start_tape, reuse_observations=True)
    assert replay_success, "Failed to replay tape"


def test_data_science():
    run_dir = f"{res_path}/data_science"
    llm = mock_llm(run_dir)
    agent, start_tape, env = data_science.make_world(llm, EmptyEnvironment())
    final_tape = TeamTape.model_validate(load_tape_dict(run_dir, "final_tape.json"))
    replay_success = replay_tape(agent, final_tape, start_tape=start_tape, env=env, reuse_observations=True)
    assert replay_success, "Failed to replay tape"


def test_tape_improver():
    run_dir = f"{res_path}/tape_improver"
    llm = mock_llm(run_dir)
    agent, _, improver_tape = tape_improver.make_world(llm)
    final_tape = tape_improver.CodeImproverTape.model_validate(load_tape_dict(run_dir, "final_tape.json"))
    replay_success = replay_tape(agent, final_tape, start_tape=improver_tape, reuse_observations=True)
    assert replay_success, "Failed to replay tape"


def test_optimize():
    with run_test_in_tmp_dir("optimize"):
        with open("config.yaml") as f:
            cfg = DictConfig(yaml.safe_load(f))
        agent = make_agentic_rag_agent(cfg)
        env = make_env()
        tape = DialogTape.model_validate(load_tape_dict(""))
        replay_success = replay_tape(agent, tape, env=env, reuse_observations=True)
        assert replay_success, "Failed to replay tape"


def test_gsm8k_tuning_tapes_generation():
    run_dir = f"{res_path}/gsm8k_tuning"
    llm = mock_llm(run_dir)
    agent = MathAgent.create(llm)
    tapes = load_tapes(MathTape, os.path.join(run_dir, "tapes"), file_extension=".json")
    logger.info(f"Validate {len(tapes)} tapes")
    fails = replay_tapes(agent, tapes, reuse_observations=True)
    assert fails == 0, f"{fails} failed tapes"


def test_gsm8k_tuning_samples_prep():
    run_dir = f"{res_path}/gsm8k_tuning"
    training_samples = load_samples(f"{run_dir}/training_samples.jsonl")
    new_training_samples = get_training_samples_from_tapes(f"{run_dir}/tapes/")
    assert training_samples == new_training_samples


def test_rl_for_math_data():
    run_dir = f"{res_path}/rl_math"
    sqlite_path = f"{run_dir}/tapedata.sqlite"
    llm_calls = retrieve_all_llm_calls(sqlite_path)
    tapes = load_tapes(RLMathTape, run_dir, file_extension=".json")
    agent = CoTMathAgent.create(mock_llm(run_dir))
    cfg = DictConfig({"use_rejection_sampling": False, "finetune": {"seq_length": 1024}})
    training_samples = []
    for tape in tapes:
        _, training_sample, _ = extract_tape_training_samples(tape, agent, "train", cfg, llm_calls)
        training_samples.extend(training_sample)
    new_training_samples = load_samples(f"{run_dir}/training_samples.jsonl")
    assert training_samples == new_training_samples


if __name__ == "__main__":
    test_llama_agent()
    test_llama_agent_traces()
    test_llama_agent_tape_reuse()
    test_gaia_agent()
    test_workarena_agent()
    test_delegate()
    test_delegate_stack()
    test_data_science()
    test_tape_improver()
    test_gsm8k_tuning_tapes_generation()
    test_gsm8k_tuning_samples_prep()
    test_rl_for_math_data()
