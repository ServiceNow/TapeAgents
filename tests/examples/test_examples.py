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
from omegaconf import DictConfig

from tapeagents.config import ATTACHMENT_DEFAULT_DIR, DB_DEFAULT_FILENAME
from tapeagents.core import AgentStep, TrainingText
from tapeagents.dialog_tape import DialogTape
from tapeagents.environment import EmptyEnvironment
from tapeagents.io import load_tapes
from tapeagents.llms import LLM, LiteLLM, ReplayLLM, TrainableLLM
from tapeagents.observe import init_sqlite_if_not_exists, retrieve_tape_llm_calls
from tapeagents.orchestrator import get_agent_and_env_from_config, replay_tape, replay_tapes
from tapeagents.team import TeamTape
from tests.make_test_data import run_test_in_tmp_dir

sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))  # allow to import from examples

from examples.data_science import data_science
from examples.delegate import ExampleTape, FindIrregularVerbs
from examples.delegate_stack import (
    ExampleTape as ExampleTapeStack,
    Linguist,
    make_analyze_text_chain,
)
from examples.form_filler.environment import FormFillerEnvironment
from examples.form_filler.scripts.prepare_test_assets import (
    get_teacher_agent,
    get_user_simulator_agent,
    load_teacher_input_tapes,
    load_teacher_reference_tapes,
    load_user_input_tapes,
    load_user_reference_tapes,
)
from examples.gaia_agent.steps import GaiaTape
from examples.llama_agent import LLAMAChatBot
from examples.optimize.optimize import make_agentic_rag_agent, make_env
from examples.tape_improver import tape_improver
from examples.workarena.agent import WorkArenaAgent
from examples.workarena.steps import WorkArenaTape

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
    run_dir = res_path / "gaia_agent"
    db_file = run_dir / "tapedata.sqlite"
    logger.info(f"Copy {run_dir / 'tapedata.sqlite.gz'} to {db_file}")
    with gzip.open(run_dir / "tapedata.sqlite.gz", "rb") as f_in:
        with open(db_file, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    try:
        os.environ["TAPEAGENTS_CACHE_DIR"] = str(run_dir / "cache")
        with open(run_dir / "config.yaml") as f:
            cfg = DictConfig(yaml.safe_load(f))
        agent, env = get_agent_and_env_from_config(cfg)
        agent.llms["default"] = ReplayLLM.from_llm(agent.llm, run_dir)
        tapes = load_tapes(
            GaiaTape,
            run_dir / "tapes",
            file_extension=".json",
            attachment_dir=str(run_dir / ATTACHMENT_DEFAULT_DIR),
        )
        logger.info(f"Validate {len(tapes)} tapes")
        fails = replay_tapes(agent, tapes, env, reuse_observations=True, stop_on_error=True)
        assert fails == 0, f"{fails} failed tapes"
    finally:
        if os.path.exists(db_file):
            os.remove(db_file)


def test_workarena_agent():
    return
    run_dir = str(res_path / "workarena" / "guided")
    llm = ReplayLLM.from_llm(LiteLLM(model_name="mock"), run_dir)
    agent = WorkArenaAgent.create(llm)
    tapes = load_tapes(WorkArenaTape, os.path.join(run_dir, "tapes"), file_extension=".json")
    logger.info(f"Validate {len(tapes)} tapes")
    fails = replay_tapes(agent, tapes, reuse_observations=True, stop_on_error=True)
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


def test_form_filler():
    os.environ["TAPEAGENTS_MOCK_DATE"] = "2024-12-09"
    assets_dir = str(Path(__file__).parent / "res" / "form_filler")
    forms_path = str(
        Path(__file__).parent.parent.parent / "examples" / "form_filler" / "assets" / "forms" / "train" / "FlyCorp"
    )
    env = FormFillerEnvironment.from_spec(forms_path)

    teacher_agent = get_teacher_agent()
    user_agent = get_user_simulator_agent()
    teacher_mock_llm = ReplayLLM.from_llm(teacher_agent.llms["default"], assets_dir)
    teacher_agent.llms = {"default": teacher_mock_llm}

    user_mock_llm = ReplayLLM.from_llm(user_agent.llms["default"], assets_dir)
    user_agent.llms = {"default": user_mock_llm}

    # teacher_output_tapes, user_output_tapes = get_completions(save_as_references=False)
    teacher_input_tapes = load_teacher_input_tapes()
    user_input_tapes = load_user_input_tapes()
    teacher_reference_tapes = load_teacher_reference_tapes()
    user_reference_tapes = load_user_reference_tapes()

    # patch envspecs
    for tape in teacher_input_tapes + teacher_reference_tapes:
        assert tape.context
        tape.context.env_spec = forms_path
    for tape in user_input_tapes + user_reference_tapes:
        assert tape.context
        assert tape.context.context
        tape.context.context.env_spec = forms_path

    # with set_sqlite_db_dir(assets_dir):
    teacher_failures = replay_tapes(
        teacher_agent, tapes=teacher_reference_tapes, env=env, start_tapes=teacher_input_tapes, reuse_observations=True
    )
    assert teacher_failures == 0, "Failed to replay teacher tapes"

    user_failures = replay_tapes(
        user_agent, tapes=user_reference_tapes, env=env, start_tapes=user_input_tapes, reuse_observations=True
    )
    assert user_failures == 0, "Failed to replay user tapes"


def test_tape_improver():
    run_dir = f"{res_path}/tape_improver"
    llm = mock_llm(run_dir)
    agent, _, improver_tape = tape_improver.make_world(llm)
    final_tape = tape_improver.CodeImproverTape.model_validate(load_tape_dict(run_dir, "final_tape.json"))
    replay_success = replay_tape(agent, final_tape, start_tape=improver_tape, reuse_observations=True)
    assert replay_success, "Failed to replay tape"


def test_optimize_gpt35():
    assets_dir = f"{res_path}/optimize/gpt-3.5-turbo"
    with run_test_in_tmp_dir(assets_dir):
        with open("config.yaml") as f:
            cfg = DictConfig(yaml.safe_load(f))
        agent = make_agentic_rag_agent(cfg)
        mock_llm = ReplayLLM.from_llm(agent.llms["default"], assets_dir)
        agent.llms = {"default": mock_llm}
        env = make_env()
        tape = DialogTape.model_validate(load_tape_dict(""))
        replay_success = replay_tape(agent, tape, env=env, reuse_observations=True)
        assert replay_success, "Failed to replay tape"


def test_optimize_gpt4o_mini():
    assets_dir = f"{res_path}/optimize/gpt-4o-mini"
    with run_test_in_tmp_dir(assets_dir):
        with open("config.yaml") as f:
            cfg = DictConfig(yaml.safe_load(f))
        agent = make_agentic_rag_agent(cfg)
        mock_llm = ReplayLLM.from_llm(agent.llms["default"], assets_dir)
        agent.llms = {"default": mock_llm}
        env = make_env()
        tape = DialogTape.model_validate(load_tape_dict(""))
        replay_success = replay_tape(agent, tape, env=env, reuse_observations=True)
        assert replay_success, "Failed to replay tape"


def test_optimize_llama33_70b():
    assets_dir = f"{res_path}/optimize/llama-3.3-70b-instruct"
    with run_test_in_tmp_dir(assets_dir):
        with open("config.yaml") as f:
            cfg = DictConfig(yaml.safe_load(f))
        agent = make_agentic_rag_agent(cfg)
        mock_llm = ReplayLLM.from_llm(agent.llms["default"], assets_dir)
        agent.llms = {"default": mock_llm}
        env = make_env()
        tape = DialogTape.model_validate(load_tape_dict(""))
        replay_success = replay_tape(agent, tape, env=env, reuse_observations=True)
        assert replay_success, "Failed to replay tape"


if __name__ == "__main__":
    test_llama_agent()
    test_llama_agent_traces()
    test_llama_agent_tape_reuse()
    test_gaia_agent()
    test_workarena_agent()
    test_delegate()
    test_delegate_stack()
    test_data_science()
    test_form_filler()
    test_tape_improver()
    test_optimize_gpt35()
    test_optimize_gpt4o_mini()
    test_optimize_llama33_70b()
