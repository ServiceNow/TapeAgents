"""
Entrypoint for streamlit, see https://docs.streamlit.io/
"""

import asyncio
import base64
import os
import subprocess
import traceback
from contextlib import contextmanager
from datetime import datetime, timedelta
from enum import StrEnum
from functools import partial
from pathlib import PosixPath
from typing import cast
from examples.gaia_agent.environment import get_env
from examples.gaia_agent.agent import GaiaAgent
from examples.gaia_agent.steps import GaiaQuestion
from examples.gaia_agent.tape import GaiaTape
import httpx
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from tapeagents.dialog_tape import UserStep
from tapeagents.orchestrator import main_loop
from tapeagents.renderers import to_pretty_str
from tapeagents.steps import ReasoningThought
from omegaconf import OmegaConf
import hydra
from hydra.utils import instantiate

CONFIG_DIR = PosixPath("~/.anthropic").expanduser()
API_KEY_FILE = CONFIG_DIR / "api_key"
STREAMLIT_STYLE = """
<style>
    /* Highlight the stop button in red */
    button[kind=header] {
        background-color: rgb(255, 75, 75);
        border: 1px solid rgb(255, 75, 75);
        color: rgb(255, 255, 255);
    }
    button[kind=header]:hover {
        background-color: rgb(255, 51, 51);
    }
     /* Hide the streamlit deploy button */
    .stAppDeployButton {
        visibility: hidden;
    }
</style>
"""

WARNING_TEXT = "⚠️ Security Alert: Never provide access to sensitive accounts or data, as malicious web content can hijack Claude's behavior"
INTERRUPT_TEXT = "(user stopped or interrupted and wrote the following)"
INTERRUPT_TOOL_ERROR = "human stopped or interrupted tool execution"


class Sender(StrEnum):
    USER = "user"
    BOT = "assistant"
    TOOL = "tool"


def setup_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "api_key" not in st.session_state:
        # Try to load API key from file first, then environment
        st.session_state.api_key = load_from_storage("api_key") or os.getenv("OPENAI_API_KEY", "")
    if "responses" not in st.session_state:
        st.session_state.responses = {}
    if "tools" not in st.session_state:
        st.session_state.tools = {}
    if "only_n_most_recent_images" not in st.session_state:
        st.session_state.only_n_most_recent_images = 3
    if "custom_system_prompt" not in st.session_state:
        st.session_state.custom_system_prompt = load_from_storage("system_prompt") or ""
    if "hide_images" not in st.session_state:
        st.session_state.hide_images = False
    if "in_sampling_loop" not in st.session_state:
        st.session_state.in_sampling_loop = False
    if "tape" not in st.session_state:
        st.session_state.tape = None


async def main():
    """Render loop for streamlit"""
    setup_state()

    st.markdown(STREAMLIT_STYLE, unsafe_allow_html=True)

    st.title("TapeAgents Operator Demo")

    if not os.getenv("HIDE_WARNING", False):
        st.warning(WARNING_TEXT)

    with st.sidebar:
        st.text_input("Model", key="model")

        st.text_input(
            "Openai API Key",
            type="password",
            key="api_key",
            on_change=lambda: save_to_storage("api_key", st.session_state.api_key),
        )

    # Load config
    cfg = OmegaConf.load("conf/gaia_demo.yaml")
    
    # Initialize environment and agent
    playwright_dir = ".pw-browsers"
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = playwright_dir
    os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(cfg.exp_path, "tapedata.sqlite")

    llm = instantiate(cfg.llm)


    env = get_env(cfg.exp_path, **cfg.env)
    agent = GaiaAgent.create(llm, actions=env.actions(), **cfg.agent)

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "TapeAgent Ready"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if prompt.lower() == "reset":
            st.session_state.tape = None
            st.session_state.messages.append({"role": "assistant", "content": "Reset conversation, you can ask a new question now."})
            st.rerun()

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.write("Thinking...")

            today_date_str = datetime.now().strftime("%Y-%m-%d")
            
            if st.session_state.tape is None:
                st.session_state.tape = GaiaTape(steps=[GaiaQuestion(content=f"Today is {today_date_str}.\n{prompt}")])
            else:
                st.session_state.tape.steps[-1] = ReasoningThought(reasoning=st.session_state.tape.steps[-1].long_answer)
                st.session_state.tape = st.session_state.tape.append(UserStep(content=prompt))

            try:
                for event in main_loop(agent, st.session_state.tape, env, max_loops=50):
                    if partial_tape := (event.agent_tape or event.env_tape):
                        st.session_state.tape = partial_tape
                    if event.agent_event and event.agent_event.step:
                        step = event.agent_event.step
                        if step.kind in ["set_next_node"]:
                            continue
                        msg = render_step(step)
                        if msg:
                            message_placeholder.write(msg)
                            st.session_state.messages.append({"role": "assistant", "content": msg})
                    elif event.observation:
                        step = event.observation
                        msg = render_step(step)
                        if msg:
                            message_placeholder.write(msg)
                            st.session_state.messages.append({"role": "assistant" if step.kind == "page_observation" else "user", "content": msg})
            except Exception as e:
                error_msg = f"Failed to solve task: {e}"
                st.session_state.tape.metadata.error = str(e)
                message_placeholder.write(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

def render_step(step):
    # Copy the render_step function from ui.py
    # ...existing code from ui.py render_step function...

def load_from_storage(filename: str) -> str | None:
    """Load data from a file in the storage directory."""
    try:
        file_path = CONFIG_DIR / filename
        if file_path.exists():
            data = file_path.read_text().strip()
            if data:
                return data
    except Exception as e:
        st.write(f"Debug: Error loading {filename}: {e}")
    return None


def save_to_storage(filename: str, data: str) -> None:
    """Save data to a file in the storage directory."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        file_path = CONFIG_DIR / filename
        file_path.write_text(data)
        # Ensure only user can read/write the file
        file_path.chmod(0o600)
    except Exception as e:
        st.write(f"Debug: Error saving {filename}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
