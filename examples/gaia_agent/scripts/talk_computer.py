"""
Entrypoint for streamlit, see https://docs.streamlit.io/
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from hydra import compose, initialize
from hydra.utils import instantiate

from examples.gaia_agent.agent import GaiaAgent
from examples.gaia_agent.environment import get_computer_env
from examples.gaia_agent.steps import GaiaQuestion
from examples.gaia_agent.tape import GaiaTape
from tapeagents.core import Step
from tapeagents.dialog_tape import UserStep
from tapeagents.orchestrator import main_loop
from tapeagents.steps import ReasoningThought
from tapeagents.tools.computer.computer import Computer
from tapeagents.tools.computer.remote import RemoteComputer

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
    h1 {
        font-size: 2em !important;
    }
</style>
"""


async def main():
    st.markdown(STREAMLIT_STYLE, unsafe_allow_html=True)

    computer = RemoteComputer(computer_url="http://localhost:8000")

    with st.chat_message("assistant"):
        st.write("Language controlled computer activated.")

    if prompt := st.chat_input():
        with st.chat_message("user"):
            st.write(prompt)
        if prompt.startswith("exit"):
            return
        elif prompt.startswith("clear"):
            st.empty()
        elif prompt.startswith("click"):
            target = prompt.split(" ")[1]
            computer.move_and_click(target)
            msg = "Clicked"
        elif prompt.startswith("type"):
            text = prompt.split(" ")[1]
            computer.type_text(text)
            msg = "Typed"

        if msg:
            with st.chat_message("assistant"):
                st.write(msg)


if __name__ == "__main__":
    asyncio.run(main())
