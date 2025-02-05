"""
Entrypoint for streamlit, see https://docs.streamlit.io/
"""

import asyncio

import streamlit as st

from tapeagents.tools.computer.remote import (
    KeyPressAction,
    MouseClickAction,
    OpenUrlAction,
    RemoteComputer,
    TypeTextAction,
)

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
    exp_path = ".talk_computer"
    if "computer" not in st.session_state:
        st.session_state.computer = RemoteComputer(exp_path=exp_path, computer_url="http://localhost:8000")
    computer = st.session_state.computer
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Language controlled computer activated."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input():
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if prompt.startswith("open"):
            st.status("Opening...")
            url = prompt.split(" ", maxsplit=1)[1]
            computer.execute_action(OpenUrlAction(url=url))
            msg = f"Opened {url}"
        elif prompt.startswith("click"):
            st.status("Clicking...")
            target = prompt.split(" ", maxsplit=1)[1]
            computer.execute_action(MouseClickAction(element_description=target))
            msg = f"Clicked '{target}'"
        elif prompt.startswith("type"):
            st.status("Typing...")
            text = prompt.split(" ", maxsplit=1)[1]
            computer.execute_action(TypeTextAction(text=text))
            msg = "Done"
        elif prompt.startswith("input"):
            st.status("Typing...")
            text = prompt.split(" ", maxsplit=1)[1]
            computer.execute_action(TypeTextAction(text=text))
            computer.execute_action(KeyPressAction(text="Return"))
            msg = "Done"
        elif prompt == "up" or prompt == "scroll up":
            st.status("Scrolling...")
            computer.execute_action(KeyPressAction(text="Page_Up"))
            msg = "Moved up"
        elif prompt == "down" or prompt == "scroll":
            st.status("Scrolling...")
            computer.execute_action(KeyPressAction(text="Page_Down"))
            msg = "Moved down"

        if msg:
            with st.chat_message("assistant"):
                st.write(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})


if __name__ == "__main__":
    asyncio.run(main())
