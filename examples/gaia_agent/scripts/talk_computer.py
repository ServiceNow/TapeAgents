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
    print("Initializing computer")
    computer = RemoteComputer(exp_path=exp_path, computer_url="http://localhost:8000")
    print("Computer initialized")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    print(f"Found {len(st.session_state.messages)} messages in session state")

    with st.chat_message("assistant"):
        st.write("Language controlled computer activated.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input():
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if prompt.startswith("open"):
            url = prompt.split(" ")[1]
            computer.execute_action(OpenUrlAction(url=url))
            msg = f"Opened {url}"
        elif prompt.startswith("click"):
            target = prompt.split(" ")[1]
            computer.execute_action(MouseClickAction(element_description=target))
            msg = "Clicked"
        elif prompt.startswith("type"):
            text = prompt.split(" ")[1]
            computer.execute_action(TypeTextAction(text=text))
            msg = "Typed"
        elif prompt.startswith("input"):
            text = prompt.split(" ")[1]
            computer.execute_action(TypeTextAction(text=text))
            computer.execute_action(KeyPressAction(text="Return"))
            msg = "Done"
        elif prompt == "up":
            computer.execute_action(KeyPressAction(text="Page_Up"))
            msg = "Moved up"
        elif prompt == "down":
            computer.execute_action(KeyPressAction(text="Page_Down"))
            msg = "Moved down"

        if msg:
            with st.chat_message("assistant"):
                st.write(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})


if __name__ == "__main__":
    asyncio.run(main())
