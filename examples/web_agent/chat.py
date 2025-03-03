"""
Entrypoint for streamlit, see https://docs.streamlit.io/
"""

import asyncio
import logging
import os
import traceback
from datetime import datetime
from pathlib import Path

import streamlit as st
from hydra import compose, initialize

from tapeagents.core import Step, Tape
from tapeagents.dialog_tape import AssistantAnswer, UserStep
from tapeagents.io import save_json_tape, save_tape_images
from tapeagents.orchestrator import get_agent_and_env_from_config, main_loop
from tapeagents.steps import ReasoningThought
from tapeagents.tools.computer.steps import MouseHoverAction

# Set up all loggers to print to stdout
logging.basicConfig(
    level=logging.INFO,
    force=True,  # This overwrites any existing logger configurations
    format="%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(funcName)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # Explicitly add stdout handler
    ],
)

CONFIG_DIR = Path(".streamlit_config")
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
    h1 {
        font-size: 2em !important;
    }
</style>
"""

INTERRUPT_TEXT = "(user stopped or interrupted and wrote the following)"
INTERRUPT_TOOL_ERROR = "human stopped or interrupted tool execution"


def setup_state(cfg):
    if "api_key" not in st.session_state:
        # Try to load API key from file first, then environment
        st.session_state.api_key = load_from_storage("api_key") or os.getenv("OPENAI_API_KEY", "")
    else:
        os.environ["OPENAI_API_KEY"] = st.session_state.api_key
    if "serper_api_key" not in st.session_state:
        st.session_state.serper_api_key = load_from_storage("serper_api_key") or os.getenv("SERPER_API_KEY", "")
    else:
        os.environ["SERPER_API_KEY"] = st.session_state.serper_api_key
    if "grounding_api_key" not in st.session_state:
        st.session_state.grounding_api_key = load_from_storage("grounding_api_key") or os.getenv(
            "GROUNDING_API_KEY", ""
        )
    if st.session_state.grounding_api_key:
        os.environ["GROUNDING_API_KEY"] = st.session_state.grounding_api_key
    if "tape" not in st.session_state:
        st.session_state.tape = None
    if "env" not in st.session_state:
        os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(cfg.exp_path, "tapedata.sqlite")
        agent, env = get_agent_and_env_from_config(cfg)
        st.session_state.env = env
        st.session_state.agent = agent
    if st.session_state.tape is None:
        st.session_state.tapes_dir = os.path.join(cfg.exp_path, "tapes")
        st.session_state.tape_name = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        st.session_state.images_dir = os.path.join(cfg.exp_path, "attachments", "images")
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi, TapeAgents Operator here! How can I help you today?"}
        ]
        if not st.session_state.api_key:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Please enter your OpenAI API key in the sidebar to start the conversation.",
                }
            )
        if not st.session_state.serper_api_key:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Please enter your Serper search API key in the sidebar to start the conversation.",
                }
            )


async def main(cfg):
    setup_state(cfg)

    st.markdown(STREAMLIT_STYLE, unsafe_allow_html=True)

    with st.sidebar:
        st.text_input(
            "Openai API Key",
            type="password",
            key="api_key",
            on_change=lambda: save_to_storage("api_key", st.session_state.api_key),
        )
        st.text_input(
            "Serper API Key",
            type="password",
            key="serper_api_key",
            on_change=lambda: save_to_storage("serper_api_key", st.session_state.serper_api_key),
        )
        st.text_input(
            "Grounding API Key",
            type="password",
            key="grounding_api_key",
            on_change=lambda: save_to_storage("grounding_api_key", st.session_state.grounding_api_key),
        )
        if st.button("Reset Conversation"):
            st.session_state.tape = None
            st.session_state.env.reset()
            st.rerun()

    today_date_str = datetime.now().strftime("%Y-%m-%d")
    if st.session_state.tape is None:
        initial_obs = st.session_state.env.step(MouseHoverAction(element_description="center of the screen"))
        st.session_state.tape = Tape(steps=[initial_obs, UserStep(content=f"Today is {today_date_str}")])
    else:
        if isinstance(st.session_state.tape.steps[-1], AssistantAnswer):
            st.session_state.tape.steps[-1] = ReasoningThought(reasoning=st.session_state.tape.steps[-1].answer)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            msg_type = message.get("type", "write")
            if msg_type == "code":
                st.code(message["content"], language="python")
            elif msg_type == "html":
                st.html(message["content"])
            elif msg_type == "markdown":
                st.markdown(message["content"])
            else:
                st.write(message["content"])

    if prompt := st.chat_input():
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        last_status = st.status("Thinking...")
        st.session_state.tape = st.session_state.tape.append(UserStep(content=prompt))

        try:
            for event in main_loop(st.session_state.agent, st.session_state.tape, st.session_state.env, max_loops=50):
                msg = None
                icon = ""
                if partial_tape := (event.agent_tape or event.env_tape):
                    st.session_state.tape = partial_tape
                    save_json_tape(partial_tape, st.session_state.tapes_dir, st.session_state.tape_name)
                    save_tape_images(partial_tape, st.session_state.images_dir)
                if event.agent_event and event.agent_event.step:
                    step = event.agent_event.step
                    if step.kind in ["set_next_node"]:
                        continue
                    msg, msg_type, icon = render_step(step)
                elif event.observation:
                    step = event.observation
                    msg, msg_type, icon = render_step(step)
                elif event.agent_event and event.agent_event.final_tape:
                    st.session_state.tape = event.agent_event.final_tape
                if msg:
                    if last_status is not None:
                        last_status.update(state="complete")
                    last_status = None
                    if msg_type == "progress":
                        last_status = st.status(msg)
                    elif msg_type == "code":
                        with st.chat_message("assistant", avatar=":material/code:"):
                            st.code(msg, language="python")
                    elif msg_type == "html":
                        if icon:
                            with st.chat_message("assistant", avatar=f":material/{icon}:"):
                                st.html(msg)
                        else:
                            with st.chat_message("assistant"):
                                st.html(msg)
                    elif msg_type == "markdown":
                        with st.chat_message("assistant"):
                            st.markdown(msg)
                    else:  # default case for "write"
                        with st.chat_message("assistant"):
                            st.write(msg)
                    st.session_state.messages.append({"role": "assistant", "content": msg, "type": msg_type})

        except Exception as e:
            error_msg = f"Failed to solve task: {e}\nStack trace:\n{traceback.format_exc()}"
            print(error_msg)
            st.session_state.tape.metadata.error = str(e)
            with st.chat_message("assistant"):
                st.write(error_msg)


def render_step(step: Step) -> str:
    msg = ""
    msg_type = "html"
    icon = ""
    if step.kind == "plan_thought":
        steps_html = "\n".join(
            [
                f'<div class="step" style="margin: 0; padding: 0 10px; display: flex;">'
                f'<div style="color: #666; margin-right: 10px;">{i + 1}.</div>'
                f"<div>{step}</div>"
                f"</div>"
                for i, step in enumerate(step.plan)
            ]
        )
        msg = f"""
        <div style="font-family: Arial, sans-serif; background: #f8f9fa; border-radius: 8px; padding: 15px; margin: 10px 0;">
            <div style="font-weight: bold; color: #1a73e8; margin-bottom: 10px;">📋 Plan:</div>
            <div style="margin-left: 10px;">
                {steps_html}
            </div>
        </div>
        """
    elif step.kind == "page_observation":
        msg = "Reading web page..."
        msg_type = "progress"
    elif step.kind == "image":
        if step.error:
            msg = f"<div style='color: red;'>{step.error}</div>"
        else:
            msg = "Looking at the image..."
            msg_type = "progress"
    elif step.kind == "computer_observation":
        if not step.base64_image:
            msg = f"""
            <div style="font-family: 'Monaco', 'Menlo', monospace; font-size: 10pt; margin: 10px 0; border-radius: 8px; overflow: hidden;">
                <div style="padding: 15px; background: #f8f9fa; border: 1px solid #eee;">
                    <div style="white-space: pre-wrap;"><code>{step.output}</code></div>
                    {f'<div style="white-space: pre-wrap; color: #dc3545;"><span style="color: #b71c1c;">Error:</span><br><code>{step.error}</code></div>' if step.error else ''}
                </div>
            </div>
            """
            icon = "output"
        else:
            msg = "Looking at the screen..."
            msg_type = "progress"
    elif step.kind == "facts_survey_thought":
        sections = []
        if step.given_facts:
            facts_html = "\n".join([f'<li style="margin: 4px 0;">{fact}</li>' for fact in step.given_facts])
            sections.append(f"""
                <div class="section">
                    <div style="font-weight: bold; color: #1a73e8;">📌 Given Facts:</div>
                    <ul style="margin: 8px 0; padding-left: 20px;">{facts_html}</ul>
                </div>
            """)
        if step.facts_to_lookup:
            facts_html = "\n".join([f'<li style="margin: 4px 0;">{fact}</li>' for fact in step.facts_to_lookup])
            sections.append(f"""
                <div class="section">
                    <div style="font-weight: bold; color: #1a73e8;">🔎 Facts to Look Up:</div>
                    <ul style="margin: 8px 0; padding-left: 20px;">{facts_html}</ul>
                </div>
            """)
        if step.facts_to_derive:
            facts_html = "\n".join([f'<li style="margin: 4px 0;">{fact}</li>' for fact in step.facts_to_derive])
            sections.append(f"""
                <div class="section">
                    <div style="font-weight: bold; color: #1a73e8;">🔄 Facts to Derive:</div>
                    <ul style="margin: 8px 0; padding-left: 20px;">{facts_html}</ul>
                </div>
            """)
        if step.facts_to_guess:
            facts_html = "\n".join([f'<li style="margin: 4px 0;">{fact}</li>' for fact in step.facts_to_guess])
            sections.append(f"""
                <div class="section">
                    <div style="font-weight: bold; color: #1a73e8;">🎲 Facts to Guess:</div>
                    <ul style="margin: 8px 0; padding-left: 20px;">{facts_html}</ul>
                </div>
            """)

        msg = f"""
        <div style="font-family: Arial, sans-serif; background: #f8f9fa; border-radius: 8px; padding: 15px; margin: 10px 0;">
            {"".join(sections)}
        </div>
        """
    elif step.kind == "reasoning_thought":
        msg = step.reasoning
        msg_type = "markdown"
    elif step.kind == "reading_result_thought":
        msg = f'{step.fact_description}\nSupporting quote: "{step.quote_with_fact}"'
    elif step.kind == "assistant_answer":
        if step.success:
            msg = f"""
            <div style="font-family: Arial, sans-serif; margin: 10px 0;">
                <div style="background: #e7f3ff; border-left: 4px solid #1a73e8; padding: 15px; border-radius: 4px;">
                    <div style="font-weight: bold; color: #1a73e8; font-size: 1.1em; margin-bottom: 10px;">
                        🎯 Answer
                    </div>
                    <div style="color: #202124; line-height: 1.5;">
                        {step.answer}
                    </div>
                </div>
            </div>
            """
        else:
            msg = f"""
            <div style="font-family: Arial, sans-serif; margin: 10px 0;">
                <div style="background: #ffeaea; border-left: 4px solid #dc3545; padding: 15px; border-radius: 4px;">
                    <div style="font-weight: bold; color: #dc3545; font-size: 1.1em; margin-bottom: 10px;">
                        ❌ Could Not Find Answer
                    </div>
                    <div style="color: #202124; line-height: 1.5;">
                        {step.answer}
                    </div>
                </div>
            </div>
            """
    elif step.kind == "search_action":
        source_icon = "🌐" if step.source == "web" else "📚" if step.source == "wiki" else "▶️"
        source_name = "Web" if step.source == "web" else "Wikipedia" if step.source == "wiki" else "YouTube"
        msg = f'<div style="padding: 10px; border-left: 3px solid #ccc; background: #f9f9f9;">{source_icon} Searching <b>{source_name}</b> for:<br/><i>"{step.query}"</i></div>'
    elif step.kind == "python_code_action":
        msg = step.code
        msg_type = "code"
    elif step and step.kind == "page_up_action":
        msg = "Scrolling up..."
        msg_type = "progress"
    elif step.kind == "page_down_action":
        msg = "Scrolling down..."
        msg_type = "progress"
    elif step.kind == "mouse_click_at_action":
        msg = f"Clicking {step.element_description}..."
        msg_type = "progress"
    elif step.kind == "run_terminal_command":
        msg = f"$> {step.command}"
        msg_type = "progress"
    elif step.kind == "mouse_hover_action":
        msg = f"Hovering over {step.element_description}..."
        msg_type = "progress"
    elif step.kind in ["input_text_action", "type_text_action", "key_press_action"]:
        msg = f"Typing '{step.text}'..."
        msg_type = "progress"
    elif step.kind in [
        "go_forward_action",
        "go_back_action",
        "click_action",
        "select_option_action",
    ]:
        msg = "Interacting with the computer..."
        msg_type = "progress"
    elif step.kind == "open_url_action":
        msg = f"Opening {step.url}"
        msg_type = "progress"

    elif step.kind == "search_results_observation":
        if step.error:
            msg = step.error
        else:
            results_html = []
            for r in step.serp:
                results_html.append(f"""
                    <div style="margin: 5px 0; padding: 8px; border: 1px solid #eee; border-radius: 4px; background: white;">
                        <div style="line-height: 1.2;">
                            <a href="{r['url']}" style="color: #1a0dab; text-decoration: none; font-size: 0.95em;">{r['title']}</a>
                            <span style="color: #006621; font-size: 0.8em; margin-left: 8px;">
                                {r['url'][:70]}{'...' if len(r['url']) > 70 else ''}
                            </span>
                        </div>
                        <div style="color: #545454; font-size: 0.85em; margin-top: 2px;">
                            {r['content']}
                        </div>
                    </div>
                """)
            msg = '<div style="font-family: Arial, sans-serif;">' + "\n".join(results_html) + "</div>"
        icon = "output"
    elif step.kind == "code_execution_result":
        result = step.result
        status_color = "#28a745" if result.exit_code == 0 else "#dc3545"  # green for success, red for error
        output = result.output.strip() if result.exit_code == 0 else ""
        error = result.output.strip() if result.exit_code != 0 else ""
        msg = f"""
        <div style="font-family: 'Monaco', 'Menlo', monospace; font-size: 10pt; margin: 10px 0; border-radius: 8px; overflow: hidden;">
            <div style="padding: 8px 15px; background: {status_color}; color: white; font-size: 0.9em;">
                Exit Code: {result.exit_code}
            </div>
            <div style="padding: 15px; background: #f8f9fa; border: 1px solid #eee;">
                <div style="white-space: pre-wrap; margin-bottom: {' 15px' if error else '0'};"><code>{output}</code></div>
                {f'<div style="white-space: pre-wrap; color: #dc3545;"><span style="color: #b71c1c;">Error:</span><br><code>{error}</code></div>' if error else ''}
            </div>
        </div>
        """
        icon = "output"
    elif step.kind == "extracted_facts_thought":
        msg = step.extracted_facts
        msg_type = "write"
    else:
        msg = step.llm_dict()
        msg_type = "write"
    return msg, msg_type, icon


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
        st.warning(f"{filename} saved to {CONFIG_DIR}")
    except Exception as e:
        st.write(f"Debug: Error saving {filename}: {e}")


if __name__ == "__main__":
    with initialize(version_base=None, config_path="../../conf", job_name="web_chat"):
        cfg = compose(config_name="web_agent.yaml")
    asyncio.run(main(cfg))
