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


def setup_state():
    if "api_key" not in st.session_state:
        # Try to load API key from file first, then environment
        st.session_state.api_key = load_from_storage("api_key") or os.getenv("OPENAI_API_KEY", "")
    if "only_n_most_recent_images" not in st.session_state:
        st.session_state.only_n_most_recent_images = 3
    if "tape" not in st.session_state:
        st.session_state.tape = None


async def main():
    with initialize(version_base=None, config_path="../../../conf", job_name="computer_demo"):
        cfg = compose(config_name="gaia_demo")
    setup_state()

    st.markdown(STREAMLIT_STYLE, unsafe_allow_html=True)

    with st.sidebar:
        st.text_input(
            "Openai API Key",
            type="password",
            key="api_key",
            on_change=lambda: save_to_storage("api_key", st.session_state.api_key),
        )
        if st.button("Reset Conversation"):
            st.session_state.tape = None
            st.rerun()

    # Initialize environment and agent
    playwright_dir = ".pw-browsers"
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = playwright_dir
    os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(cfg.exp_path, "tapedata.sqlite")

    llm = instantiate(cfg.llm)
    env = get_computer_env(cfg.exp_path, **cfg.env)
    agent = GaiaAgent.create(llm, actions=env.actions(), **cfg.agent)

    with st.chat_message("assistant"):
        st.write("Hi, TapeAgents Operator here! How can I help you today?")

    if prompt := st.chat_input():
        with st.chat_message("user"):
            st.write(prompt)

        today_date_str = datetime.now().strftime("%Y-%m-%d")

        if st.session_state.tape is None:
            st.session_state.tape = GaiaTape(steps=[GaiaQuestion(content=f"Today is {today_date_str}.\n{prompt}")])
        else:
            st.session_state.tape.steps[-1] = ReasoningThought(reasoning=st.session_state.tape.steps[-1].long_answer)
            st.session_state.tape = st.session_state.tape.append(UserStep(content=prompt))

        try:
            for event in main_loop(agent, st.session_state.tape, env, max_loops=50):
                msg = None
                if partial_tape := (event.agent_tape or event.env_tape):
                    st.session_state.tape = partial_tape
                if event.agent_event and event.agent_event.step:
                    step = event.agent_event.step
                    if step.kind in ["set_next_node"]:
                        continue
                    msg, msg_type = render_step(step)
                elif event.observation:
                    step = event.observation
                    msg, msg_type = render_step(step)
                if msg:
                    if msg_type == "progress":
                        st.spinner(msg)
                    elif msg_type == "code":
                        with st.chat_message("assistant", avatar="💻"):
                            st.code(msg, language="python")
                    elif msg_type == "html":
                        with st.chat_message("assistant"):
                            st.html(msg)
                    else:  # default case for "write"
                        with st.chat_message("assistant"):
                            st.write(msg)

        except Exception as e:
            error_msg = f"Failed to solve task: {e}"
            st.session_state.tape.metadata.error = str(e)
            with st.chat_message("assistant"):
                st.write(error_msg)


def render_step(step: Step) -> str:
    msg = ""
    msg_type = "html"
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
    elif step.kind == "reading_result_thought":
        msg = f'{step.fact_description}\nSupporting quote: "{step.quote_with_fact}"'
    elif step.kind == "gaia_answer_action":
        if step.success:
            msg = f"""
            <div style="font-family: Arial, sans-serif; margin: 10px 0;">
                <div style="background: #e7f3ff; border-left: 4px solid #1a73e8; padding: 15px; border-radius: 4px;">
                    <div style="font-weight: bold; color: #1a73e8; font-size: 1.1em; margin-bottom: 10px;">
                        🎯 Final Answer
                    </div>
                    <div style="color: #202124; line-height: 1.5;">
                        {step.long_answer}
                    </div>
                    {f'<div style="margin-top: 8px; color: #666; font-size: 0.9em;">Raw answer: {step.answer}</div>' if str(step.answer) != step.long_answer else ''}
                    {f'<div style="margin-top: 4px; color: #666; font-size: 0.9em;">Unit: {step.answer_unit}</div>' if step.answer_unit else ''}
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
                        {step.overview}
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
    elif step and step.kind == "page_up_action" or step.kind == "page_down_action":
        msg = "Scrolling..."
        msg_type = "progress"
    elif step.kind in [
        "mouse_click_action",
        "mouse_hover_action",
        "open_url_action",
        "input_text_action",
        "type_text_action",
        "click_action",
        "select_option_action",
        "go_forward_action",
        "go_back_action",
    ]:
        msg = "Interacting with the browser..."
        msg_type = "progress"
    elif step and step.kind == "search_results_observation":
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
    elif step and step.kind == "code_execution_result":
        result = step.result
        status_color = "#28a745" if result.exit_code == 0 else "#dc3545"  # green for success, red for error
        output = result.output.strip() if result.exit_code == 0 else ""
        error = result.output.strip() if result.exit_code != 0 else ""

        msg = f"""
        <div style="font-family: 'Monaco', 'Menlo', monospace; margin: 10px 0; border-radius: 8px; overflow: hidden;">
            <div style="padding: 8px 15px; background: {status_color}; color: white; font-size: 0.9em;">
                Exit Code: {result.exit_code}
            </div>
            <div style="padding: 15px; background: #f8f9fa; border: 1px solid #eee;">
                <div style="white-space: pre-wrap; margin-bottom: {' 15px' if error else '0'};"><code>{output}</code></div>
                {f'<div style="white-space: pre-wrap; color: #dc3545;"><span style="color: #b71c1c;">Error:</span><br><code>{error}</code></div>' if error else ''}
            </div>
        </div>
        """
    else:
        msg = step.llm_dict()
        msg_type = "write"
    return msg, msg_type


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
