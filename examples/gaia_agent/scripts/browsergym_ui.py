import datetime
import getpass
import logging
import os

import hydra
from omegaconf import DictConfig

from examples.gaia_agent.steps import GaiaQuestion, GaiaTape
from tapeagents.core import Action, Observation, Step
from tapeagents.dialog_tape import UserStep
from tapeagents.io import save_json_tape, save_tape_images
from tapeagents.orchestrator import get_agent_and_env_from_config, main_loop
from tapeagents.renderers import to_pretty_str
from tapeagents.steps import ReasoningThought
from tapeagents.tools.container_executor import init_code_sandbox

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../../conf",
    config_name="gaia_demo",
)
def main(cfg: DictConfig) -> None:
    playwright_dir = ".pw-browsers"
    assert os.path.exists(playwright_dir), f"playwright browsers directory not found: {playwright_dir}"
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = playwright_dir
    os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(cfg.exp_path, "tapedata.sqlite")
    if not os.environ.get("OPENAI_API_KEY"):
        oai_key = getpass.getpass(
            "Please enter your OpenAI API key (you can get one on https://platform.openai.com/settings/organization/api-keys) and press Enter to continue: "
        )
        os.environ["OPENAI_API_KEY"] = oai_key
    if not os.environ.get("SERPER_API_KEY"):
        serper_key = getpass.getpass(
            "Please enter your Serper API key (you can get one on https://serper.dev/api-key) and press Enter to continue: "
        )
        os.environ["SERPER_API_KEY"] = serper_key
    tapes_dir = os.path.join(cfg.exp_path, "tapes")
    os.makedirs(tapes_dir, exist_ok=True)
    images_dir = os.path.join(cfg.exp_path, "attachments", "images")
    os.makedirs(images_dir, exist_ok=True)
    init_code_sandbox(cfg.exp_path)
    agent, env = get_agent_and_env_from_config(cfg)
    chat = [t for t in env.tools if t.__class__.__name__ == "Browser"][0]._env.chat
    chat.add_message(role="assistant", msg="TapeAgent Ready")
    today_date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    tape = None
    chat.wait_for_user_message()
    content = chat.messages[-1]["message"]
    while content.lower() != "stop":
        if content.lower() == "reset":
            tape = None
            chat.add_message(role="assistant", msg="Reset conversation, you can ask a new question now.")
            chat.wait_for_user_message()
            content = chat.messages[-1]["message"]
        if tape is None:
            tape = GaiaTape(steps=[GaiaQuestion(content=f"Today is {today_date_str}.\n{content}")])
        else:
            # continue the conversation, replace the last answer step with a reasoning step
            tape.steps[-1] = ReasoningThought(reasoning=tape.steps[-1].long_answer)
            tape = tape.append(UserStep(content=content))
        chat.add_message(role="assistant", msg="Thinking...")
        try:
            for event in main_loop(agent, tape, env, max_loops=50):
                if partial_tape := (event.agent_tape or event.env_tape):
                    tape = partial_tape
                if event.agent_event and event.agent_event.step:
                    step = event.agent_event.step
                    if step.kind in ["set_next_node"]:
                        continue
                    msg = render_step(step)
                    if msg:
                        chat.add_message(role="assistant", msg=msg)
                elif event.observation:
                    step = event.observation
                    msg = render_step(step)
                    if msg:
                        chat.add_message(role="assistant" if step.kind == "page_observation" else "user", msg=msg)
        except Exception as e:
            tape.metadata.error = str(e)
            logger.exception(f"Failed to solve task: {e}")
            chat.add_message(role="assistant", msg=f"Failed to solve task: {e}")
        chat.wait_for_user_message()
        content = chat.messages[-1]["message"]
    env.close()
    save_json_tape(tape, tapes_dir, "demo1")
    save_tape_images(tape, images_dir)
    logger.info(f"Saved to {tapes_dir}")


def render_step(step: Step) -> str:
    logger.info(step.short_view() if isinstance(step, Observation) else step.llm_view())
    msg = ""
    if step.kind == "llm_output_parsing_failure_action":
        msg = "LLM response error, retry"
    elif step.kind == "plan_thought":
        msg = f"Plan:\n{to_pretty_str(step.plan)}"
    elif step.kind == "page_observation":
        msg = "Reading web page..."
    elif step.kind == "facts_survey_thought":
        msg = f"Given facts:\n{to_pretty_str(step.given_facts)}"
        if len(step.facts_to_lookup) > 0:
            msg += f"\nFacts to look up:\n{to_pretty_str(step.facts_to_lookup)}"
    elif step.kind == "reasoning_thought":
        msg = step.reasoning
    elif step.kind == "reading_result_thought":
        msg = f'{step.fact_description}\nSupporting quote: "{step.quote_with_fact}"'
    elif step.kind == "gaia_answer_action":
        if step.success:
            msg = f"Answer: {step.long_answer}"
        else:
            msg = f"No answer found:\n{step.overview}"
    elif step.kind in ["python_code_action", "search_action", "watch_video_action"]:
        msg = to_pretty_str(step.llm_dict())
    elif step and step.kind == "scroll_action":
        msg = "Scrolling..."
    elif isinstance(step, Action):
        msg = "Interacting with the browser..."
    elif step and step.kind == "search_results_observation":
        msg = step.error or to_pretty_str([f"[{r['url'][:30]}...]{r['title'][:60]}..." for r in step.serp])
    elif step and step.kind == "code_execution_result":
        msg = f"Code execution result: {step.result.output or ''}"
    return msg


if __name__ == "__main__":
    main()
