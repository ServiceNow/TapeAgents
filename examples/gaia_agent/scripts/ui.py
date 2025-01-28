import datetime
import logging
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from tapeagents.config import ATTACHMENT_DEFAULT_DIR
from tapeagents.io import load_tapes, save_json_tape, save_tape_images
from tapeagents.orchestrator import main_loop
from tapeagents.renderers import to_pretty_str
from tapeagents.renderers.camera_ready_renderer import CameraReadyRenderer
from tapeagents.studio import Studio
from tapeagents.tools.container_executor import init_code_sandbox

from ..agent import GaiaAgent
from ..environment import get_env
from ..steps import GaiaQuestion
from ..tape import GaiaTape

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../../conf",
    config_name="gaia_demo",
)
def main(cfg: DictConfig) -> None:
    os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(cfg.exp_path, "tapedata.sqlite")
    tapes_dir = os.path.join(cfg.exp_path, "tapes")
    os.makedirs(tapes_dir, exist_ok=True)
    images_dir = os.path.join(cfg.exp_path, "attachments", "images")
    os.makedirs(images_dir, exist_ok=True)
    llm = instantiate(cfg.llm)
    init_code_sandbox(cfg.exp_path)
    env = get_env(cfg.exp_path, **cfg.env)
    agent = GaiaAgent.create(llm, actions=env.actions(), **cfg.agent)
    env.chat.wait_for_user_message()
    content = env.chat.messages[-1]["message"]
    env.chat.add_message(role="assistant", msg="Thinking...")
    today_date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    tape = GaiaTape(steps=[GaiaQuestion(content=f"Today is {today_date_str}.\n{content}")])
    try:
        for event in main_loop(agent, tape, env, max_loops=50):
            if partial_tape := (event.agent_tape or event.env_tape):
                tape = partial_tape
            if event.agent_event and event.agent_event.step:
                if event.agent_event.step.kind in ["set_next_node"]:
                    continue
                elif event.agent_event.step.kind == "llm_output_parsing_failure_action":
                    env.chat.add_message(role="assistant", msg="LLM response error, retry")
                elif event.agent_event.step.kind == "plan_thought":
                    env.chat.add_message(role="assistant", msg=f"Plan:\n{to_pretty_str(event.agent_event.step.plan)}")
                elif event.agent_event.step.kind == "facts_survey_thought":
                    msg = f"Given facts:\n{to_pretty_str(event.agent_event.step.given_facts)}\nFacts to look up:\n{to_pretty_str(event.agent_event.step.facts_to_lookup)}"
                    env.chat.add_message(role="assistant", msg=msg)
                elif event.agent_event.step.kind == "reasoning_thought":
                    env.chat.add_message(role="assistant", msg=event.agent_event.step.reasoning)
                elif event.agent_event.step.kind == "reading_result_thought":
                    msg = f'{event.agent_event.step.fact_description}\nSupporting quote: "{event.agent_event.step.quote_with_fact}"'
                    env.chat.add_message(role="assistant", msg=msg)
                elif event.agent_event.step.kind == "gaia_answer_action":
                    msg = (
                        f"Answer: {event.agent_event.step.answer}\n\n{event.agent_event.step.overview}"
                        if event.agent_event.step.success
                        else f"No answer found:\n{event.agent_event.step.overview}"
                    )
                    env.chat.add_message(role="assistant", msg=msg)
                elif event.agent_event.step.kind in ["python_code_action", "search_action", "watch_video_action"]:
                    env.chat.add_message(role="assistant", msg=to_pretty_str(event.agent_event.step.llm_dict()))
                else:
                    env.chat.add_message(role="assistant", msg="Interacting with the browser...")
            elif event.observation and event.observation.kind == "search_results_observation":
                env.chat.add_message(role="user", msg=event.observation.error or to_pretty_str(event.observation.serp))
            elif event.observation and event.observation.kind == "code_execution_result":
                env.chat.add_message(role="user", msg=f"Code execution result: {event.observation.result.output or ''}")
    except Exception as e:
        tape.metadata.error = str(e)
        logger.exception(f"Failed to solve task: {e}")
    env.chat.wait_for_user_message()
    env.close()
    save_json_tape(tape, tapes_dir, "demo1")
    save_tape_images(tape, images_dir)
    logger.info(f"Saved to {tapes_dir}")


if __name__ == "__main__":
    main()
