import logging
import os
from typing import Any

from browsergym.core.task import AbstractBrowserTask

from tapeagents.core import LLMOutputParsingFailureAction
from tapeagents.environment import Environment
from tapeagents.steps import ActionExecutionFailure
from tapeagents.tools.browser import Browser
from tapeagents.tools.simple_browser import PageObservation
from tapeagents.utils import FatalError

from .steps import (
    Action,
    FinalAnswerAction,
    ReflectionThought,
    WebTape,
    WebTask,
)

logger = logging.getLogger(__name__)


class WebEnvironment(Environment):
    """
    Web environment for running tasks.
    Translates action steps into gym browser python commands in the form of a string.
    """

    def __init__(self, exp_path: str, headless: bool = True, ax_tree: bool = False, html: bool = True, markdown_html: bool = False) -> None:
        super().__init__()
        os.makedirs(exp_path, exist_ok=True)
        self.browser = Browser(headless=headless, exp_path=exp_path, lazy_env_init=True, axtree=ax_tree, html=html, markdown_html=markdown_html)

    def start_task(self, task_entrypoint: type[AbstractBrowserTask], seed: int = 42) -> tuple[WebTape, dict[str, Any]]:
        task_id = f"browsergym/{task_entrypoint.get_task_id()}"
        info = self.browser.start_task(task_id, seed, wait_for_user_message=False)  # type: ignore
        obs = self.browser.run_browser_action("noop()")
        tape = WebTape(steps=[obs, WebTask(task=info["goal"])])
        return tape, info

    def finish_task(self) -> None:
        try:
            self.browser.close()
        except Exception as e:
            logger.error(f"Failed to properly close task: {e}")

    def validate_task(self, tape: WebTape) -> tuple[bool, dict]:
        answer = tape.steps[-1].text if isinstance(tape.steps[-1], FinalAnswerAction) else "Task finished"
        self.browser._env.unwrapped.chat.add_message(role="assistant", msg=answer)
        assert self.browser._env.unwrapped.task is not None
        try:
            reward, stop, message, info = self.browser._env.unwrapped.task.validate(
                self.browser._env.unwrapped.page, self.browser._env.unwrapped.chat.messages
            )
        except Exception as e:
            logger.exception(f"Error during task validation: {e}")
            reward = 0
            stop = True
            message = f"Task validation failed with error: {e}"
            info = {}
        result_dict = {
            "reward": reward,
            "stop": stop,
            "message": message,
            "info": info,
        }
        return bool(reward > 0), result_dict

    def react(self, tape: WebTape) -> WebTape:
        actions = []
        for step in tape.steps[-tape.metadata.n_added_steps :]:
            if isinstance(step, Action):
                actions.append(step)
            elif isinstance(step, ReflectionThought):
                # send reflection to chat for user to see
                self.browser._env.unwrapped.chat.add_message(
                    role="assistant", msg=f"{step.last_action_achieved_effect}\nTodo: {step.next_action}"
                )
        for action in actions:
            try:
                if isinstance(action, LLMOutputParsingFailureAction):
                    continue
                observation = self.browser.run(action)
                if isinstance(observation, ActionExecutionFailure):
                    logger.exception(f"Error during action execution: {observation.error}")
                    tape = tape.append(observation)
                    break
                assert isinstance(observation, PageObservation), f"Observation is not a PageObservation: {observation}"
                tape = tape.append(observation)  # type: ignore
            except FatalError:
                raise
            except Exception as e:
                logger.exception(f"Error during action execution: {e}")
                tape = tape.append(ActionExecutionFailure(error=str(e)))
                break
        return tape
