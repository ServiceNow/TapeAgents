import logging
import os
import time
from typing import Any, Literal

from browsergym.core.task import AbstractBrowserTask
from browsergym.miniwob.base import AbstractMiniwobTask
from joblib import Parallel, delayed

from tapeagents.core import Action, LLMOutputParsingFailureAction
from tapeagents.environment import Environment
from tapeagents.steps import ActionExecutionFailure
from tapeagents.tools.browser import Browser
from tapeagents.tools.simple_browser import PageObservation
from tapeagents.utils import FatalError

from .steps import (
    FinalAnswerAction,
    ReflectionThought,
    WebTape,
    WebTapeMetadata,
    WebTask,
)

logger = logging.getLogger(__name__)


class WebEnvironment(Environment):
    """
    Web environment for running tasks.
    Translates action steps into gym browser python commands in the form of a string.
    """

    def __init__(
        self,
        exp_path: str | None,
        headless: bool = True,
        observation_format: Literal["axtree", "html", "markdown_html"] = "html",
    ) -> None:
        super().__init__()
        if exp_path:
            os.makedirs(exp_path, exist_ok=True)
        self.browser = Browser(headless=headless, exp_path=exp_path, mock=True, observation_format=observation_format)
        self.timers = {}  # keep track of time taken in each method

    def start_task(self, task_entrypoint: type[AbstractBrowserTask], seed: int = 42) -> tuple[WebTape, dict[str, Any]]:
        self.timers = {}  # reset timers
        start_start_task = time.perf_counter()
        task_id = f"browsergym/{task_entrypoint.get_task_id()}"
        # logger.info(f"WebEnv.start_task {task_id} starting task...")
        # _zero = time.perf_counter()
        info = self.browser.start_task(task_id, seed, wait_for_user_message=False)  # type: ignore
        # zero = time.perf_counter() - _zero
        # logger.info(f"WebEnv.start_task {task_id} browser.start_task took {zero:.2f}s")
        # _one = time.perf_counter()
        obs = self.browser.run_browser_action("noop()")
        # one = time.perf_counter() - _one
        # logger.info(f"WebEnv.start_task {task_id} browser.run_browser_action took {one:.2f}s")
        # _two = time.perf_counter()
        tape = WebTape(
            metadata=WebTapeMetadata(task_name=task_entrypoint.get_task_id(), seed=seed),
            steps=[obs, WebTask(task=info["goal"])],
        )
        # two = time.perf_counter() - _two
        # logger.info(f"WebEnv.start_task {task_id} WebTape creation took {two:.2f}s")
        self.timers["start_task"] = time.perf_counter() - start_start_task
        return tape, info

    def finish_task(self) -> None:
        start_finish_task = time.perf_counter()
        try:
            self.browser.close()
        except Exception as e:
            logger.exception(f"Failed to properly close task: {e}", stack_info=True)
        self.timers["finish_task"] = time.perf_counter() - start_finish_task

    def validate_task(self, tape: WebTape) -> tuple[bool, dict]:
        start_validate_task = time.perf_counter()
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
        # in AbstractMiniwobTask.validate() the reward is defined as float(info["RAW_REWARD_GLOBAL"] > 0) so it will alwayd be 0 or 1
        # let's keep the raw reward for now and define success as reward > 0.5
        if isinstance(self.browser._env.unwrapped.task, AbstractMiniwobTask) and "RAW_REWARD_GLOBAL" in info:
            reward = info["RAW_REWARD_GLOBAL"]
        result_dict = {
            "reward": reward,
            "stop": stop,
            "message": message,
            "info": info,
        }
        self.timers["validate_task"] = time.perf_counter() - start_validate_task
        return bool(reward > 0.5), result_dict

    def react(self, tape: WebTape) -> WebTape:
        if "react" not in self.timers:
            self.timers["react"] = []
        start_react = time.perf_counter()

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
                self.timers["react"].append(time.perf_counter() - start_react)
                raise
            except Exception as e:
                logger.exception(f"Error during action execution {action}: {e}", stack_info=True)
                tape = tape.append(ActionExecutionFailure(error=str(e)))
                break
        self.timers["react"].append(time.perf_counter() - start_react)
        return tape
        # TODO: MAYBE make sure to update parent_id, author_name, etc... in the new tape.metadata just like in agent.run()

    def react_batch(self, tapes: list[WebTape], n_processes: int) -> list[WebTape]:
        results = Parallel(n_jobs=n_processes)([delayed(self.react)(tape) for tape in tapes])
        return results
