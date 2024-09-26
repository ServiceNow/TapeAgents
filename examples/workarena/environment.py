import logging
from time import sleep
from typing import Any

from browsergym.workarena.tasks.base import AbstractServiceNowTask

from tapeagents.core import AgentResponseParsingFailureAction
from tapeagents.environment import Environment
from tapeagents.tools.gym_browser import GymBrowser
from tapeagents.utils import FatalError

from ..gaia_agent.steps import ActionExecutionFailure
from .steps import (
    Action,
    ClickAction,
    CloseTabAction,
    FinalAnswerAction,
    GoBackAction,
    GoForwardAction,
    GotoPageAction,
    HoverAction,
    InputTextAction,
    NewTabAction,
    PageObservation,
    PressAction,
    ReflectionThought,
    ScrollAction,
    SelectOptionAction,
    TabFocusAction,
    WorkArenaTape,
    WorkArenaTask,
)

logger = logging.getLogger(__name__)


class WorkArenaEnvironment(Environment):
    """
    WorkArena environment for running tasks.
    Translates action steps into gym browser python commands in the form of a string.
    """

    def __init__(self, exp_path: str, baseline_obs: bool = False, headless: bool = True) -> None:
        super().__init__()
        self.baseline_obs = baseline_obs  # use baseline observation format to replicate original workarena agent
        self.browser = GymBrowser(headless=headless, log_path=exp_path)
        self.action_map = {
            ClickAction: self.click,
            SelectOptionAction: self.select_option,
            CloseTabAction: self.close_tab,
            InputTextAction: self.input_text,
            GoBackAction: self.go_back,
            GoForwardAction: self.go_forward,
            GotoPageAction: self.goto_page,
            HoverAction: self.hover,
            NewTabAction: self.new_tab,
            PressAction: self.press,
            ScrollAction: self.scroll,
            TabFocusAction: self.tab_focus,
        }

    def start_task(
        self, task_entrypoint: type[AbstractServiceNowTask], seed: int = 42
    ) -> tuple[WorkArenaTape, dict[str, Any]]:
        info = self.browser.start_task(task_entrypoint, seed)
        sleep(5)  # wait for the page to load
        text, screen, _, _ = self.browser.perform_action("noop()", self.baseline_obs)
        obs = PageObservation(
            text=text,
            current_page=self.browser.current_viewport,
            total_pages=self.browser.n_viewports,
        )
        obs.metadata.other["screenshot_path"] = screen
        tape = WorkArenaTape(steps=[obs, WorkArenaTask(task=info["goal"])])
        return tape, info

    def finish_task(self, task_name: str) -> None:
        self.browser.close(task_name)

    def validate_task(self, tape: WorkArenaTape) -> tuple[bool, dict]:
        answer = tape.steps[-1].text if isinstance(tape.steps[-1], FinalAnswerAction) else "Task finished"
        self.browser.env.chat.add_message(role="assistant", msg=answer)
        assert self.browser.env.task is not None
        reward, stop, message, info = self.browser.env.task.validate(
            self.browser.env.page, self.browser.env.chat.messages
        )
        result_dict = {
            "reward": reward,
            "stop": stop,
            "message": message,
            "info": info,
        }
        return bool(reward > 0), result_dict

    def react(self, tape: WorkArenaTape) -> WorkArenaTape:
        actions = []
        for step in tape.steps[-tape.metadata.n_added_steps :]:
            if isinstance(step, Action):
                actions.append(step)
            elif isinstance(step, ReflectionThought):
                # send reflection to chat for user to see
                self.browser.env.chat.add_message(
                    role="assistant", msg=f"{step.last_action_achieved_effect}\nTodo: {step.next_action}"
                )
        for action in actions:
            try:
                action_type = type(action)
                if action_type == AgentResponseParsingFailureAction:
                    continue
                elif action_type not in self.action_map:
                    raise Exception(f"Unknown action: {action_type}")
                observation = self.action_map[action_type](action)
                tape = tape.append(observation)
            except FatalError:
                raise
            except Exception as e:
                logger.exception(f"Error during action execution: {e}")
                tape = tape.append(ActionExecutionFailure(error=str(e)))
                break
        return tape

    def perform_browser_action(self, action: str) -> PageObservation:
        text, screen, last_action_error, finished = self.browser.perform_action(action, self.baseline_obs)
        obs = PageObservation(
            text=text,
            current_page=self.browser.current_viewport,
            total_pages=self.browser.n_viewports,
            last_action_error=last_action_error,
        )
        obs.metadata.other["screenshot_path"] = screen
        obs.metadata.other["env_finished"] = finished
        return obs

    def goto_page(self, action: GotoPageAction) -> PageObservation:
        return self.perform_browser_action(f"goto('{action.url}')")

    def click(self, action: ClickAction) -> PageObservation:
        self.perform_browser_action(f"click('{action.bid}', button='{action.button}', modifiers={action.modifiers})")
        sleep(1)  # wait for the page to load in case click triggers a page changeg
        return self.perform_browser_action("noop()")

    def select_option(self, action: SelectOptionAction) -> PageObservation:
        return self.perform_browser_action(f"select_option('{action.bid}', '{action.option}')")

    def hover(self, action: HoverAction) -> PageObservation:
        return self.perform_browser_action(f"hover('{action.bid}')")

    def input_text(self, action: InputTextAction) -> PageObservation:
        text = action.text.replace("'", "\\'")
        return self.perform_browser_action(f"fill('{action.bid}', '{text}')")

    def press(self, action: PressAction) -> PageObservation:
        return self.perform_browser_action(f"press('{action.bid}', '{action.key_comb}')")

    def scroll(self, action: ScrollAction) -> PageObservation:
        return PageObservation(
            text=self.browser.scroll(action.direction),
            current_page=self.browser.current_viewport,
            total_pages=self.browser.n_viewports,
        )

    def tab_focus(self, action: TabFocusAction) -> PageObservation:
        return self.perform_browser_action(f"tab_focus({action.index})")

    def new_tab(self, action: NewTabAction) -> PageObservation:
        return self.perform_browser_action("new_tab()")

    def close_tab(self, action: CloseTabAction) -> PageObservation:
        return self.perform_browser_action("tab_close()")

    def go_back(self, action: GoBackAction) -> PageObservation:
        return self.perform_browser_action("go_back()")

    def go_forward(self, action: GoForwardAction) -> PageObservation:
        return self.perform_browser_action("go_forward()")
