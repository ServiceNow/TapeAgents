import base64
import logging
import os
import time
from typing import Literal

import requests
from PIL import Image
from pydantic import Field

from tapeagents.core import Action
from tapeagents.steps import ImageObservation
from tapeagents.tools.base import Multitool
from tapeagents.tools.grounding import GroundingModel

from .steps import (
    ComputerObservation,
    GetCursorPositionAction,
    KeyPressAction,
    MouseClickAction as MouseXYClickAction,
    MouseMoveAction as MouseXYMoveAction,
    OpenUrlAction,
    RunTerminalCommand,
    TypeTextAction,
)

logger = logging.getLogger("remote")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(funcName)s - %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class MouseClickAction(Action):
    """
    Action that clicks an element on the computer screen.
    When mentioning a date in the element description, use the format commonly spoken or written by humans,
    such as "2 February 2025," rather than machine-readable formats. The day should come before the month,
    and the year should be written in full (e.g., "3 November 2023" instead of "2023-11-03").
    Only describe one specific element that is currently visible on the screen!
    """

    kind: Literal["mouse_click_action"] = "mouse_click_action"
    button: Literal["left", "double_left", "right"] = Field(
        description="mouse button to click, double_left for double left click", default="left"
    )
    element_description: str = Field(description="brief description of the element to click")


class MouseHoverAction(Action):
    """
    Action that hovers over an icon or control on the computer screen
    """

    kind: Literal["mouse_hover_action"] = "mouse_hover_action"
    element_description: str = Field(description="brief description of the element to hover over")


class PageDownAction(Action):
    """
    Action that scrolls down to display the next page of the current view.
    """

    kind: Literal["page_down_action"] = "page_down_action"


class PageUpAction(Action):
    """
    Action that scrolls up to display the previous page of the current view.
    """

    kind: Literal["page_up_action"] = "page_up_action"


class RemoteComputer(Multitool):
    exp_path: str | None = None
    actions: tuple[type[Action], ...] = ()
    observations: tuple[type[ImageObservation], ...] = (ImageObservation,)
    computer_url: str = Field(description="Remote tool API URL")
    use_grounding: bool = Field(description="Whether to use grounding model", default=True)
    grounding_api_url: str = Field(description="Grounding API URL")

    def model_post_init(self, __context):
        self._grounding = GroundingModel(url=self.grounding_api_url)
        self._screenshot_dir = f"{self.exp_path}/attachments/remote_screenshots/"
        os.makedirs(self._screenshot_dir, exist_ok=True)
        self._action_map = {
            TypeTextAction: self.remote_execute_action,
            OpenUrlAction: self.remote_execute_action,
            KeyPressAction: self.remote_execute_action,
            PageUpAction: self.page_up,
            PageDownAction: self.page_down,
            RunTerminalCommand: self.remote_execute_action,
        }
        if self.use_grounding:
            self._action_map[MouseClickAction] = self.mouse_click
            self._action_map[MouseHoverAction] = self.mouse_hover
        else:
            self._action_map[MouseXYClickAction] = self.remote_execute_action
            self._action_map[MouseXYMoveAction] = self.remote_execute_action
            self._action_map[GetCursorPositionAction] = self.remote_execute_action
        self.actions = tuple(self._action_map.keys())

    def execute_action(self, action: Action) -> ImageObservation:
        action_type = type(action)
        if action_type in self._action_map:
            return self._action_map[action_type](action)
        raise ValueError(f"Unknown action type: {action_type}")

    def mouse_hover(self, action: MouseHoverAction) -> ImageObservation:
        x, y = self._grounding.get_coords(self.get_screen(), action.element_description)
        return self.remote_execute_action(MouseXYMoveAction(x=int(x), y=int(y)))

    def mouse_click(self, action: MouseClickAction) -> ImageObservation:
        x, y = self._grounding.get_coords(self.get_screen(), action.element_description)
        return self.remote_execute_action(MouseXYClickAction(x=int(x), y=int(y), button=action.button))

    def page_up(self, action: PageUpAction) -> ImageObservation:
        return self.remote_execute_action(KeyPressAction(text="Page_Up"))

    def page_down(self, action: PageDownAction) -> ImageObservation:
        return self.remote_execute_action(KeyPressAction(text="Page_Down"))

    def remote_execute_action(self, action: Action) -> ImageObservation:
        payload = {"kind": action.kind, "params": action.model_dump()}
        try:
            response = requests.post(f"{self.computer_url}/execute", json=payload)
            response.raise_for_status()
            obs_dict = response.json()
            logger.info(f"Received observation: {obs_dict.keys()}")
            return self.save_screenshot(ComputerObservation(**obs_dict))
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return ImageObservation(image_path="", error=f"API request failed: {str(e)}")

    def save_screenshot(self, obs: ComputerObservation) -> ImageObservation:
        if obs.base64_image:
            fname = f"{self._screenshot_dir}/screen_{int(time.time())}.png"
            with open(fname, "wb") as f:
                f.write(base64.b64decode(obs.base64_image))
            obs.image_path = fname
            obs.base64_image = None
        return obs

    def get_screen(self) -> Image:
        obs = self.remote_execute_action(GetCursorPositionAction())
        if obs.error:
            raise ValueError(f"Failed to get screen: {obs.error}")
        return Image.open(obs.image_path)

    def reset(self):
        self.remote_execute_action(RunTerminalCommand(command='xdotool search "" windowkill %@'))
