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
    MouseClickAction as CompMouseClickAction,
    MouseMoveAction,
    OpenUrlAction,
    TypeTextAction,
)

logger = logging.getLogger("remote")
logger.setLevel(logging.INFO)


class MouseClickAction(Action):
    """
    Action that clicks an element on the computer screen.
    When mentioning a date in the element description, use the format commonly spoken or written by humans,
    such as "2 February 2025," rather than machine-readable formats. The day should come before the month,
    and the year should be written in full (e.g., "3 November 2023" instead of "2023-11-03").
    Only describe one specific element that is currently visible on the screen!
    """

    kind: Literal["mouse_click_action"] = "mouse_click_action"
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
    actions: tuple[type[Action], ...] = (
        TypeTextAction,
        MouseHoverAction,
        MouseClickAction,
        OpenUrlAction,
        KeyPressAction,
        PageUpAction,
        PageDownAction,
        GetCursorPositionAction,
    )
    observations: tuple[type[ImageObservation], ...] = (ImageObservation,)
    computer_url: str = Field(description="Remote tool API URL")
    grounding_api_url: str = Field(description="Grounding API URL")

    def model_post_init(self, __context):
        self._grounding = GroundingModel(url=self.grounding_api_url)
        self._screenshot_dir = f"{self.exp_path}/attachments/remote_screenshots/"
        os.makedirs(self._screenshot_dir, exist_ok=True)
        self._action_map = {
            TypeTextAction: self.remote_execute_action,
            MouseHoverAction: self.mouse_hover,
            MouseClickAction: self.mouse_click,
            OpenUrlAction: self.remote_execute_action,
            KeyPressAction: self.remote_execute_action,
            PageUpAction: self.page_up,
            PageDownAction: self.page_down,
            GetCursorPositionAction: self.remote_execute_action,
        }

    def execute_action(self, action: Action) -> ImageObservation:
        action_type = type(action)
        if action_type in self._action_map:
            return self._action_map[action_type](action)
        raise ValueError(f"Unknown action type: {action_type}")

    def mouse_hover(self, action: MouseHoverAction) -> ImageObservation:
        x, y = self._grounding.get_coords(self.get_screen(), f"click {action.element_description}")
        return self.remote_execute_action(MouseMoveAction(x=int(x), y=int(y)))

    def mouse_click(self, action: MouseClickAction) -> ImageObservation:
        self.mouse_hover(action)
        return self.remote_execute_action(CompMouseClickAction(button="left"))

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
            return self.convert_observation(ComputerObservation(**obs_dict))
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return ImageObservation(image_path="", error=f"API request failed: {str(e)}")

    def convert_observation(self, obs: ComputerObservation) -> ImageObservation:
        bimage = obs.base64_image
        if not bimage:
            return ComputerObservation(error="Failed to get screenshot")
        image_name_with_timestamp = f"{self._screenshot_dir}/screen_{int(time.time())}.png"
        with open(image_name_with_timestamp, "wb") as f:
            f.write(base64.b64decode(bimage))
        return ImageObservation(
            image_path=image_name_with_timestamp,
            error=obs.error,
            image_caption=f"Current state of the computer screen. Additional info: {obs.text}",
        )

    def get_screen(self) -> Image:
        obs = self.remote_execute_action(GetCursorPositionAction())
        if obs.error:
            raise ValueError(f"Failed to get screen: {obs.error}")
        return Image.open(obs.image_path)
