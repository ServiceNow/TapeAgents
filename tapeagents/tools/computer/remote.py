import base64
import os
import time

import requests
from PIL import Image
from pydantic import Field

from tapeagents.core import Action
from tapeagents.steps import ImageObservation
from tapeagents.tools.base import Multitool
from tapeagents.tools.browser import MouseClickAction, MouseHoverAction, OpenUrlAction, PageDownAction, PageUpAction
from tapeagents.tools.grounding import GroundingModel

from .steps import (
    ComputerObservation,
    GetCursorPositionAction,
    KeyPressAction,
    MouseClickAction as CompMouseClickAction,
    MouseMoveAction,
    TypeTextAction,
)


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

    def model_post_init(self, __context):
        self._grounding = GroundingModel()
        self._screenshot_dir = f"{self.exp_path}/attachments/remote_screenshots/"
        os.makedirs(self._screenshot_dir, exist_ok=True)
        self._action_map = {
            TypeTextAction: self.remote_execute_action,
            MouseHoverAction: self.mouse_hover,
            MouseClickAction: self.mouse_click,
            OpenUrlAction: self.open_url,
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

    def open_url(self, action: OpenUrlAction) -> ImageObservation:
        self.mouse_hover("top address bar")
        self.remote_execute_action(CompMouseClickAction(button="left"))
        self.remote_execute_action(TypeTextAction(text=action.url))
        self.remote_execute_action(KeyPressAction(text="Return"))
        time.sleep(5)
        return self.remote_execute_action(GetCursorPositionAction())

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
            print(f"API request failed: {str(e)}")
            return ImageObservation(image_path="", error=f"API request failed: {str(e)}")

    def convert_observation(self, obs: ComputerObservation) -> ImageObservation:
        bimage = obs.base64_image
        if not bimage:
            return ComputerObservation(error="Failed to get screenshot")
        image_data = base64.b64decode(bimage)
        image_name_with_timestamp = f"{self._screenshot_dir}/screen_{int(time.time())}.png"
        with open(image_name_with_timestamp, "wb") as f:
            f.write(image_data)
        return ImageObservation(
            image_path=image_name_with_timestamp,
            error=obs.error,
            image_caption=f"Current state of the computer screen. Additional info: {obs.text}",
        )

    def get_screen(self) -> Image:
        obs = self.remote_execute_action(GetCursorPositionAction())
        return Image.open(obs.image_path)
