import base64
import os
import time
from io import BytesIO

import requests
from PIL import Image
from pydantic import Field

from tapeagents.core import Action
from tapeagents.steps import ImageObservation
from tapeagents.tools.base import Multitool
from tapeagents.tools.browser import MouseClickAction, MouseHoverAction, OpenUrlAction
from tapeagents.tools.locator import Locator

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
    )
    observations: tuple[type[ImageObservation], ...] = (ImageObservation,)
    computer_url: str = Field(description="Remote tool API URL")

    def model_post_init(self, __context):
        self._locator = Locator()
        self._last_image = None
        self._screenshot_dir = f"{self.exp_path}/attachments/remote_screenshots/"
        os.makedirs(self._screenshot_dir, exist_ok=True)
        self.remote_execute_action(GetCursorPositionAction())
        return super().model_post_init(__context)

    def execute_action(self, action: Action) -> ImageObservation:
        if isinstance(action, MouseClickAction):
            self.mouse_move(action.element_description)
            return self.remote_execute_action(CompMouseClickAction(button="left"))
        if isinstance(action, MouseHoverAction):
            return self.mouse_move(action.element_description)
        else:
            return self.remote_execute_action(action)

    def remote_execute_action(self, action: Action) -> ImageObservation:
        payload = {"kind": action.kind, "params": action.model_dump()}
        try:
            response = requests.post(f"{self.computer_url}/execute", json=payload)
            response.raise_for_status()
            obs_dict = response.json()
            return self.convert_observation(ComputerObservation(**obs_dict))
        except requests.exceptions.RequestException as e:
            return ImageObservation(image_path="", error=f"API request failed: {str(e)}")

    def convert_observation(self, obs: ComputerObservation) -> ImageObservation:
        bimage = obs.base64_image
        if not bimage:
            return ComputerObservation(error="Failed to get screenshot")
        image_data = base64.b64decode(bimage)
        image_name_with_timestamp = f"{self._screenshot_dir}/screen_{int(time.time())}.png"
        with open(image_name_with_timestamp, "wb") as f:
            f.write(image_data)
        self._last_image = Image.open(BytesIO(image_data))
        return ImageObservation(image_path=image_name_with_timestamp, error=obs.error, image_caption=obs.text)

    def mouse_move(self, element_description: str, button: str = "left") -> ImageObservation:
        x, y = self._locator.get_coords(self._last_image, f"click at {element_description}")
        x, y = int(x), int(y)
        return self.remote_execute_action(MouseMoveAction(x=x, y=y))
