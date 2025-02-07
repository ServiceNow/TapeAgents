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
    )
    observations: tuple[type[ImageObservation], ...] = (ImageObservation,)
    computer_url: str = Field(description="Remote tool API URL")

    def model_post_init(self, __context):
        self._grounding = GroundingModel()
        self._screenshot_dir = f"{self.exp_path}/attachments/remote_screenshots/"
        os.makedirs(self._screenshot_dir, exist_ok=True)
        return super().model_post_init(__context)

    def execute_action(self, action: Action) -> ImageObservation:
        if isinstance(action, MouseClickAction):
            self.mouse_move(action.element_description)
            return self.remote_execute_action(CompMouseClickAction(button="left"))
        elif isinstance(action, MouseHoverAction):
            return self.mouse_move(action.element_description)
        elif isinstance(action, PageUpAction):
            return self.remote_execute_action(KeyPressAction(text="Page_Up"))
        elif isinstance(action, PageDownAction):
            return self.remote_execute_action(KeyPressAction(text="Page_Down"))
        elif isinstance(action, OpenUrlAction):
            self.mouse_move("top address bar")
            self.remote_execute_action(CompMouseClickAction(button="left"))
            self.remote_execute_action(TypeTextAction(text=action.url))
            self.remote_execute_action(KeyPressAction(text="Return"))
            time.sleep(5)  # wait for page to load
            return self.remote_execute_action(GetCursorPositionAction())
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

    def mouse_move(self, element_description: str, button: str = "left") -> ImageObservation:
        x, y = self._grounding.get_coords(self.get_screen(), f"click {element_description}")
        return self.remote_execute_action(MouseMoveAction(x=int(x), y=int(y)))
