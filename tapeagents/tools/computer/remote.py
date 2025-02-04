import base64
from io import BytesIO

import requests
from PIL import Image
from pydantic import Field

from tapeagents.core import Action
from tapeagents.tools.base import Multitool
from tapeagents.tools.locator import Locator

from .steps import (
    ComputerObservation,
    GetCursorPositionAction,
    KeyPressAction,
    MouseClickAction,
    MouseDragAction,
    MouseMoveAction,
    TypeTextAction,
)


class RemoteComputer(Multitool):
    actions: tuple[type[Action], ...] = (
        KeyPressAction,
        TypeTextAction,
        MouseMoveAction,
        MouseClickAction,
        MouseDragAction,
        GetCursorPositionAction,
    )
    observations: tuple[type[ComputerObservation], ...] = (ComputerObservation,)
    computer_url: str = Field(description="Remote tool API URL")

    def model_post_init(self, __context):
        self._locator = Locator()
        return super().model_post_init(__context)

    def execute_action(self, action: Action) -> ComputerObservation:
        payload = {"kind": action.kind, "params": action.model_dump()}
        try:
            response = requests.post(f"{self.computer_url}/execute", json=payload)
            response.raise_for_status()
            obs_dict = response.json()
            return ComputerObservation(**obs_dict)
        except requests.exceptions.RequestException as e:
            return ComputerObservation(error=f"API request failed: {str(e)}")

    def move_and_click(self, element_description: str, button: str = "left") -> ComputerObservation:
        obs = self.execute_action(GetCursorPositionAction())
        bimage = obs.base64_image
        if not bimage:
            return ComputerObservation(error="Failed to get screenshot")
        image_data = base64.b64decode(bimage)
        with open("last_screenshot.png", "wb") as f:
            f.write(image_data)
        last_image = Image.open(BytesIO(image_data))
        x, y = self._locator.get_coords(last_image, f"click at {element_description}")
        x, y = int(x), int(y)
        self.execute_action(MouseMoveAction(x=x, y=y))
        return self.execute_action(MouseClickAction(button=button))

    def type_text(self, text: str) -> ComputerObservation:
        return self.execute_action(TypeTextAction(text=text))
