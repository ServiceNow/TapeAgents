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
        payload = {"kind": action.__class__.__name__.replace("Action", "").lower(), "params": action.model_dump()}

        try:
            response = requests.post(f"{self.url}/execute", json=payload)
            response.raise_for_status()
            return ComputerObservation(**response.json())
        except requests.exceptions.RequestException as e:
            return ComputerObservation(error=f"API request failed: {str(e)}")

    def move_and_click(self, element_description: str, button: str = "left") -> ComputerObservation:
        self._take_screenshot()
        obs = self.execute_action(GetCursorPositionAction())
        last_image = Image.open(BytesIO(base64.b64decode(obs.base64_image)))
        x, y = self._locator.get_coords(last_image, f"click at {element_description}")
        self.execute_action(MouseMoveAction(x=x, y=y))
        return self.execute_action(MouseClickAction(button=button))

    def type_text(self, text: str) -> ComputerObservation:
        return self.execute_action(TypeTextAction(text=text))
