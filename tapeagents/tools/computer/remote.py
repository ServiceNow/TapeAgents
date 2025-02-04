import requests
from pydantic import Field

from tapeagents.core import Action
from tapeagents.tools.base import Multitool

from .steps import (
    ComputerObservation,
    GetCursorPositionAction,
    KeyPressAction,
    MouseClickAction,
    MouseDragAction,
    MouseMoveAction,
    ScreenshotAction,
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
        ScreenshotAction,
    )
    observations: tuple[type[ComputerObservation], ...] = (ComputerObservation,)
    url: str = Field(description="Remote tool API URL")

    def execute_action(self, action: Action) -> ComputerObservation:
        payload = {"kind": action.__class__.__name__.replace("Action", "").lower(), "params": action.model_dump()}

        try:
            response = requests.post(f"{self.url}/execute", json=payload)
            response.raise_for_status()
            return ComputerObservation(**response.json())
        except requests.exceptions.RequestException as e:
            return ComputerObservation(error=f"API request failed: {str(e)}")
