from typing import Literal

from pydantic import Field

from tapeagents.core import Action, Observation


class KeyPressAction(Action):
    """Action that simulates keyboard key press"""

    kind: Literal["key_press_action"] = "key_press_action"
    text: str = Field(description="Key combination to press")


class TypeTextAction(Action):
    """Action that types text character by character"""

    kind: Literal["type_text_action"] = "type_text_action"
    text: str = Field(description="Text to type")


class MouseMoveAction(Action):
    """Action that moves mouse cursor to coordinates"""

    kind: Literal["mouse_move_action"] = "mouse_move_action"
    x: int = Field(description="X coordinate")
    y: int = Field(description="Y coordinate")


class MouseClickAction(Action):
    """Action that performs mouse click"""

    kind: Literal["mouse_click_action"] = "mouse_click_action"
    button: Literal["left", "right", "middle", "double"] = Field(description="Mouse button to click", default="left")


class MouseDragAction(Action):
    """Action that performs mouse drag from current position"""

    kind: Literal["mouse_drag_action"] = "mouse_drag_action"
    x: int = Field(description="Target X coordinate")
    y: int = Field(description="Target Y coordinate")


class GetCursorPositionAction(Action):
    """Action that gets current cursor position"""

    kind: Literal["get_cursor_position_action"] = "get_cursor_position_action"


class ComputerObservation(Observation):
    """Base observation returned by computer actions"""

    kind: Literal["computer_observation"] = "computer_observation"
    text: str = ""
    error: str = ""
    base64_image: str | None = None
