from typing import Literal

from pydantic import Field

from tapeagents.core import Action
from tapeagents.steps import ImageObservation
from tapeagents.utils import image_base64_message


class KeyPressAction(Action):
    """Action that simulates keyboard key press"""

    kind: Literal["key_press_action"] = "key_press_action"
    text: str = Field(description="Key combination to press")


class TypeTextAction(Action):
    """Action that types text character by character. Use \n in the end for Enter key press"""

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
    button: Literal["left", "right", "middle", "double_left"] = Field(
        description="Mouse button to click", default="left"
    )


class MouseDragAction(Action):
    """Action that performs mouse drag from current position"""

    kind: Literal["mouse_drag_action"] = "mouse_drag_action"
    x: int = Field(description="Target X coordinate")
    y: int = Field(description="Target Y coordinate")


class GetCursorPositionAction(Action):
    """Action that gets current cursor position"""

    kind: Literal["get_cursor_position_action"] = "get_cursor_position_action"


class OpenUrlAction(Action):
    """
    Action that opens a URL in the browser.
    """

    kind: Literal["open_url_action"] = "open_url_action"
    url: str = Field(description="URL to navigate to")


class RunTerminalCommand(Action):
    """
    Action that executes a command in the terminal.
    """

    kind: Literal["run_terminal_command"] = "run_terminal_command"
    command: str = Field(description="Command to execute")


class ComputerObservation(ImageObservation):
    """Base observation returned by computer actions"""

    kind: Literal["computer_observation"] = "computer_observation"
    output: str = ""
    image_path: str = ""
    base64_image: str | None = None

    def llm_view(self) -> list[dict]:
        content = []
        if self.output:
            content.append({"type": "text", "text": self.output})
        if self.error:
            content.append({"type": "text", "text": f"Error: {self.error}"})
        if self.image_path:
            content.append(image_base64_message(self.image_path))
        return content
