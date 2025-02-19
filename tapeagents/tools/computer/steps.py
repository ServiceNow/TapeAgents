from typing import Literal

from pydantic import Field

from tapeagents.core import Action
from tapeagents.steps import ImageObservation
from tapeagents.utils import image_base64_message


class KeyPressAction(Action):
    """
    Action that simulates keyboard key press. Use + to separate keys.
    Supports aliases: ctrl, shift, alt, esc, enter, tab, backspace, delete, home, end, pageup, pagedown, up, down, left, right.
    Example: "ctrl+c" or "ctrl+shift+t"
    """

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


class MouseHoverAction(Action):
    """
    Action that hovers over an icon or control on the computer screen
    """

    kind: Literal["mouse_hover_action"] = "mouse_hover_action"
    element_description: str = Field(description="brief description of the element to hover over")


class MouseClickAction(Action):
    """Action that performs mouse click"""

    kind: Literal["mouse_click_action"] = "mouse_click_action"
    x: int = Field(description="X coordinate")
    y: int = Field(description="Y coordinate")
    button: Literal["left", "right", "middle", "double_left"] = Field(
        description="Mouse button to click", default="left"
    )


class MouseClickAtAction(Action):
    """
    Action that clicks an element on the computer screen.
    When mentioning a date in the element description, use the format commonly spoken or written by humans,
    such as "2 February 2025," rather than machine-readable formats. The day should come before the month,
    and the year should be written in full (e.g., "3 November 2023" instead of "2023-11-03").
    Only describe one specific element that is currently visible on the screen!
    """

    kind: Literal["mouse_click_at_action"] = "mouse_click_at_action"
    button: Literal["left", "double_left", "right"] = Field(
        description="mouse button to click, double_left for double left click", default="left"
    )
    element_description: str = Field(description="brief description of the element to click")


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
    wait_output: bool = Field(description="Whether to wait for command to finish", default=True)


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
