import logging
from typing import Any, Literal

from pydantic import Field

from tapeagents.core import Action, Observation
from tapeagents.tools.base import Multitool
from tapeagents.tools.computer.api import (
    get_cursor_position,
    key_press,
    mouse_click,
    mouse_drag,
    mouse_move,
    open_url,
    run_command,
    type_text,
)
from tapeagents.tools.computer.steps import (
    ComputerObservation,
    GetCursorPositionAction,
    KeyPressAction,
    MouseClickAction,
    MouseDragAction,
    MouseMoveAction,
    OpenUrlAction,
    RunTerminalCommand,
    TypeTextAction,
)

logger = logging.getLogger(__name__)

MAX_SCALING_TARGETS = {
    "XGA": (1024, 768),  # 4:3
    "WXGA": (1280, 800),  # 16:10
    "FWXGA": (1366, 768),  # ~16:9
}


def chunks(s: str, chunk_size: int) -> list[str]:
    return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]


class Computer(Multitool):
    actions: tuple[type[Action], ...] = (
        KeyPressAction,
        TypeTextAction,
        MouseMoveAction,
        MouseClickAction,
        MouseDragAction,
        GetCursorPositionAction,
        OpenUrlAction,
        RunTerminalCommand,
    )
    observations: tuple[type[Observation], ...] = (ComputerObservation,)

    width: int = Field(description="Screen width", default=MAX_SCALING_TARGETS["XGA"][0])
    height: int = Field(description="Screen height", default=MAX_SCALING_TARGETS["XGA"][1])
    scaling_enabled: bool = Field(default=True, description="Enable resolution scaling")
    typing_delay_ms: int = 180

    def model_post_init(self, __context: Any) -> None:
        self._action_map = {
            KeyPressAction: self._handle_key_press,
            TypeTextAction: self._handle_type_text,
            MouseMoveAction: self._handle_mouse_move,
            MouseClickAction: self._handle_mouse_click,
            MouseDragAction: self._handle_mouse_drag,
            GetCursorPositionAction: self._handle_get_cursor_position,
            OpenUrlAction: self._handle_open_url,
            RunTerminalCommand: self._handle_command,
        }

    def execute_action(self, action: Action) -> ComputerObservation:
        action_type = type(action)
        if action_type in self._action_map:
            return self._action_map[action_type](action)
        raise ValueError(f"Unknown action type: {action_type}")

    def _handle_command(self, action: RunTerminalCommand) -> ComputerObservation:
        try:
            obs_dict = run_command(action.command)
            obs = ComputerObservation(**obs_dict)
        except Exception as e:
            obs = ComputerObservation(error=f"Command {action.command} failed: {e}")
        return obs

    def _handle_open_url(self, action: OpenUrlAction) -> ComputerObservation:
        try:
            obs_dict = open_url(action.url)
            obs = ComputerObservation(**obs_dict)
        except Exception as e:
            obs = ComputerObservation(error=f"Open URL {action.url} failed: {e}")
        return obs

    def _handle_key_press(self, action: KeyPressAction) -> ComputerObservation:
        try:
            obs_dict = key_press(action.text)
            obs = ComputerObservation(**obs_dict)
        except Exception as e:
            obs = ComputerObservation(error=f"Key press {action.text} failed: {e}")
        return obs

    def _handle_type_text(self, action: TypeTextAction) -> ComputerObservation:
        try:
            obs_dict = type_text(action.text, delay_ms=self.typing_delay_ms)
            obs = ComputerObservation(**obs_dict)
        except Exception as e:
            obs = ComputerObservation(error=f"Typing failed: {e}")
        return obs

    def _handle_mouse_move(self, action: MouseMoveAction) -> ComputerObservation:
        x, y = self._scale_coordinates("api", action.x, action.y)
        try:
            obs_dict = mouse_move(x, y)
            obs = ComputerObservation(**obs_dict)
        except Exception as e:
            obs = ComputerObservation(error=f"Mouse move failed: {e}")
        return obs

    def _handle_mouse_click(self, action: MouseClickAction) -> ComputerObservation:
        try:
            obs_dict = mouse_click(action.x, action.y, button=action.button)
            obs = ComputerObservation(**obs_dict)
        except Exception as e:
            obs = ComputerObservation(error=f"Mouse click failed: {e}")
        return obs

    def _handle_mouse_drag(self, action: MouseDragAction) -> ComputerObservation:
        x, y = self._scale_coordinates("api", action.x, action.y)
        try:
            obs_dict = mouse_drag(x, y)
            obs = ComputerObservation(**obs_dict)
        except Exception as e:
            obs = ComputerObservation(error=f"Mouse drag failed: {e}")
        return obs

    def _handle_get_cursor_position(self, action: GetCursorPositionAction) -> ComputerObservation:
        try:
            obs_dict = get_cursor_position()
            obs = ComputerObservation(**obs_dict)
        except Exception as e:
            obs = ComputerObservation(error=f"Get cursor position failed: {e}")
        return obs

    def _scale_coordinates(self, source: Literal["api", "computer"], x: int, y: int) -> tuple[int, int]:
        """Scale coordinates based on target resolution"""
        if not self.scaling_enabled:
            return x, y
        ratio = self.width / self.height
        target_dimension = None
        for width, height in MAX_SCALING_TARGETS.values():
            # allow some error in the aspect ratio - not ratios are exactly 16:9
            if abs(width / height - ratio) < 0.02:
                if width < self.width:
                    target_dimension = (width, height)
                break
        if target_dimension is None:
            return x, y
        # should be less than 1
        x_scaling_factor = target_dimension[0] / self.width
        y_scaling_factor = target_dimension[1] / self.height
        if source == "api":
            if x > self.width or y > self.height:
                raise Exception(f"Coordinates {x}, {y} are out of bounds")
            # scale up
            return round(x / x_scaling_factor), round(y / y_scaling_factor)
        # scale down
        return round(x * x_scaling_factor), round(y * y_scaling_factor)
