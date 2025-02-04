import base64
import logging
import os
import shlex
import shutil
import time
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field

from tapeagents.core import Action, Observation
from tapeagents.tools.base import Multitool

from .steps import (
    ComputerObservation,
    GetCursorPositionAction,
    KeyPressAction,
    MouseClickAction,
    MouseDragAction,
    MouseMoveAction,
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
    )
    observations: tuple[type[Observation], ...] = (ComputerObservation,)

    width: int = Field(description="Screen width", default=MAX_SCALING_TARGETS["XGA"][0])
    height: int = Field(description="Screen height", default=MAX_SCALING_TARGETS["XGA"][1])
    display_num: int | None = Field(default=1, description="X display number")
    display_height_px: int = Field(default=MAX_SCALING_TARGETS["XGA"][0], description="Display height in pixels")
    display_width_px: int = Field(default=MAX_SCALING_TARGETS["XGA"][1], description="Display width in pixels")
    screenshot_delay: float = Field(default=2.0, description="Delay before screenshot")
    scaling_enabled: bool = Field(default=True, description="Enable resolution scaling")
    tmp_screenshots_dir: str = "/tmp/screenshots/"
    typing_delay_ms: int = 12
    typing_group_size: int = 50
    _xdotool: str = ""
    _display_prefix: str = ""

    def model_post_init(self, __context: Any) -> None:
        if (display_num := os.getenv("DISPLAY_NUM")) is not None:
            self.display_num = int(display_num)
        self._display_prefix = f"DISPLAY=:{self.display_num} "

        self._xdotool = f"{self._display_prefix}xdotool"

        # Add action mapping
        self._action_map = {
            KeyPressAction: self._handle_key_press,
            TypeTextAction: self._handle_type_text,
            MouseMoveAction: self._handle_mouse_move,
            MouseClickAction: self._handle_mouse_click,
            MouseDragAction: self._handle_mouse_drag,
            GetCursorPositionAction: self._handle_get_cursor_position,
        }

    def execute_action(self, action: Action) -> ComputerObservation:
        action_type = type(action)
        if action_type in self._action_map:
            return self._action_map[action_type](action)
        raise ValueError(f"Unknown action type: {action_type}")

    def _handle_key_press(self, action: KeyPressAction) -> ComputerObservation:
        return self._execute_shell(f"{self._xdotool} key -- {action.text}")

    def _handle_type_text(self, action: TypeTextAction) -> ComputerObservation:
        output = []
        for chunk in chunks(action.text, self.typing_group_size):
            cmd = f"{self._xdotool} type --delay {self.typing_delay_ms} -- {shlex.quote(chunk)}"
            result = self._execute_shell(cmd, take_screenshot=False)
            output.append(result.text)
        return self._take_screenshot(output="".join(output))

    def _handle_mouse_move(self, action: MouseMoveAction) -> ComputerObservation:
        x, y = self._scale_coordinates("api", action.x, action.y)
        return self._execute_shell(f"{self._xdotool} mousemove --sync {x} {y}")

    def _handle_mouse_click(self, action: MouseClickAction) -> ComputerObservation:
        click_arg = {"left": "1", "right": "3", "middle": "2", "double": "--repeat 2 --delay 500 1"}[action.button]
        return self._execute_shell(f"{self._xdotool} click {click_arg}")

    def _handle_mouse_drag(self, action: MouseDragAction) -> ComputerObservation:
        x, y = self._scale_coordinates("api", action.x, action.y)
        return self._execute_shell(f"{self._xdotool} mousedown 1 mousemove --sync {x} {y} mouseup 1")

    def _handle_get_cursor_position(self, action: GetCursorPositionAction) -> ComputerObservation:
        result = self._execute_shell(f"{self._xdotool} getmouselocation --shell", take_screenshot=False)
        output = result.text
        x, y = self._scale_coordinates(
            "computer", int(output.split("X=")[1].split("\n")[0]), int(output.split("Y=")[1].split("\n")[0])
        )
        return ComputerObservation(text=f"X={x},Y={y}")

    def _execute_shell(self, command: str, take_screenshot=True) -> ComputerObservation:
        """Execute shell command and return observation"""
        try:
            output = os.popen(command).read()
            error = ""
        except Exception as e:
            output = ""
            error = str(e)

        obs = ComputerObservation(text=output, error=error)

        if take_screenshot:
            time.sleep(self.screenshot_delay)
            screenshot = self._take_screenshot()
            obs.base64_image = screenshot.base64_image

        return obs

    def _take_screenshot(self) -> ComputerObservation:
        """Take screenshot and return observation with base64 image"""
        output_dir = Path(self.tmp_screenshots_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"screenshot_{uuid4().hex}.png"

        if shutil.which("gnome-screenshot"):
            cmd = f"{self._display_prefix}gnome-screenshot -f {path} -p"
        else:
            cmd = f"{self._display_prefix}scrot -p {path}"

        try:
            os.system(cmd)
            if self.scaling_enabled:
                x, y = self._scale_coordinates("computer", self.width, self.height)
                os.system(f"convert {path} -resize {x}x{y}! {path}")

            base64_image = base64.b64encode(path.read_bytes()).decode()
            path.unlink()  # Delete temporary file
            return ComputerObservation(base64_image=base64_image)
        except Exception as e:
            return ComputerObservation(error=f"Screenshot failed: {e}")

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
