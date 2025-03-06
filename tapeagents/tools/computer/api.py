import base64
import logging
import os
import subprocess
import tempfile
import time
import traceback

import pyautogui
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(funcName)s - %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

TYPING_DELAY_MS = 180
SCREENSHOT_DELAY = 0.5
CLICK_DELAY = 3.0
WEB_PAGE_LOAD_DELAY = 4.0
MAX_RESPONSE_LEN: int = 16000
RUN_TIMEOUT_SEC = 5.0
TRUNCATED_MESSAGE: str = "<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>"


def _take_screenshot() -> dict:
    """Take screenshot and return observation with base64 image"""
    time.sleep(SCREENSHOT_DELAY)
    try:
        screenshot: Image = pyautogui.screenshot()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            screenshot.save(tmp.name)
            tmp.seek(0)
            base64_image = base64.b64encode(tmp.read()).decode()
        return {"base64_image": base64_image, "output": "Screenshot of the current computer screen"}
    except Exception as e:
        err = f"Screenshot failed: {e}\n{traceback.format_exc()}"
        logger.exception(str(e))
        return {"error": err}


def key_press(text: str) -> dict:
    try:
        keys = text.lower().replace("+", " ").split()
        logger.info(f"Pressing keys: {keys}")
        pyautogui.hotkey(*keys)
        return _take_screenshot()
    except Exception as e:
        return {"error": f"Key press {text} failed: {e}"}


def type_text(text: str, delay_ms: int = TYPING_DELAY_MS) -> dict:
    try:
        pyautogui.write(text, interval=delay_ms / 1000)
        return _take_screenshot()
    except Exception as e:
        return {"error": f"Typing failed: {e}"}


def mouse_move(x: int, y: int) -> dict:
    try:
        pyautogui.moveTo(x, y, duration=0.2)
        return _take_screenshot()
    except Exception as e:
        return {"error": f"Mouse move failed: {e}"}


def mouse_click(x: int, y: int, button: str) -> dict:
    try:
        pyautogui.moveTo(x, y, duration=0.2)
        time.sleep(TYPING_DELAY_MS / 1000)
        if button == "double_left":
            pyautogui.doubleClick()
        else:
            pyautogui.click(button=button)
        time.sleep(CLICK_DELAY)
        return _take_screenshot()
    except Exception as e:
        return {"error": f"Mouse click failed: {e}"}


def mouse_drag(x: int, y: int) -> dict:
    try:
        pyautogui.dragTo(x, y)
        return _take_screenshot()
    except Exception as e:
        return {"error": f"Mouse drag failed: {e}"}


def get_cursor_position() -> dict:
    try:
        obs = _take_screenshot()
        x, y = pyautogui.position()
        obs["output"] = f"[{x}, {y}]"
        return obs
    except Exception as e:
        return {"error": f"Get cursor position failed: {e}"}


def open_url(url: str, load_delay: int = WEB_PAGE_LOAD_DELAY) -> dict:
    try:
        logger.info(f"Opening URL: {url}")
        os.system(f"open {url} &")
        time.sleep(load_delay)
        return _take_screenshot()
    except Exception as e:
        error = f"Open URL {url} failed: {e}"
        logger.error(error)
        return {"error": error}


def maybe_truncate(content: str, truncate_after: int | None = MAX_RESPONSE_LEN):
    """Truncate content and append a notice if content exceeds the specified length."""
    return (
        content
        if not truncate_after or len(content) <= truncate_after
        else content[:truncate_after] + TRUNCATED_MESSAGE
    )


def run(
    cmd: str,
    wait_output: bool,
    timeout_sec: float = RUN_TIMEOUT_SEC,
    truncate_after: int | None = MAX_RESPONSE_LEN,
):
    """Run a shell command with a timeout."""
    try:
        stdout, stderr = "", ""
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if wait_output:
            stdout, stderr = process.communicate(timeout=timeout_sec)
            stdout = stdout.decode()
            stderr = stderr.decode()
        return (
            process.returncode or 0,
            maybe_truncate(stdout, truncate_after=truncate_after),
            maybe_truncate(stderr, truncate_after=truncate_after),
        )
    except subprocess.TimeoutExpired as exc:
        process.kill()
        raise TimeoutError(f"Command '{cmd}' timed out after {timeout_sec} seconds") from exc


def run_command(command: str, wait_output: bool) -> dict:
    try:
        exit_code, output, error = run(command, wait_output)
        obs = _take_screenshot()
        obs["output"] = output
        if exit_code != 0:
            obs["error"] = f"exit code: {exit_code}"
            if error:
                logger.info(f"Error: {error}")
                obs["error"] += f"\nstderr: {error}"
        return obs
    except Exception as e:
        error = f"Command {command} failed: {e}"
        logger.error(error)
        return {"error": error}


ACTION_MAP: dict[str:callable] = {
    "key_press_action": key_press,
    "type_text_action": type_text,
    "mouse_move_action": mouse_move,
    "mouse_click_action": mouse_click,
    "mouse_drag_action": mouse_drag,
    "get_cursor_position_action": get_cursor_position,
    "open_url_action": open_url,
    "run_terminal_command": run_command,
}

app = FastAPI()


@app.post("/execute")
async def execute_action(request: dict):
    try:
        if not isinstance(request, dict) or "kind" not in request or "params" not in request:
            raise ValueError("Request must contain 'kind' and 'params' fields")

        if request["kind"] not in ACTION_MAP:
            raise ValueError(f"Unknown action type: {request['kind']}")
        kwargs = request["params"]
        kwargs.pop("metadata", None)
        kwargs.pop("kind", None)
        logger.info(f"Run action: {request['kind']} with params: {kwargs}")
        observation = ACTION_MAP[request["kind"]](**kwargs)
        if observation.get("error"):
            logger.error(f"Action failed: {observation['error']}")
        logger.info(f"Return observation: {observation.keys()}")
        return observation
    except Exception as e:
        logger.exception(str(e))
        raise HTTPException(status_code=400, detail=f"{str(e)}\n{traceback.format_exc()}")


if __name__ == "__main__":
    port = 8000
    logger.info(f"Starting api on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
