import base64
import os
import time
import traceback

import pyautogui
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

app = FastAPI()
TYPING_DELAY_SEC = 0.18
SCREENSHOT_DELAY = 2.0


def _take_screenshot() -> dict:
    """Take screenshot and return observation with base64 image"""
    time.sleep(SCREENSHOT_DELAY)
    try:
        screenshot: Image = pyautogui.screenshot()
        screenshot_path = "temp_screenshot.png"
        screenshot.save(screenshot_path)
        with open(screenshot_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode()
        os.remove(screenshot_path)
        x, y = pyautogui.position()
        return {"base64_image": base64_image, "text": f"[{x}, {y}]"}
    except Exception as e:
        err = f"Screenshot failed: {e}\n{traceback.format_exc()}"
        print(err)
        return {"error": err}


def _scale_coordinates(source: str, x: int, y: int):
    return x, y


def key_press(text: str) -> dict:
    try:
        keys = text.lower().replace("+", " ").split()
        print(f"Pressing keys: {keys}")
        pyautogui.hotkey(*keys)
        return _take_screenshot()
    except Exception as e:
        return {"error": f"Key press {text} failed: {e}"}


def type_text(text: str) -> dict:
    try:
        pyautogui.write(text, interval=TYPING_DELAY_SEC)
        return _take_screenshot()
    except Exception as e:
        return {"error": f"Typing failed: {e}"}


def mouse_move(x: int, y: int) -> dict:
    try:
        scaled_x, scaled_y = _scale_coordinates("api", x, y)
        pyautogui.moveTo(scaled_x, scaled_y)
        return _take_screenshot()
    except Exception as e:
        return {"error": f"Mouse move failed: {e}"}


def mouse_click(button: str) -> dict:
    try:
        if button == "double":
            pyautogui.doubleClick()
        else:
            pyautogui.click(button=button)
        return _take_screenshot()
    except Exception as e:
        return {"error": f"Mouse click failed: {e}"}


def mouse_drag(x: int, y: int) -> dict:
    try:
        scaled_x, scaled_y = _scale_coordinates("api", x, y)
        pyautogui.dragTo(scaled_x, scaled_y)
        return _take_screenshot()
    except Exception as e:
        return {"error": f"Mouse drag failed: {e}"}


def get_cursor_position() -> dict:
    try:
        return _take_screenshot()
    except Exception as e:
        return {"error": f"Get cursor position failed: {e}"}


def open_url(url: str) -> dict:
    try:
        os.popen(f"open {url}").read()
        return _take_screenshot()
    except Exception as e:
        return {"error": f"Get cursor position failed: {e}"}


ACTION_MAP: dict[str:callable] = {
    "key_press_action": key_press,
    "type_text_action": type_text,
    "mouse_move_action": mouse_move,
    "mouse_click_action": mouse_click,
    "mouse_drag_action": mouse_drag,
    "get_cursor_position_action": get_cursor_position,
    "open_url_action": open_url,
}


@app.post("/execute")
async def execute_action(request: dict):
    try:
        if not isinstance(request, dict) or "kind" not in request or "params" not in request:
            raise ValueError("Request must contain 'kind' and 'params' fields")

        print(f"Received action request: {request}")
        print(f"Action kind: {request['kind']}")
        print(f"Action params: {request['params']}")

        if request["kind"] not in ACTION_MAP:
            raise ValueError(f"Unknown action type: {request['kind']}")
        kwargs = request["params"]
        kwargs.pop("metadata", None)
        kwargs.pop("kind", None)
        observation = ACTION_MAP[request["kind"]](**kwargs)
        return observation
    except Exception as e:
        detail = f"{str(e)}\n{traceback.format_exc()}"
        print(f"Error: {detail}")
        raise HTTPException(status_code=400, detail=detail)


@app.post("/save_file")
async def save_file(file: UploadFile = File(...), path: str = Form(...)):
    try:
        content = await file.read()
        with open(path, "wb") as f:
            f.write(content)
        return {"status": "success", "message": f"File saved to {path}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    port = 8000
    print(f"Starting api on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
