import traceback
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from .computer import Computer
from .steps import (
    GetCursorPositionAction,
    KeyPressAction,
    MouseClickAction,
    MouseDragAction,
    MouseMoveAction,
    TypeTextAction,
)

app = FastAPI()

ACTION_MAP = {
    "key_press": KeyPressAction,
    "type_text": TypeTextAction,
    "mouse_move": MouseMoveAction,
    "mouse_click": MouseClickAction,
    "mouse_drag": MouseDragAction,
    "get_cursor_position": GetCursorPositionAction,
}


class ActionRequest(BaseModel):
    kind: str
    params: Dict[str, Any]


@app.post("/execute")
async def execute_action(request: ActionRequest):
    try:
        if request.kind not in ACTION_MAP:
            raise ValueError(f"Unknown action type: {request.kind}")

        action_class = ACTION_MAP[request.kind]
        action = action_class(**request.params)

        observation = computer.execute_action(action)  # Use existing instance

        return observation.model_dump()
    except Exception as e:
        detail = f"{str(e)}\n{traceback.format_exc()}"
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
    global computer
    computer = Computer()  # Create single instance
    port = 8000
    print(f"Starting api on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
