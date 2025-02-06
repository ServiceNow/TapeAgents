import traceback
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from tapeagents.core import Action
from tapeagents.tools.computer.computer import Computer
from tapeagents.tools.computer.steps import (
    GetCursorPositionAction,
    KeyPressAction,
    MouseClickAction,
    MouseDragAction,
    MouseMoveAction,
    TypeTextAction,
)

app = FastAPI()

ACTION_MAP: dict[str : type[Action]] = {
    "key_press_action": KeyPressAction,
    "type_text_action": TypeTextAction,
    "mouse_move_action": MouseMoveAction,
    "mouse_click_action": MouseClickAction,
    "mouse_drag_action": MouseDragAction,
    "get_cursor_position_action": GetCursorPositionAction,
}


class ActionRequest(BaseModel):
    kind: str
    params: Dict[str, Any]


@app.post("/execute")
async def execute_action(request: ActionRequest):
    try:
        print(f"Received action request: {request}")
        print(f"Action kind: {request.kind}")
        print(f"Action params: {request.params}")
        if request.kind not in ACTION_MAP:
            raise ValueError(f"Unknown action type: {request.kind}")

        action_class: type[Action] = ACTION_MAP[request.kind]
        print(f"Action class: {action_class}")
        action = action_class.model_validate(request.params)
        print(f"Action deserialized: {action}")
        observation = computer.execute_action(action)  # Use existing instance
        print(f"Observation received {observation.kind}")
        obs = observation.model_dump()
        print_obs = observation.model_dump()
        print_obs["base64_image"] = print_obs["base64_image"][:10]
        print(f"Observation serialized: {print_obs}")
        return obs
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
    global computer
    computer = Computer()  # Create single instance
    port = 8000
    print(f"Starting api on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
