import base64
import json
import os
from typing import Literal

from pydantic import Field
from tapeagents.core import Action, Observation, Thought

################# Actions #################


class GetVideoAction(Action):
    """
    Action that loads the video from the provided url and returns the subtitle and its contact sheet.
    """

    kind: Literal["get_video_action"] = "get_video_action"
    video_url: str = Field(description="url of the video to watch")
    start_time: str = Field(
        description="time of the video to start watching from, if applicable, otherwise empty str", default=""
    )


################### Observations ###################


class ImageObservation(Observation):
    kind: Literal["image_observation"] = "image_observation"
    image_path: str
    thumbnail_path: str
    image_caption: str
    error: int | None = None

    def llm_view_trimmed(self, indent: int | None = 2) -> list[dict]:
        step_data = {"kind": self.kind, "image_caption": self.image_caption, "error": self.error}
        return json.dumps(step_data, indent=indent, ensure_ascii=False)

    def llm_view(self, indent: int | None = 2) -> list[dict]:
        llm_view = [{"type": "text", "text": self.llm_view_trimmed()}]
        llm_view.append(image_base64_message(self.image_path))
        return llm_view


class VideoObservation(Observation):
    kind: Literal["video_observation"] = "video_observation"
    video_path: str
    video_contact_sheet_paths: list[str]
    thumbnail_path: str | None = None
    subtitle_path: str | None = None
    subtitle_text: str | None = None
    error: int | None = None

    def llm_view_trimmed(self, indent: int | None = 2) -> list[dict]:
        step_data = {"kind": self.kind, "subtitle_text": self.subtitle_text, "error": self.error}
        return json.dumps(step_data, indent=indent, ensure_ascii=False)

    def llm_view(self, indent: int | None = 2) -> list[dict]:
        llm_view = [
            {"type": "text", "text": self.llm_view_trimmed()},
        ]
        for path in self.video_contact_sheet_paths:
            llm_view.append(image_base64_message(path))
        return llm_view


################### Thoughts ###################


class AnalyzeVideoThought(Thought):
    """
    Thought that outputs the detailed description of each frame of the video contact sheet
    """

    kind: Literal["analyze_video_thought"] = "analyze_video_thought"
    description: str = Field(description="description of the video using both the subtitle and the video contact sheet")
    frame_description: list[str] = Field(
        description="detailed and specific description of each frame of the video contact sheet"
    )


def image_base64_message(image_path: str) -> str:
    _, image_extension = os.path.splitext(image_path)
    content_type = f"image/{image_extension}"
    base64_image = encode_image(image_path)
    message = {"type": "image_url", "image_url": {"url": f"data:{content_type};base64,{base64_image}"}}
    print(message)
    return message


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
