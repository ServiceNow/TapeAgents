import base64
import json
import os
from typing import Literal

from pydantic import Field

from tapeagents.core import Action, Observation, Thought

################# Actions #################


class WatchVideoAction(Action):
    """
    Action that loads the video from the provided url and returns the subtitle and its contact sheet from start_time to end_time.
    """

    kind: Literal["watch_video_action"] = "watch_video_action"
    video_url: str = Field(description="url of the video to watch")
    start_time: str = Field(
        description="time of the video to start watching from, if applicable, otherwise empty str. Format is HH:MM:SS.mmm",
        default="",
    )
    end_time: str = Field(
        description="time of the video to stop watching at, if applicable, otherwise empty str. Format is 'HH:MM:SS.mmm'",
        default="",
    )


################### Observations ###################


class ImageObservation(Observation):
    kind: Literal["image_observation"] = "image_observation"
    image_path: str
    thumbnail_path: str | None = None
    image_caption: str | None = None
    error: int | None = None

    def llm_view(self) -> list[dict]:
        llm_view = []
        if self.image_caption:
            llm_view.append[{"type": "text", "text": self.image_caption}]
        if self.image_path:
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

    def llm_view(self) -> list[dict]:
        llm_view = []
        if self.subtitle_text:
            llm_view.append({"type": "text", "text": self.subtitle_text})
        if self.video_contact_sheet_paths:
            for path in self.video_contact_sheet_paths:
                llm_view.append(image_base64_message(path))
        return llm_view


################### Thoughts ###################


class WatchingVideoThought(Thought):
    """
    Thought that outputs the detailed description of the video
    """

    kind: Literal["watching_video_thought"] = "watching_video_thought"
    description: str = Field(description="description of the video using both the subtitle and the video contact sheet")
    frame_description: list[str] = Field(
        description="detailed and specific description of each frame of the video contact sheet"
    )


################### Utils ###################


def image_base64_message(image_path: str) -> dict:
    _, image_extension = os.path.splitext(image_path)
    content_type = f"image/{image_extension}"
    base64_image = encode_image(image_path)
    message = {"type": "image_url", "image_url": {"url": f"data:{content_type};base64,{base64_image}"}}
    return message


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
