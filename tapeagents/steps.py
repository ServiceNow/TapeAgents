import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from tapeagents.core import Action, Observation, Step
from tapeagents.utils import image_base64_message

logger = logging.getLogger(__name__)

################# Actions #################


class WatchVideoAction(Action):
    """
    Action that loads the video from the provided url and returns the video content.
    """

    kind: Literal["watch_video_action"] = "watch_video_action"
    video_url: str = Field(description="url of the video")
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
    kind: Literal["image"] = "image"
    image_path: str
    thumbnail_path: str = ""
    image_caption: str = ""
    error: int | None = None

    def llm_view(self) -> list[dict]:
        content = []
        if self.image_caption:
            content.append({"type": "text", "text": self.image_caption})
        if self.image_path:
            content.append(image_base64_message(self.image_path))
        return content


class VideoObservation(Observation):
    """
    Observation that contains a video, thumbnail, subtitle and its contact sheet.
    """

    kind: Literal["video_observation"] = "video_observation"
    attachment_dir: str
    video_path: str
    video_contact_sheet_paths: list[str]
    thumbnail_path: str | None = None
    subtitle_path: str | None = None
    subtitle_text: str | None = None
    error: int | None = None

    def llm_dict(self) -> dict[str, Any]:
        # exclude attachment_dir as we don't want to send sensitive information to LLM
        return self.model_dump(exclude_none=True, exclude={"metadata", "attachment_dir"})

    def llm_view(self) -> list[dict]:
        llm_view = []
        if self.subtitle_text:
            llm_view.append({"type": "text", "text": self.subtitle_text})
        if self.video_contact_sheet_paths:
            for path in self.video_contact_sheet_paths:
                llm_view.append(image_base64_message(Path(self.attachment_dir) / path))
        return llm_view


class UnknownStep(Step):
    content: str
    kind: Literal["unknown"] = "unknown"  # type: ignore


class Annotation(Action):
    kind: Literal["annotation"] = "annotation"
    step: int
    text: str
