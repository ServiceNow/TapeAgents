import logging
import os
from typing import Any, Literal

from pydantic import Field

from tapeagents.core import Action, Observation, Step
from tapeagents.utils import get_relative_path_to_root, image_base64_message

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
    local_dir: str
    video_path: str
    video_contact_sheet_paths: list[str]
    thumbnail_path: str | None = None
    subtitle_path: str | None = None
    subtitle_text: str | None = None
    error: int | None = None

    def llm_dict(self) -> dict[str, Any]:
        normalized_observation = normalize_step_paths(self.model_copy(), self.local_dir)
        return normalized_observation.model_dump(exclude_none=True, exclude={"metadata", "local_dir"})

    def llm_view(self) -> list[dict]:
        llm_view = []
        if self.subtitle_text:
            llm_view.append({"type": "text", "text": self.subtitle_text})
        if self.video_contact_sheet_paths:
            for path in self.video_contact_sheet_paths:
                llm_view.append(image_base64_message(path))
        return llm_view


def normalize_step_paths(step: Step, root_path: str) -> Step:
    """
    Normalizes the paths in the step to be relative to the root_path folder.
    This is needed for tape replay from different environment paths.
    This also prevents prompts to contain full paths, which can be sensitive.
    """
    s = step.model_copy()
    for key, _ in s.model_fields.items():
        if key.endswith(("_path", "_paths")):
            value = getattr(s, key)
            if isinstance(value, list):
                setattr(s, key, [get_relative_path_to_root(path, root_path) for path in value])
            elif isinstance(value, str):
                setattr(s, key, get_relative_path_to_root(value, root_path))
            else:
                raise ValueError(f"Expected a list or string, got {type(value)}")
    return s


def update_step_paths(step: Step, dir: str) -> Step:
    """
    Updates the paths in the step to be relative to the provided directory.

    Parameters:
    step (Step): The step object containing paths to be updated.
    dir (str): The directory to which the paths should be made relative.

    Returns:
    Step: The updated step object with paths relative to the provided directory.
    """
    s = step.model_copy()
    if hasattr(s, "local_dir"):
        s.local_dir = dir
    for key, _ in s.model_fields.items():
        if key.endswith(("_path", "_paths")):
            value = getattr(s, key)
            if isinstance(value, list):
                setattr(s, key, [os.path.join(dir, os.path.basename(path)) for path in value])
            elif isinstance(value, str):
                setattr(s, key, os.path.join(dir, os.path.basename(value)))
            else:
                raise ValueError(f"Expected a list or string, got {type(value)}")
    return s


class UnknownStep(Step):
    content: str
    kind: Literal["unknown"] = "unknown"  # type: ignore


class Annotation(Action):
    kind: Literal["annotation"] = "annotation"
    step: int
    text: str
