from typing import Literal
from tapeagents.core import Action


class HumanAnnotation(Action):
    kind: Literal["human_annotation"] = "human_annotation"
    step: int
    text: str