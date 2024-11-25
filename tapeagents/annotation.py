from typing import Literal
from tapeagents.core import Action


class Annotation(Action):
    kind: Literal["annotation"] = "annotation"
    step: int
    text: str