from typing import Any

from pydantic import Field

from tapeagents.core import Tape, TapeMetadata
from tapeagents.dialog_tape import DialogContext

from .steps import GaiaStep


class GaiaMetadata(TapeMetadata):
    task: dict = Field(default_factory=dict)
    result: Any = None
    terminated: bool = False
    attempt_number: int = 0
    level: int = 0
    other: dict = Field(default_factory=dict)


class GaiaTape(Tape[DialogContext, GaiaStep]):
    metadata: GaiaMetadata = Field(default_factory=GaiaMetadata)
    context: DialogContext | None = DialogContext(tools=[])
