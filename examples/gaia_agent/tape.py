from typing import Any

from pydantic import Field

from tapeagents.core import Tape, TapeMetadata
from tapeagents.dialog_tape import DialogContext
from tapeagents.utils import json_value_from_str

from .steps import (
    CalculationResultObservation,
    CodeResultObservation,
    GaiaStep,
    ListOfFactsThought,
    NewFactThought,
    PlanThought,
)


class GaiaMetadata(TapeMetadata):
    task: dict = Field(default_factory=dict)
    result: Any = None
    terminated: bool = False
    attempt_number: int = 0
    level: int = 0


class GaiaTape(Tape[DialogContext, GaiaStep]):
    metadata: GaiaMetadata = GaiaMetadata()
    context: DialogContext = DialogContext(tools=[])

    def has_fact_schemas(self) -> bool:
        for step in self.steps:
            if isinstance(step, ListOfFactsThought):
                return True
        return False

    def plan(self) -> list[str]:
        for step in reversed(self.steps):
            if isinstance(step, (PlanThought)):
                return step.plan
        return []

    def facts(self) -> dict[str, Any]:
        facts = {}
        for step in self.steps:
            if isinstance(step, ListOfFactsThought):
                for fact in step.given_facts:
                    if fact.value != "" and fact.value is not None:
                        facts[fact.name] = json_value_from_str(fact.value)
            elif isinstance(step, NewFactThought):
                facts[step.fact_name] = json_value_from_str(step.value)
            elif isinstance(step, CalculationResultObservation):
                facts[step.name] = json_value_from_str(step.result)
            elif isinstance(step, CodeResultObservation):
                facts[step.name] = json_value_from_str(step.result)
        return facts
