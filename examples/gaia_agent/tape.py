from typing import Any

from pydantic import Field

from tapeagents.core import Tape, TapeMetadata
from tapeagents.dialog import DialogContext
from tapeagents.utils import _value_from_str

from .steps import (
    CalculationResultObservation,
    CodeResultObservation,
    FinishSubtask,
    GaiaStep,
    ListOfFactsThought,
    NewFactThought,
    PlanThought,
    StartSubtask,
)


class GaiaMetadata(TapeMetadata):
    task: dict = Field(default_factory=dict)
    result: Any = None
    attempt_number: int = 0


class GaiaTape(Tape[DialogContext, GaiaStep]):
    metadata: GaiaMetadata = GaiaMetadata()
    context: DialogContext = DialogContext(tools=[])

    def next_subtask(self) -> str:
        plan = self.plan()
        started_tasks = []
        finished_tasks = []
        for step in self.steps:
            if isinstance(step, FinishSubtask):
                finished_tasks.append(step.plan_step)
            elif isinstance(step, StartSubtask):
                started_tasks.append(step.plan_step)
        next_tasks = [task for task in plan if task not in started_tasks]
        return next_tasks[0] if next_tasks else ""

    def current_subtask(self) -> str:
        started_tasks = []
        finished_tasks = []
        for step in self.steps:
            if isinstance(step, FinishSubtask):
                finished_tasks.append(step.plan_step)
            elif isinstance(step, StartSubtask):
                started_tasks.append(step.plan_step)
        not_finished = [task for task in started_tasks if task not in finished_tasks]
        return not_finished[0] if not_finished else ""

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
                        facts[fact.name] = _value_from_str(fact.value)
            elif isinstance(step, NewFactThought):
                facts[step.fact_name] = _value_from_str(step.value)
            elif isinstance(step, CalculationResultObservation):
                facts[step.name] = _value_from_str(step.result)
            elif isinstance(step, CodeResultObservation):
                facts[step.name] = _value_from_str(step.result)
        return facts
