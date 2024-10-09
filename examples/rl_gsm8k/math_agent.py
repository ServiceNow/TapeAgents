import logging
import os
from typing import Annotated, Any, Literal, TypeAlias, Union

from pydantic import Field

from tapeagents.agent import Agent
from tapeagents.core import Action, AgentResponseParsingFailureAction, FinalStep, Observation, Tape, Thought
from tapeagents.environment import Environment
from tapeagents.guided_agent import GuidanceNode, GuidedAgent
from tapeagents.llms import LLM
from tapeagents.runtime import main_loop
from tapeagents.tools.calculator import calculate
from tapeagents.utils import get_step_schemas_from_union_type

from .prompts import (
    ActionExecutionFailure,
    AnswerAction,
    CalculationResultObservation,
    MathAgentStep,
    MathTape,
    PromptRegistry,
    Task,
    UseCalculatorAction,
)

logger = logging.getLogger(__name__)


class MathNode(GuidanceNode):
    system_prompt: str = PromptRegistry.system_prompt
    allowed_steps: str = PromptRegistry.allowed_steps
    start_step_cls: Any = Task
    agent_step_cls: Any = MathAgentStep


#### Agent and Environment ####
class MathAgent(GuidedAgent):
    @classmethod
    def create(cls, llm: LLM):
        return super().create(
            llm,
            nodes=[
                MathNode(
                    name="Task",
                    trigger_step="Task",
                    guidance=PromptRegistry.task_guidance,
                ),
                MathNode(
                    name="Thought",
                    trigger_step=["Thought", "CalculationResultObservation", "ActionExecutionFailure"],
                    guidance=PromptRegistry.hints,
                    steps_prompt=PromptRegistry.allowed_steps,
                ),
            ],
            max_iterations=2,
        )


class MathEnvironment(Environment):
    def __init__(self) -> None:
        super().__init__()

    def react(self, tape: MathTape) -> MathTape:
        actions = [step for step in tape.steps[-tape.metadata.n_added_steps :] if isinstance(step, Action)]
        for action in actions:
            if isinstance(action, AgentResponseParsingFailureAction):
                continue
            try:
                match action:
                    case UseCalculatorAction():
                        observation = CalculationResultObservation(result=calculate(action.expression, {}))
                        tape = tape.append(observation)
                    case _:
                        raise Exception(f"Unknown action: {type(action)}")
            except Exception as e:
                tape = tape.append(ActionExecutionFailure(error=str(e)))
                break
        return tape


def save_tape(tape_path, tape):
    with open(tape_path, "w") as f:
        f.write(tape.model_dump_json(indent=4))


def solve_task(agent: Agent, env: Environment, task: dict, tape_file: str = "") -> Tape:
    tmp_tape_file = f"{tape_file}.tmp" if tape_file else None
    start_step = Task(task=task["question"])
    tape = MathTape(steps=[start_step], context=None)
    metadata = task.copy()

    for event in main_loop(agent, tape, env, max_loops=30):
        step = None
        if event.agent_event and event.agent_event.step:
            step = event.agent_event.step
        elif event.observation:
            step = event.observation
        if step:
            tape = tape.append(step)  # type: ignore
            if tmp_tape_file:
                save_tape(tmp_tape_file, tape)
    if tmp_tape_file:
        os.unlink(tmp_tape_file)  # remove temporary file
    metadata["solved"] = False
    if isinstance(tape[-1], AnswerAction) and task["value"] == tape[-1].value:  # type: ignore
        metadata["solved"] = True
    tape.metadata.result = metadata
    return tape


def extract_result_value(sample) -> dict:
    sample = dict(sample)
    expected_result = str(sample["answer"]).rsplit("####", maxsplit=1)[-1].strip().replace(",", "")
    if expected_result.isdigit():
        expected_result = int(expected_result)
    else:
        expected_result = float(expected_result)
    sample["value"] = expected_result
    return sample
