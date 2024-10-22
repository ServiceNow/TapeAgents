import logging
import os
from typing import Annotated, Literal, TypeAlias, Union

from pydantic import Field

from tapeagents.agent import Agent
from tapeagents.core import Action, AgentResponseParsingFailureAction, FinalStep, Observation, Tape, Thought, TrainingText
from tapeagents.environment import Environment
from tapeagents.llms import LLM
from tapeagents.mono_agent import MonoAgent, MonoNode
from tapeagents.orchestrator import main_loop
from tapeagents.tools.calculator import calculate
from tapeagents.utils import get_step_schemas_from_union_type

logger = logging.getLogger(__name__)


#### Steps ####
class Task(Observation):
    kind: Literal["task"] = "task"
    task: str


class ReasoningThought(Thought):
    """
    Thoughts produced by the agent during the reasoning process.
    """

    kind: Literal["reasoning_thought"] = "reasoning_thought"
    reasoning: str = Field(description="chain of thoughts")


class UseCalculatorAction(Action):
    """
    Action to use calculator to find the new value. This math expression should be a single line. You can use exp, cos, sin, tan, abs, trunc, sgn, round
    """

    kind: Literal["use_calculator_action"] = "use_calculator_action"
    expression: str = Field(description="math expression to calculate")


class CalculationResultObservation(Observation):
    kind: Literal["calculation_result_observation"] = "calculation_result_observation"
    result: str


class ActionExecutionFailure(Observation):
    kind: Literal["action_execution_failure"] = "action_execution_failure"
    error: str


class AnswerAction(FinalStep):
    """
    Action that provides the final answer to the user after completing the task. Should be produced when the agent has finished the task.
    """

    kind: Literal["final_answer_action"] = "final_answer_action"
    text: str = Field(description="final answer to the user")
    value: int | float | None = Field(description="numerical value of the answer or null if solution is not found")


# FIXME: should not agent AgentResponseParsingFailureAction in steps
# But otherwise, it crashes _, llm_calls = self.reuse(tape)
MathAgentStep: TypeAlias = Annotated[
    Union[
        UseCalculatorAction,
        ReasoningThought,
        AnswerAction,
        AgentResponseParsingFailureAction
    ],
    Field(discriminator="kind"),
]

MathAgentStep2: TypeAlias = Annotated[
    Union[
        UseCalculatorAction,
        ReasoningThought,
        AnswerAction,
    ],
    Field(discriminator="kind"),
]

MathTape = Tape[
    None,
    Union[
        Task,
        ReasoningThought,
        UseCalculatorAction,
        CalculationResultObservation,
        ActionExecutionFailure,
        AgentResponseParsingFailureAction,
        AnswerAction,
    ],
]

#### Prompts ####

SYSTEM_PROMPT = """You are the genius math agent. Help user solve math problems.
Your role is to understand user queries and respond in a helpful and accurate manner.
Keep your replies concise and direct. Prioritize clarity and avoid over-elaboration.
"""




ALLOWED_STEPS = f"""
You are allowed to produce ONLY steps with the following json schemas:
{get_step_schemas_from_union_type(MathAgentStep2)}
Do not reproduce schema when producing the steps, use it as a reference.
"""

HINTS = """
Important considerations:
- Always produce only one step at a time.
- Step kind is always lowercase and underscore separated.
- DO NOT OUTPUT ANYTHING BESIDES A SINGLE JSON. It will break the system that processes the output.
"""

START_TASK_GUIDANCE = f"Let's think step by step using reasoning and calculations.\n{HINTS}"


#### Agent and Environment ####
class MathAgent(MonoAgent):
    @classmethod
    def create(cls, llm: LLM):
        return super().create(
            llm,
            nodes=[
                MonoNode(
                    name="start",
                    trigger_step="task",
                    system_prompt=SYSTEM_PROMPT,
                    steps_prompt=ALLOWED_STEPS,
                    agent_step_cls=MathAgentStep,
                    guidance=START_TASK_GUIDANCE,
                ),
                MonoNode(
                    name="default",
                    trigger_step="default",
                    system_prompt=SYSTEM_PROMPT,
                    steps_prompt=ALLOWED_STEPS,
                    agent_step_cls=MathAgentStep,
                    guidance=HINTS,
                ),
            ],
            max_iterations=2,
        )

    def make_training_data(self, tape: Tape) -> list[TrainingText]:
        """
        We only train on the last completion
        """
        logger.info(f"Making training data for tape: {tape.metadata.id}")
        _, llm_calls = self.reuse(tape)
        return [self.make_training_text(llm_call, compute_log_probs=True) for llm_call in llm_calls]


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
