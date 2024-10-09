import logging
import os
from typing import Annotated, Any, Literal, TypeAlias, Union

from pydantic import Field

from tapeagents.agent import Agent
from tapeagents.core import Action, AgentResponseParsingFailureAction, FinalStep, Observation, Tape, Thought
from tapeagents.environment import Environment
from tapeagents.guided_agent import GuidanceNode, GuidedAgent
from tapeagents.runtime import main_loop
from tapeagents.tools.calculator import calculate
from tapeagents.utils import get_step_schemas_from_union_type


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


MathAgentStep: TypeAlias = Annotated[
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
{get_step_schemas_from_union_type(MathAgentStep)}
Do not reproduce schema when producing the steps, use it as a reference.
"""

HINTS = """
Important considerations:
- Always produce only one step at a time.
- Step kind is always lowercase and underscore separated.
- DO NOT OUTPUT ANYTHING BESIDES THE JSON. It will break the system that processes the output.
"""

START_TASK_GUIDANCE = f"Let's think step by step using reasoning and calculations.\n{HINTS}"


class PromptRegistry:
    system_prompt = SYSTEM_PROMPT
    allowed_steps = ALLOWED_STEPS
    task_guidance = START_TASK_GUIDANCE
    hints = HINTS