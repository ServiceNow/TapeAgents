import logging
import os
from typing import Annotated, Any, Generator, Literal, Type, TypeAlias, Union

from pydantic import Field, TypeAdapter, ValidationError
from tapeagents.environment import Environment
from examples.gsm8k_tuning.math_agent import (
    ActionExecutionFailure,
    AnswerAction,
    Task,
    extract_result_value,
)
from tapeagents.agent import Agent
from tapeagents.core import (
    Action,
    FinalStep,
    LLMOutputParsingFailureAction,
    Observation,
    SetNextNode,
    Step,
    Tape,
    Thought,
)
from tapeagents.llms import LLM
from tapeagents.nodes import MonoNode
from tapeagents.utils import get_step_schemas_from_union_type

logger = logging.getLogger(__name__)


class ReasoningThoughtwithValue(Thought):
    """
    Thoughts produced by the agent during the reasoning process.
    """

    kind: Literal["reasoning_thought_with_value"] = "reasoning_thought_with_value"
    reasoning: str = Field(description="chain of thoughts")
    value: float = Field(description="value of the reasoning")


MathAgentStep: TypeAlias = Annotated[
    Union[
        ReasoningThoughtwithValue,
        AnswerAction,
    ],
    Field(discriminator="kind"),
]

MathTape = Tape[
    None,
    Union[
        Task,
        ReasoningThoughtwithValue,
        ActionExecutionFailure,
        LLMOutputParsingFailureAction,
        SetNextNode,
    ],
]

#### Prompts ####

SYSTEM_PROMPT = ""

START_TASK_GUIDANCE = ""
STEP_PROMPT = ""
COT_GUIDANCE = "Think step by step. When you know the answer to the question, provide it in the following format: The answer is: <number>"


class ReasoningNode(MonoNode):
    def parse_completion(self, completion: str, prompt_id: str) -> Generator[Step, None, None]:
        try:
            value = completion.split("The answer is:")[-1]
            value = value.replace(",", "")
            value = value.strip().strip("\n").strip("$").strip("€")
            step = ReasoningThoughtwithValue(reasoning=completion, value=float(value))
        except Exception as e:
            logger.info(f"Failed to parse agent output: {completion}\n\nError: {e}")
            yield LLMOutputParsingFailureAction(error=f"Failed to parse agent output: {completion}\n\nError: {e}")
            return
        yield step


#### Agent and Environment ####
class COTMathAgent(Agent):
    @classmethod
    def create(cls, llm: LLM):
        return super().create(
            llm,
            nodes=[
                ReasoningNode(
                    name="cot",
                    system_prompt=SYSTEM_PROMPT,
                    steps_prompt=STEP_PROMPT,
                    agent_step_cls=MathAgentStep,
                    guidance=COT_GUIDANCE,
                    next_node=-1,
                ),
            ],
            max_iterations=1,
        )

class MathEnvironment(Environment):
    def __init__(self) -> None:
        super().__init__()

    def react(self, tape: MathTape) -> MathTape:
        actions = [step for step in tape.steps[-tape.metadata.n_added_steps :] if isinstance(step, Action)]
        for action in actions:
            if isinstance(action, LLMOutputParsingFailureAction):
                continue
        return tape