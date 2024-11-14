import logging
import os
from typing import Annotated, Literal, TypeAlias, Union

from pydantic import Field
from typing import Any, Generator, Type

from pydantic import Field, TypeAdapter, ValidationError

from examples.gsm8k_tuning.math_agent import (
    AnswerAction,
    Task,
    extract_result_value,
    ActionExecutionFailure,
    MathEnvironment,
    ReasoningThought,
)
from tapeagents.agent import Agent
from tapeagents.core import (
    Action,
    FinalStep,
    LLMOutputParsingFailureAction,
    Observation,
    SetNextNode,
    Tape,
    Thought,
    Step,
)
from tapeagents.llms import LLM
from tapeagents.nodes import MonoNode
from tapeagents.utils import get_step_schemas_from_union_type

logger = logging.getLogger(__name__)

MathAgentStep: TypeAlias = Annotated[
    Union[
        ReasoningThought,
        AnswerAction,
    ],
    Field(discriminator="kind"),
]

AnswerStep: TypeAlias = Annotated[
    Union[
        AnswerAction,
    ],
    Field(discriminator="kind"),
]

MathTape = Tape[
    None,
    Union[
        Task,
        ReasoningThought,
        AnswerAction,
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


class ReasoningThoughtwithValue(Thought):
    """
    Thoughts produced by the agent during the reasoning process.
    """

    kind: Literal["reasoning_thought"] = "reasoning_thought"
    reasoning: str = Field(description="chain of thoughts")
    value: float = Field(description="value of the reasoning")


class ReasoningNode(MonoNode):
    def parse_completion(self, completion: str, prompt_id: str) -> Generator[Step, None, None]:
        try:
            value = completion.split("The answer is:")[-1]
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
                ),
            ],
        )
