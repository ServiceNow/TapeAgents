import logging
from typing import Annotated, Generator, Literal, TypeAlias, Union

from pydantic import Field

from examples.gsm8k_tuning.math_agent import extract_result_value  # noqa
from tapeagents.agent import Agent
from tapeagents.core import (
    Action,
    LLMOutputParsingFailureAction,
    Observation,
    Step,
    Tape,
    Thought,
)
from tapeagents.environment import Environment
from tapeagents.llms import LLM
from tapeagents.nodes import MonoNode

logger = logging.getLogger(__name__)

COT_GUIDANCE = "Think step by step. When you know the answer to the question, provide it in the following format: The answer is: <number>"


class Task(Observation):
    kind: Literal["task"] = "task"
    task: str

    def llm_view(self, indent: int | None = 2) -> str:
        return f"{self.task} {COT_GUIDANCE}"


class ReasoningThoughtwithValue(Thought):
    """
    Thoughts produced by the agent during the reasoning process.
    """

    kind: Literal["reasoning_thought_with_value"] = "reasoning_thought_with_value"
    reasoning: str = Field(description="chain of thoughts")
    value: float = Field(description="value of the reasoning")


MathAgentStep: TypeAlias = Annotated[
    ReasoningThoughtwithValue,
    Field(discriminator="kind"),
]

RLMathTape = Tape[
    None,
    Union[
        Task,
        ReasoningThoughtwithValue,
        LLMOutputParsingFailureAction,
    ],
]


class ReasoningNode(MonoNode):
    def parse_completion(self, completion: str, prompt_id: str) -> Generator[Step, None, None]:
        if "The answer is" not in completion:
            yield LLMOutputParsingFailureAction(
                error=f"Failed to parse agent output: {completion}", llm_output=completion
            )
            return
        try:
            value = completion.split("The answer is")[-1]
            value = value.replace(",", "")
            value = value.replace(" ", "")
            value = value.replace(":", "")
            value = value.replace("$", "")
            value = value.replace("%", "")
            value = value.replace("€", "")
            value = value.strip()
            step = ReasoningThoughtwithValue(reasoning=completion, value=float(value))
        except Exception as e:
            logger.info(f"Failed to parse agent output: {completion}\n\nError: {e}")
            yield LLMOutputParsingFailureAction(
                error=f"Failed to parse agent output: {completion}\n\nError: {e}", llm_output=completion
            )
            return
        yield step


#### Agent and Environment ####
class CoTMathAgent(Agent):
    @classmethod
    def create(cls, llm: LLM):
        return super().create(
            llm,
            nodes=[
                ReasoningNode(
                    name="cot",
                    agent_step_cls=MathAgentStep,
                ),
            ],
            max_iterations=1,
        )


class MathEnvironment(Environment):
    def __init__(self) -> None:
        super().__init__()

    def react(self, tape: RLMathTape) -> RLMathTape:
        actions = [step for step in tape.steps[-tape.metadata.n_added_steps :] if isinstance(step, Action)]
        for action in actions:
            if isinstance(action, LLMOutputParsingFailureAction):
                continue
        return tape
