import logging
from typing import Annotated, Generator, Literal, TypeAlias, Union

from pydantic import Field

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


class Task(Observation):
    kind: Literal["task"] = "task"
    task: str

    def llm_view(self, indent: int | None = 2) -> str:
        # Same prompt as https://github.com/deepseek-ai/DeepSeek-Math/blob/b8b0f8ce093d80bf8e9a641e44142f06d092c305/evaluation/run_subset_parallel.py#L26
        return f"{self.task}\nPlease reason step by step, and put your final answer within " + "\\boxed{}."


class ReasoningThought(Thought):
    """
    Thoughts produced by the agent during the reasoning process.
    """

    kind: Literal["reasoning_thought_with_value"] = "reasoning_thought_with_value"
    reasoning: str = Field(description="chain of thoughts")


MathAgentStep: TypeAlias = Annotated[
    ReasoningThought,
    Field(discriminator="kind"),
]

RLMathTape = Tape[
    None,
    Union[
        Task,
        ReasoningThought,
        LLMOutputParsingFailureAction,
    ],
]


class ReasoningNode(MonoNode):
    def parse_completion(self, completion: str, prompt_id: str) -> Generator[Step, None, None]:
        try:
            step = ReasoningThought(reasoning=completion)
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
