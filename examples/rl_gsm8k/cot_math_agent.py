import logging
from typing import Annotated, Generator, Literal, TypeAlias, Union

from pydantic import Field

from tapeagents.agent import Agent
from tapeagents.core import (
    LLMOutputParsingFailureAction,
    Observation,
    Step,
    Tape,
    Thought,
)
from tapeagents.llms import LLM
from tapeagents.nodes import MonoNode

logger = logging.getLogger(__name__)


class Task(Observation):
    kind: Literal["task"] = "task"
    task: str
    template: str = Field(
        description="Template for the task. Should contain a {task} placeholder for the task text.", default="{task}"
    )

    def llm_view(self, indent: int | None = 2) -> str:
        return self.template.format(task=self.task)


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
    trim_tape_when_too_long: bool = False

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
    def create(cls, system_prompt: str, llm: LLM):
        agent = super().create(
            llm,
            nodes=[
                ReasoningNode(
                    name="cot",
                    agent_step_cls=MathAgentStep,
                    system_prompt=system_prompt if system_prompt else "",
                ),
            ],
            max_iterations=1,
        )
        agent.store_llm_calls = True
        return agent
