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

# https://github.com/PRIME-RL/PRIME/blob/49a58a8e4afd464f559f8d9f80418052f29cf3e4/eval/system_prompt.md?plain=1
# but note that sometimes they do not include the newline at the beginning
# https://github.com/PRIME-RL/PRIME/blob/49a58a8e4afd464f559f8d9f80418052f29cf3e4/data_preprocessing/sft_prompt.py#L1
EURUS_SYSTEM_PROMPT = """\nWhen tackling complex reasoning tasks, you have access to the following actions. Use them as needed to progress through your thought process.\n\n[ASSESS]\n\n[ADVANCE]\n\n[VERIFY]\n\n[SIMPLIFY]\n\n[SYNTHESIZE]\n\n[PIVOT]\n\n[OUTPUT]\n\nYou should strictly follow the format below:\n\n[ACTION NAME]\n\n# Your action step 1\n\n# Your action step 2\n\n# Your action step 3\n\n...\n\nNext action: [NEXT ACTION NAME]\n\n"""

class Task(Observation):
    kind: Literal["task"] = "task"
    task: str

    def llm_view(self, indent: int | None = 2) -> str:
        # Same prompt as https://github.com/deepseek-ai/DeepSeek-Math/blob/b8b0f8ce093d80bf8e9a641e44142f06d092c305/evaluation/run_subset_parallel.py#L26
        #return f"{self.task}\nPlease reason step by step, and put your final answer within " + "\\boxed{}."
        return self.task


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
        agent = super().create(
            llm,
            nodes=[
                ReasoningNode(
                    name="cot",
                    agent_step_cls=MathAgentStep,
                    system_prompt=EURUS_SYSTEM_PROMPT,
                ),
            ],
            max_iterations=1,
        )
        agent.store_llm_calls = True
        return agent