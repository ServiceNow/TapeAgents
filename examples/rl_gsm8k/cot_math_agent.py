import logging
from typing import Annotated, Any, Generator, Literal, TypeAlias, Union

from pydantic import Field

from tapeagents.agent import Agent
from tapeagents.core import (
    LLMOutputParsingFailureAction,
    Observation,
    Prompt,
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
    max_prompt_length: int = 1024

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

    def make_prompt(self, agent: Any, tape: Tape) -> Prompt:
        """Create a prompt from tape interactions.

        This method constructs a prompt by processing the tape content and agent steps description
        into a format suitable for LLM consumption. It includes token count checks and tape trimming
        if needed to fit within context size limits.

        Args:
            agent (Any): The agent object containing LLM configuration.
            tape (Tape): The tape object containing interaction history.

        Returns:
            Prompt: A Prompt object containing formatted messages for LLM consumption.

        Note:
            The method performs the following steps:

            1. Cleans the tape content
            2. Gets steps description
            3. Converts tape to messages
            4. Checks token count and trims if needed
            5. Reconstructs messages if trimming occurred
        """
        cleaned_tape = self.prepare_tape(tape)
        steps_description = self.get_steps_description(tape, agent)
        messages = self.tape_to_messages(cleaned_tape, steps_description)
        messages = self.tape_to_messages(cleaned_tape, steps_description)
        agent.llm.load_tokenizer()
        prompt_token_ids = agent.llm.tokenizer.apply_chat_template(messages, add_special_tokens=True, add_generation_prompt=True)
        prompt_token_ids = prompt_token_ids[-self.max_prompt_length:]
        return Prompt(messages=messages, token_ids=prompt_token_ids)


#### Agent and Environment ####
class CoTMathAgent(Agent):
    @classmethod
    def create(cls, system_prompt: str, llm: LLM, max_prompt_length: int):
        agent = super().create(
            llm,
            nodes=[
                ReasoningNode(
                    name="cot",
                    agent_step_cls=MathAgentStep,
                    system_prompt=system_prompt if system_prompt else "",
                    max_prompt_length=max_prompt_length,
                ),
            ],
            max_iterations=1,
        )
        agent.store_llm_calls = True
        return agent
