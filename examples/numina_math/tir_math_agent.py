import logging
from typing import Any, Generator, Literal, Union

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
from tapeagents.steps import (
    ActionExecutionFailure,
)
from tapeagents.llms import LLM
from tapeagents.nodes import MonoNode
from tapeagents.environment import Environment
from tapeagents.tools.code_executor import PythonCodeAction
from tapeagents.tools.calculator import CalculationResultObservation

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
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # the tape is only step long and it is the task
        task = tape.steps[0]
        assert isinstance(task, Task), f"Expected a Task, got {task.__class__.__name__}"
        messages.append({"role": "user", "content": task.llm_view()})
        prompt_token_ids = agent.llm.tokenizer.apply_chat_template(
            messages, add_special_tokens=True, add_generation_prompt=True
        )
        prompt_token_ids = prompt_token_ids[-self.max_prompt_length :]
        return Prompt(messages=messages, token_ids=prompt_token_ids)


class TIRMathAgent(Agent):
    @classmethod
    def create(cls, system_prompt: str, llm: LLM, max_prompt_length: int):
        agent = super().create(
            llm,
            nodes=[
                ReasoningNode(
                    name="tir",
                    agent_steps=ReasoningThought,
                    system_prompt=system_prompt,
                    max_prompt_length=max_prompt_length,
                ),
            ],
            max_iterations=1,
        )
        agent.store_llm_calls = True
        if agent.llm.tokenizer is None:
            agent.llm.load_tokenizer()
        return agent


class MathEnvironment(Environment):
    def __init__(self) -> None:
        super().__init__()

    def react(self, tape: MathTape) -> MathTape:
        actions = [step for step in tape.steps[-tape.metadata.n_added_steps :] if isinstance(step, Action)]
        for action in actions:
            if isinstance(action, LLMOutputParsingFailureAction):
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


TIRMathTape = Tape[
    None,
    Union[
        Task,
        ReasoningThought,
        PythonCodeAction,
        CalculationResultObservation,
        ActionExecutionFailure,
        LLMOutputParsingFailureAction,
    ],
]