import time

from tapeagents.core import LLMOutput, Prompt, TrainingText
from tapeagents.llms.base import LLM, LLMEvent, LLMStream


class MockLLM(LLM):
    """A mock LLM implementation for testing purposes.

    This class simulates an LLM by returning predefined responses in a cyclic manner.
    It tracks the prompts it receives and maintains a call counter.

    Attributes:
        model_name (str): Name of the mock model, defaults to "mock"
        call_number (int): Counter for number of calls made to generate, defaults to 0
        mock_outputs (list[str]): List of predefined responses to cycle through
        prompts (list[Prompt]): List of received prompts
    """

    model_name: str = "mock"
    call_number: int = 0
    mock_outputs: list[str] = [
        "Agent: I'm good, thank you",
        "Agent: Sure, I worked at ServiceNow for 10 years",
        "Agent: I have 10 zillion parameters",
    ]
    prompts: list[Prompt] = []

    def generate(self, prompt: Prompt) -> LLMStream:
        def _implementation():
            self.prompts.append(prompt)
            output = self.mock_outputs[self.call_number % len(self.mock_outputs)]
            time.sleep(0.01)
            yield LLMEvent(output=LLMOutput(content=output))
            self.call_number += 1

        return LLMStream(_implementation(), prompt=prompt)

    def count_tokens(self, messages: list[dict] | str) -> int:
        return 42

    def make_training_text(self, prompt: Prompt, output: LLMOutput) -> TrainingText:
        return TrainingText(text="mock trace", n_predicted=10)
