import datetime
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from statistics import mean
from typing import Any, Generator

import numpy as np
from Levenshtein import ratio
from pydantic import BaseModel

from tapeagents.core import LLMCall, LLMOutput, Prompt, TrainingText
from tapeagents.observe import observe_llm_call


class LLMEvent(BaseModel):
    """An event class representing either a chunk of LLM output or the final LLM output.

    This class encapsulates events that occur during LLM processing, handling both
    intermediate chunks of output and the final complete output.

    Attributes:
        chunk (str, optional): A partial text output from the LLM stream.
        output (LLMOutput, optional): The complete output from the LLM.
        llm_call (LLMCall, optional): The entire LLMCall object.
    """

    chunk: str | None = None
    output: LLMOutput | None = None
    llm_call: LLMCall | None = None


class LLMStream:
    """A wrapper class for LLM generators that provides convenient iteration and output extraction.

    This class wraps a generator that yields LLMEvents and provides methods to:

    - Iterate through events
    - Extract complete LLM output
    - Get the assistant's response text

    LLMStream stores the LLM call object when the generator yields it.

    Attributes:
        generator: Generator yielding LLMEvents or None if empty
        prompt: The prompt used to generate the LLM response:

    Raises:
        ValueError: When trying to iterate null stream, when no output is produced,
                   or when output is not an assistant message with content
    """

    def __init__(self, generator: Generator[LLMEvent, None, None] | None, prompt: Prompt):
        self.generator = generator
        self.prompt = prompt

    def __bool__(self):
        return self.generator is not None

    def __iter__(self):
        if self.generator is None:
            raise ValueError("can't iterate a null stream")
        return self

    def __next__(self) -> LLMEvent:
        if self.generator is None:
            raise StopIteration
        event = next(self.generator)
        if event.llm_call:
            self.llm_call = event.llm_call
        return event

    def get_output(self) -> LLMOutput:
        """Returns first LLMOutput found in events"""
        for event in self:
            if event.output:
                return event.output
        raise ValueError("LLM did not produce an output")

    def get_text(self) -> str:
        """Returns content of first assistant message found"""
        o = self.get_output()
        if not o.role == "assistant" or o.content is None:
            raise ValueError("LLM did not produce an assistant message")
        return o.content

    def get_llm_call(self) -> LLMCall:
        """Returns the LLMCall object"""
        for event in self:
            if event.llm_call:
                break
        return self.llm_call


class LLM(BaseModel, ABC):
    """
    An abstract base class representing a Language Learning Model (LLM).

    This class defines the interface for interacting with different LLM implementations.
    It handles basic LLM functionality like token counting, generation, and logging.

    Attributes:
        model_name (str): Name of the LLM model
        parameters (dict): Model-specific parameters for generation
        context_size (int): Maximum context size in tokens (default: 32000)
        tokenizer_name (str): Name of the tokenizer used
        tokenizer (Any): Tokenizer instance
        token_count (int): Running count of tokens processed
        observe_llm_calls (bool): Flag to enable observation of LLM calls


    Note:
        This is an abstract class and requires implementation of the abstract methods
        in derived classes.
    """

    model_name: str
    parameters: dict = {}
    context_size: int = 32000
    tokenizer_name: str = ""
    tokenizer: Any = None
    observe_llm_calls: bool = True

    token_count: int = 0
    _log: list = []
    _stats: dict = defaultdict(list)

    @abstractmethod
    def generate(self, prompt: Prompt, **kwargs) -> LLMStream:
        """
        Generate text from a given prompt

        Args:
            prompt (Prompt): The prompt object containing messages to send to the LLM.
            **kwargs (dict, optional): Additional arguments to pass to the underlying LLM implementation.

        Returns:
            LLMStream: A stream of LLM events containing the model's response.
        """
        pass

    @abstractmethod
    def count_tokens(self, messages: list[dict] | str) -> int:
        """
        Count tokens in messages or text

        Args:
            messages (Union[List[Dict], str]): List of messages or text to count tokens in

        Returns:
            int: Number of tokens in the messages or text
        """
        pass

    @abstractmethod
    def make_training_text(self, prompt: Prompt, output: LLMOutput) -> TrainingText:
        """
        Create training text from prompt and output.

        Args:
            prompt (Prompt): The prompt object containing messages used to generate the output.
            output (LLMOutput): The output generated by the LLM.

        Returns:
            TrainingText: The training text object containing the prompt and output.
        """
        pass

    def get_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "parameters": self.parameters,
            "context_size": self.context_size,
        }

    def get_token_costs(self) -> dict:
        """Returns prices for different kinds of tokens.

        See `result['input']` for the price of input tokens and
        `result['output']` for the price of output tokens respectively.
        """
        return {"input": 0, "output": 0}

    def log_output(
        self, prompt: Prompt, message: LLMOutput, cached: bool = False, count_tokens: bool = True
    ) -> None | LLMCall:
        """
        Logs the output of an LLM (Language Model) call along with its metadata.

        Args:
            prompt (Prompt): The prompt object containing the input messages for the LLM.
            message (LLMOutput): The output message generated by the LLM.
            cached (bool, optional): Indicates whether the output was retrieved from cache. Defaults to False.
        """

        start_log_output = time.perf_counter()
        if count_tokens:
            prompt_length_tokens = self.count_tokens(prompt.messages)
            if message.content:
                output_length_tokens = (
                    self.count_tokens(prompt.messages + [{"role": "assistant", "content": message.content}])
                    - prompt_length_tokens
                )
            else:
                output_length_tokens = 0
            self._stats["prompt_length_tokens"].append(prompt_length_tokens)
            self._stats["output_length_tokens"].append(output_length_tokens)
        else:
            # -1 is the default value of prompt and output length tokens when token counting is disabled
            prompt_length_tokens = -1
            output_length_tokens = -1

        llm_call = LLMCall(
            timestamp=datetime.datetime.now().isoformat(),
            prompt=prompt,
            output=message,
            prompt_length_tokens=prompt_length_tokens,
            output_length_tokens=output_length_tokens,
            cached=cached,
            llm_info=self.get_info(),
        )
        token_costs = self.get_token_costs()
        llm_call.cost = (
            token_costs["input"] * llm_call.prompt_length_tokens + token_costs["output"] * llm_call.output_length_tokens
        )
        self._log.append(llm_call.model_dump())
        if self.observe_llm_calls:
            observe_llm_call(llm_call)
        time_log_output = time.perf_counter() - start_log_output
        self._stats["time_log_output"].append(time_log_output)
        return llm_call

    def get_stats(self) -> dict:
        return {
            "time_send_request": mean(self._stats["time_send_request"]) if self._stats["time_send_request"] else 0,
            "time_log_output": mean(self._stats["time_log_output"]) if self._stats["time_log_output"] else 0,
            "total_prompt_tokens": sum(self._stats["prompt_length_tokens"])
            if self._stats["prompt_length_tokens"]
            else 0,
            "total_output_tokens": sum(self._stats["output_length_tokens"])
            if self._stats["output_length_tokens"]
            else 0,
            "time_postprocess_llm_response": np.mean(self._stats["time_postprocess_llm_response"])
            if self._stats["time_postprocess_llm_response"]
            else 0,
        }

    def get_step_schema(self, cls):
        raise NotImplementedError("get_step_schema method not implemented")


class NoTokenizerError(ValueError):
    pass


def closest_prompt(prompt_key: str, known_prompts: list[str]) -> tuple[str, float]:
    """
    Finds the closest matching prompt from a list of known prompts based on a Levenshtein similarity ratio.

    Args:
        prompt_key (str): The prompt to compare against the known prompts.
        known_prompts (list[str]): A list of known prompts to compare with the prompt_key.

    Returns:
        tuple[str, float]: A tuple containing the closest matching prompt and its similarity score.
                           If no prompts are found, returns an empty string and a score of 0.0.
    """
    ratios = [(k, ratio(prompt_key, k, score_cutoff=0.5)) for k in known_prompts]
    if not len(ratios):
        return "", 0.0
    ratios = sorted(ratios, key=lambda x: x[1], reverse=True)
    closest, score = sorted(ratios, key=lambda x: x[1], reverse=True)[0]
    return closest, score
