"""
Wrapper for interacting with external and hosted large language models (LLMs).
"""

from __future__ import annotations

from collections import defaultdict
import datetime
import hashlib
import json
import logging
import os
import random
import threading
import time
from abc import ABC, abstractmethod
from itertools import zip_longest
from typing import Any, Callable, Generator

import litellm
import openai
import requests
from Levenshtein import ratio
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from termcolor import colored
import numpy as np

from .config import DB_DEFAULT_FILENAME
from .core import LLMOutput, Prompt, TrainingText
from .observe import LLMCall, observe_llm_call, retrieve_all_llm_calls
from .utils import FatalError, diff_strings

requests.packages.urllib3.disable_warnings()  # type: ignore
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

TAPEAGENTS_LLM_TOKEN = "TAPEAGENTS_LLM_TOKEN"

cache_write_lock = threading.Lock()
transformers = None


class LLMEvent(BaseModel):
    """An event class representing either a chunk of LLM output or the final LLM output.

    This class encapsulates events that occur during LLM processing, handling both
    intermediate chunks of output and the final complete output.

    Attributes:
        chunk (str, optional): A partial text output from the LLM stream. None if this
            event represents a complete output.
        output (LLMOutput, optional): The complete output from the LLM. None if this
            event represents a partial chunk.
    """

    chunk: str | None = None
    output: LLMOutput | None = None


class LLMStream:
    """A wrapper class for LLM generators that provides convenient iteration and output extraction.

    This class wraps a generator that yields LLMEvents and provides methods to:

    - Iterate through events
    - Extract complete LLM output
    - Get the assistant's response text

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
        return next(self.generator)

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
        _log (list): Internal log of LLM calls

    Note:
        This is an abstract class and requires implementation of the abstract methods
        in derived classes.
    """

    model_name: str
    parameters: dict = {}
    context_size: int = 32000
    tokenizer_name: str = ""
    tokenizer: Any = None

    token_count: int = 0
    _log: list = []
    stats: dict = defaultdict(list)

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

    def log_output(self, prompt: Prompt, message: LLMOutput, cached: bool = False):
        """
        Logs the output of an LLM (Language Model) call along with its metadata.

        Args:
            prompt (Prompt): The prompt object containing the input messages for the LLM.
            message (LLMOutput): The output message generated by the LLM.
            cached (bool, optional): Indicates whether the output was retrieved from cache. Defaults to False.
        """

        start_log_output = time.time()
        prompt_length_tokens = self.count_tokens(prompt.messages)
        if message.content:
            # lstrip the left spaces from the assistant message because the chat template will add one
            output_length_tokens = (
                self.count_tokens(prompt.messages + [{"role": "assistant", "content": message.content}])
                - prompt_length_tokens
            )
        else:
            output_length_tokens = 0

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
        observe_llm_call(llm_call)
        time_log_output = time.time() - start_log_output
        self.stats["time_log_output"].append(time_log_output)

    def get_stats(self) -> dict:
        return {
            "time_send_request": np.mean(self.stats["time_send_request"]) if self.stats["time_send_request"] else 0,
            "time_log_output": np.mean(self.stats["time_log_output"]) if self.stats["time_log_output"] else 0,
        }


# Use this variable to force all LLMs to use cache from the sqlite DB
# This is meant to be used for testing purposes only
_REPLAY_SQLITE: str = ""
# force replacement of the tokenizer during testing
_MOCK_TOKENIZER: str = ""


class CachedLLM(LLM):
    """A caching wrapper for LLM implementations that stores and retrieves previous LLM responses.

    This class implements caching functionality for LLM responses to avoid redundant API calls
    and enable replay of previous interactions. It supports both file-based caching and SQLite-based
    replay functionality for testing purposes.

    Attributes:
        use_cache (bool): Flag to enable/disable caching functionality. Defaults to False.
        stream (bool): Flag to enable/disable streaming responses. Defaults to False.
        _cache (dict): Internal cache storage mapping prompt keys to LLM responses.

    The cache can be initialized in two modes:
    1. SQLite replay mode: Used for testing, enforces cache hits only
    2. File-based cache mode: Stores responses in a jsonl file for persistence

    Cache keys are generated based on the prompt content, excluding the prompt ID.
    During testing (replay mode), exact text matching is used instead of hashing.
    """

    use_cache: bool = False
    stream: bool = False
    _cache: dict = {}

    def model_post_init(self, __content):
        if _REPLAY_SQLITE:
            self.use_cache = True
            self._cache = {}
            llm_calls = retrieve_all_llm_calls(_REPLAY_SQLITE)
            for llm_call in llm_calls:
                key = self.get_prompt_key(llm_call.prompt)
                self._cache[key] = [LLMEvent(output=llm_call.output)]
            logger.info(f"Enforced LLM cache from {_REPLAY_SQLITE}, {len(self._cache)} entries")
            return
        elif not self.use_cache:
            return
        logger.info("Use LLM Cache")
        param_hash = self._key(json.dumps({k: v for k, v in self.parameters.items() if k != "token"}))
        name = self.model_name.replace("/", "__")
        self._cache_file = f"llm_cache_{name}_{param_hash}.jsonl"
        if os.path.exists(self._cache_file):
            with open(self._cache_file) as f:
                for line in f:
                    key, event_dict = json.loads(line)
                    if key not in self._cache:
                        self._cache[key] = []
                    self._cache[key].append(event_dict)
            logger.info(f"Loaded cache with {len(self._cache)} keys")
        else:
            logger.info("Cache file not found")

    def reindex_log(self):
        """
        Reindex the log data into cache.

        This method iterates through the log entries, validates each prompt and output,
        and adds them to the cache using the prompt key as index. Each entry is converted
        to an LLMEvent model before caching.

        Side Effects:
            - Updates the internal cache with log data
            - Logs the total number of reindexed entries at INFO level
        """
        cnt = 0
        for log_data in self._log:
            key = self.get_prompt_key(Prompt.model_validate(log_data["prompt"]))
            self._add_to_cache(key, LLMEvent(output=LLMOutput.model_validate(log_data["output"])).model_dump())
            cnt += 1
        logger.info(f"Reindexed {cnt} log entries")

    def _add_to_cache(self, key: str, event_dict: dict):
        if not self.use_cache:
            return
        with cache_write_lock:
            if key not in self._cache:
                self._cache[key] = []
            self._cache[key].append(event_dict)
            with open(self._cache_file, "a") as f:
                f.write(json.dumps((key, event_dict), ensure_ascii=False) + "\n")

    def get_prompt_key(self, prompt: Prompt) -> str:
        prompt_text = json.dumps(prompt.model_dump(exclude={"id"}), ensure_ascii=False, sort_keys=True)
        return self._key(prompt_text)

    def _key(self, text: str) -> str:
        if _REPLAY_SQLITE:
            # use exact text as a key during testing
            return text
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def generate(self, prompt: Prompt, **kwargs) -> LLMStream:
        """Generate a response stream from the language model based on the given prompt.

        This method handles both cached and new responses, implementing a caching mechanism
        for LLM responses to avoid redundant API calls.

        Args:
            prompt (Prompt): The prompt object containing messages to send to the LLM.
            **kwargs (dict, optional): Additional arguments to pass to the underlying LLM implementation.

        Returns:
            LLMStream: A stream of LLM events containing the model's response.

        Raises:
            ValueError: If cache miss occurs when replay mode is enabled (_REPLAY_SQLITE is True).

        Notes:
            - If caching is enabled and the prompt exists in cache, returns cached response
            - If generating new response, tokens are counted and added to total token count
            - All generated events are cached for future use if caching is enabled
            - Output is logged through the logging system
        """

        def _implementation():
            key = self.get_prompt_key(prompt)
            if self.use_cache and key in self._cache:
                logger.debug(colored(f"llm cache hit, {len(self._cache[key])} events", "green"))
                for event_dict in self._cache[key]:
                    event = LLMEvent.model_validate(event_dict)
                    if event.output is not None:
                        self.log_output(prompt, event.output, cached=True)
                    yield event
            else:
                if _REPLAY_SQLITE:
                    closest, score = closest_prompt(key, list(self._cache.keys()))
                    logger.error(
                        f"llm cache miss, closest in cache has score {score:.3f}\nDIFF:\n{diff_strings(key, closest)}"
                    )
                    raise ValueError(f"llm cache miss not allowed, prompt: {key}")
                toks = self.count_tokens(prompt.messages)
                self.token_count += toks
                logger.debug(f"{toks} prompt tokens, total: {self.token_count}")
                for event in self._generate(prompt, **kwargs):
                    self._add_to_cache(key, event.model_dump())
                    # note: the underlying LLM will log the output
                    yield event

        return LLMStream(_implementation(), prompt)

    @abstractmethod
    def _generate(self, prompt: Prompt, **kwargs) -> Generator[LLMEvent, None, None]:
        pass


class NoTokenizerError(ValueError):
    pass


class LiteLLM(CachedLLM):
    """A LiteLLM implementation of the LLM interface.

    This class provides integration with the LiteLLM library for making LLM API calls.
    It supports both streaming and non-streaming responses, token counting, and handles API timeouts with retries.
    Streaming responses are handled by yielding chunks of text as they arrive.
    Non-streaming responses return complete messages.

    Note:
        Function calling during streaming is not yet implemented and will raise NotImplementedError.
    """

    def count_tokens(self, messages: list[dict] | str) -> int:
        """
        Count the number of tokens in a message or string.

        Args:
            messages (Union[List[Dict], str]): List of messages or text to count tokens in.

        Returns:
            int: The number of tokens in the messages or text.
        """
        if isinstance(messages, str):
            return litellm.token_counter(model=self.model_name, text=messages)
        else:
            return litellm.token_counter(model=self.model_name, messages=messages)

    def get_token_costs(self):
        costs = litellm.model_cost.get(self.model_name)
        if costs is None:
            logger.info(f"Model {self.model_name} not found in the LiteLLM cost database")
            return {"input": 0, "output": 0}
        return {"input": costs["input_cost_per_token"], "output": costs["output_cost_per_token"]}

    def _generate(self, prompt: Prompt, **kwargs) -> Generator[LLMEvent, None, None]:
        while True:
            try:
                response = litellm.completion(
                    model=self.model_name,
                    messages=prompt.messages,
                    tools=prompt.tools,
                    stream=self.stream,
                    **self.parameters,
                )
                break
            except openai.APITimeoutError:
                logger.error("API Timeout, retrying in 1 sec")
                time.sleep(1.0)
        if self.stream:
            buffer = []
            for part in response:
                assert isinstance(part, litellm.ModelResponse)
                if isinstance(part.choices[0], litellm.utils.StreamingChoices):
                    content_delta = part.choices[0].delta.content
                    if content_delta:
                        buffer.append(content_delta)
                        yield LLMEvent(chunk=content_delta)
                    tool_delta = part.choices[0].delta.tool_calls
                    if tool_delta:
                        raise NotImplementedError("TODO: streaming with function calls not implemented yet")
                else:
                    raise ValueError(f"Unexpected response {part.model_dump()}")
            output = LLMOutput(content="".join(buffer))
        else:
            assert isinstance(response, litellm.ModelResponse)
            assert isinstance(response.choices[0], litellm.utils.Choices)
            output = response.choices[0].message
        self.log_output(prompt, output)
        yield LLMEvent(output=output)

    def make_training_text(self, *args, **kwargs) -> TrainingText:
        """
        Generates the training text for the model.

        This method should be implemented by subclasses to provide the specific
        logic for creating the training text.

        Args:
            *args (list): Variable length argument list.
            **kwargs (dict, optional): Arbitrary keyword arguments.

        Returns:
            TrainingText: The generated training text.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError()


class TrainableLLM(CachedLLM):
    """
    Class for interacting with trainable language models through OpenAI-compatible API endpoints.

    This class implements functionality for both inference and training-related operations with
    language models served via Text Generation Inference (TGI) or vLLM endpoints that expose
    an OpenAI-compatible API interface. It supports both streaming and non-streaming modes,
    and includes methods for token counting and log probability calculations.

    Attributes:
        base_url (str): Base URL of the API endpoint
        api_token (str): Authentication token for API access
    """

    # TODO: use OpenAI Python client when the certificate issue is resolved.
    # TODO: consider using litellm

    base_url: str | list[str]
    api_token: str = Field(default="", exclude=True)
    collect_logprobs: bool = False
    # vLLM sometimes generate a leading white space https://github.com/vllm-project/vllm/issues/3935
    remove_leading_white_space: bool = False

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.api_token = os.getenv(TAPEAGENTS_LLM_TOKEN, "")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2))
    def _generate(self, prompt: Prompt) -> Generator[LLMEvent, None, None]:
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers |= {"Authorization": f"Bearer {self.api_token}"}
        data = {
            "model": self.model_name,
            "messages": prompt.messages,
            "stream": self.stream,
        }
        if self.collect_logprobs:
            data.update(
                {
                    "logprobs": 1,
                    "include_stop_str_in_output": True,
                    "skip_special_tokens": False,
                }
            )
        base_url = self.base_url if isinstance(self.base_url, str) else random.choice(self.base_url)
        logger.debug(f"POST request to {base_url}/v1/chat/completions")
        start_send_request = time.time()
        r = requests.post(
            url=f"{base_url}/v1/chat/completions",
            json=data | self.parameters,
            headers=headers,
            stream=self.stream,
            verify=False,
        )
        time_send_request = time.time() - start_send_request
        self.stats["time_send_request"].append(time_send_request)
        if not r.ok:
            logger.error(f"Failed to get completion: {r.text}")
            r.raise_for_status()
        if self.stream:
            response_buffer = []
            for byte_payload in r.iter_lines():
                if byte_payload == b"\n":
                    continue
                payload = byte_payload.decode("utf-8")
                if payload.startswith("data:"):
                    if payload == "data: [DONE]":
                        continue
                    json_payload = json.loads(payload.lstrip("data:").rstrip("\n"))
                    response_delta = json_payload["choices"][0]["delta"].get("content", "")
                    if not response_delta:
                        continue
                    response_buffer.append(response_delta)
                    yield LLMEvent(chunk=response_delta)
            output = LLMOutput(content="".join(response_buffer))
        else:
            data = r.json()
            try:
                content = data["choices"][0]["message"]["content"]
                if self.remove_leading_white_space:
                    # vllm sometimes adds a whitespace at the beginning of the completion
                    assert content[0] == " "
                    content = content[1:]
                if not content:
                    logger.warning(f"Empty completion {data}")

                if self.collect_logprobs:
                    completion_log_probs = data["choices"][0]["logprobs"]["content"]

                    if self.tokenizer.eos_token and content.endswith(self.tokenizer.eos_token):
                        # the eos was added in the case where self.collect_logprobs is True
                        # TapeAgents is not expecting the eos token in the completion
                        content = content[: -len(self.tokenizer.eos_token)]
                    output = LLMOutput(content=content, logprobs={"content": completion_log_probs})
                else:
                    output = LLMOutput(content=content)
            except Exception as e:
                logger.exception(f"Failed to parse llm response: {r}")
                raise e
        self.log_output(prompt, output)
        yield LLMEvent(output=output)


    def load_tokenizer(self):
        """
        Loads the tokenizer for the model.

        If the tokenizer is not already loaded, this method will import the
        `transformers` library and load the tokenizer using the model name or
        tokenizer name. If `_MOCK_TOKENIZER` is set, it will use that instead.

        Raises:
            ValueError: If neither `self.tokenizer_name` nor `self.model_name`
                        is provided and `_MOCK_TOKENIZER` is not set.
        """
        if self.tokenizer is None:
            global transformers
            if transformers is None:
                import transformers
            name = _MOCK_TOKENIZER if _MOCK_TOKENIZER else (self.tokenizer_name or self.model_name)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(name)

    def make_training_text(self, prompt: Prompt, output: LLMOutput) -> TrainingText:
        """
        Generates training text from a given prompt and LLM output.

        This method loads the tokenizer and uses it to create training text
        suitable for training a language model.

        Args:
            prompt (Prompt): The input prompt to generate training text from.
            output (LLMOutput): The output from the language model to be used in training.

        Returns:
            TrainingText: The generated training text.
        """
        self.load_tokenizer()
        return trainable_llm_make_training_text(prompt, output, self.tokenizer)

    def get_logprobs_complete(self, prompt: str, output: str) -> dict[str, Any]:
        """
        Get the log probabilities of the tokens in the output given the prompt.

        This method sends a request to the language model API to generate the log probabilities
        for the tokens in the provided output, given the prompt. It uses the tokenizer to encode
        the prompt and output, and extracts the log probabilities from the API response.

        Args:
            prompt (str): The input prompt text.
            output (str): The output text for which log probabilities are to be calculated.

        Returns:
            list[float]: A list of log probabilities for each token in the output.

        Raises:
            RuntimeError: If the API response is not as expected or if there is a mismatch
                          between the tokens in the response and the provided output.
        """
        if not self.tokenizer:
            self.load_tokenizer()

        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers |= {"Authorization": f"Bearer {self.api_token}"}

        if self.tokenizer.bos_token and prompt.startswith(self.tokenizer.bos_token):
            prompt = prompt[len(self.tokenizer.bos_token) :]

        prompt_text = prompt + output
        generation_args = {
            "model": self.model_name,
            "prompt": prompt_text,
            "temperature": 0.0,
            "max_tokens": 0,
            "logprobs": 1,
            "echo": True,
            "include_stop_str_in_output": True,  # self.include_stop_str_in_output,
            "skip_special_tokens": False,
            "n": 1,  # number of completions to generate
            "stream": False,  # return a single completion and not a stream of lines
        }
        base_url = self.base_url if isinstance(self.base_url, str) else random.choice(self.base_url)
        url = f"{base_url}/v1/completions"
        logger.debug(f"POST request to {url}")
        r = requests.post(url, json=generation_args, headers=headers, verify=False)
        r.raise_for_status()  # raise exception if status code is not in the 200s
        try:
            response = r.json()
            tokens = response["choices"][0]["logprobs"]["tokens"]
            log_probs = response["choices"][0]["logprobs"]["token_logprobs"]
            prompt_encoded = self.tokenizer.encode(prompt, add_special_tokens=True)
            prompt_completion_encoded = self.tokenizer.encode(prompt + output, add_special_tokens=True)
            completion_log_probs = log_probs[len(prompt_encoded) : len(prompt_completion_encoded)]
            completion_tokens = tokens[len(prompt_encoded) : len(prompt_completion_encoded)]
            assert (
                "".join(completion_tokens) == output
            ), f"Tokens do not match completion: {''.join(completion_tokens)} != {output}"
        except Exception as e:
            raise RuntimeError(f"Generation API wrong response: {r.text}", e)
        completion_log_probs = [
            {
                "logprob": lp,
                "top_logprobs": [],
                "token": t,
            }
            for lp, t in zip(completion_log_probs, completion_tokens)
        ]
        return {"content": completion_log_probs}

    def get_logprobs_chat_complete(self, prompt: Prompt, output: LLMOutput) -> dict[str, Any]:
        """
        Calculate the log probabilities of the tokens in the completion generated by the language model.

        This function sends a request to the language model API to generate completions and calculate log probabilities.
        The function uses the tokenizer to encode the prompt and completion texts.
        The log probabilities are extracted from the API response and validated against the original completion.

        Args:
            prompt (Prompt): The prompt containing the messages to be sent to the language model.
            output (LLMOutput): The output from the language model containing the generated completion.

        Returns:
            list[float]: A list of log probabilities for each token in the generated completion.

        Raises:
            RuntimeError: If the response from the generation API is incorrect or cannot be parsed.
        """
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers |= {"Authorization": f"Bearer {self.api_token}"}

        time_t0 = time.time()
        prompt_text = self.tokenizer.apply_chat_template(prompt.messages, tokenize=False)
        completion = output.content or ""
        messages = prompt.messages + [{"role": "assistant", "content": completion}]
        prompt_text = self.tokenizer.apply_chat_template(prompt.messages, tokenize=False, add_generation_prompt=True)
        prompt_completion_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        if self.tokenizer.bos_token and prompt_text.startswith(self.tokenizer.bos_token):
            prompt_text = prompt_text[len(self.tokenizer.bos_token) :]
            prompt_completion_text = prompt_completion_text[len(self.tokenizer.bos_token) :]

        prompt_encoded = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_completion_encoded = self.tokenizer.encode(prompt_completion_text, add_special_tokens=False)

        generation_args = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 1,
            "logprobs": 1,
            "echo": True,
            "include_stop_str_in_output": True,  # self.include_stop_str_in_output,
            "skip_special_tokens": False,
            "n": 1,  # number of completions to generate
            "stream": False,  # return a single completion and not a stream of lines
        }
        base_url = self.base_url if isinstance(self.base_url, str) else random.choice(self.base_url)
        r = requests.post(
            url=f"{base_url}/v1/chat/completions",
            json=generation_args,
            headers=headers,
            verify=False,
        )
        r.raise_for_status()

        try:
            response = r.json()
            log_probs = [list(log_prob.values())[0] for log_prob in response["prompt_logprobs"] if log_prob]
            decoded_prompt_completion_tokens = [log_prob["decoded_token"] for log_prob in log_probs][
                : len(prompt_completion_encoded)
            ]
            reconstructed_prompt_completion = "".join(decoded_prompt_completion_tokens)
            completion_log_probs = log_probs[len(prompt_encoded) : len(prompt_completion_encoded)]
            decoded_completion_tokens = [log_prob["decoded_token"] for log_prob in completion_log_probs]
            reconstructed_completion = "".join(decoded_completion_tokens)
            if self.tokenizer.eos_token in reconstructed_completion:
                reconstructed_completion = reconstructed_completion[: -len(self.tokenizer.eos_token)]
            assert (
                reconstructed_completion == completion
            ), f"Tokens do not match completion: {reconstructed_completion} != {completion}"
        except Exception as e:
            raise RuntimeError(f"Generation API wrong response: {r.text}", e)

        logger.debug(f"Log likelihood calculation took {time.time() - time_t0:.2f} seconds")
        logger.debug(f"Tokens per second: {len(log_probs) / (time.time() - time_t0):.2f}")

        completion_log_probs = [
            {
                "logprob": o["logprob"],
                "top_logprobs": [],
                "token": o["decoded_token"],
            }
            for o in completion_log_probs
        ]

        return {"content": completion_log_probs}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2))
    def get_logprobs(self, prompt: str | Prompt, output: str | LLMOutput) -> dict[str, Any]:
        """
        Calculate the log probabilities of the given output based on the provided prompt.

        Args:
            prompt (Union[str, Prompt]): The input prompt, which can be either a string or a Prompt object.
            output (Union[str, LLMOutput]): The output to evaluate, which can be either a string or an LLMOutput object.

        Returns:
            list[float]: A list of log probabilities corresponding to the given output.

        Raises:
            ValueError: If the input types are not valid.
        """
        if isinstance(prompt, str) and isinstance(output, str):
            return self.get_logprobs_complete(prompt=prompt, output=output)
        elif isinstance(prompt, Prompt) and isinstance(output, LLMOutput):
            return self.get_logprobs_chat_complete(prompt=prompt, output=output)
        else:
            raise ValueError("Invalid input types")

    def count_tokens(self, messages: list[dict] | str) -> int:
        """
        Count the number of tokens in the given messages.

        This method loads the tokenizer and then counts the number of tokens
        in the provided messages. The messages can be either a string or a list
        of dictionaries.

        Args:
            messages (Union[list[dict], str]): The messages to count tokens for. It can
                               be a single string or a list of dictionaries.

        Returns:
            int: The number of tokens in the provided messages.
        """
        self.load_tokenizer()
        if isinstance(messages, str):
            return len(self.tokenizer(messages).input_ids)
        else:
            add_generation_prompt = False if messages[-1]["role"] == "assistant" else True
            return len(self.tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt))


class ReplayLLM(LLM):
    """
    Specialized LLM class that replays previously recorded LLM interactions.

    Loads and replays model interactions from a SQLite database, allowing for
    deterministic replay of previous LLM conversations without making new API calls.

    The class is useful for:

    - Testing and debugging LLM interactions
    - Reproducing specific model behaviors
    - Avoiding repeated API calls during development
    - Creating deterministic test scenarios

    Attributes:
        outputs (dict[str, str]): Dictionary mapping prompt strings to their recorded outputs
        llm_calls (list[LLMCall]): List of recorded LLM call objects
        count_tokens_fn (Callable): Function to count tokens in prompts/messages
        make_training_text_fn (Callable): Function to create training text from prompt/output pairs

    Raises:
        FatalError: When a prompt is not found in the recorded outputs
        AssertionError: When the specified SQLite database file doesn't exist
    """

    outputs: dict[str, str] = Field(default_factory=dict)
    llm_calls: list[LLMCall]
    count_tokens_fn: Callable = lambda x: 0
    make_training_text_fn: Callable = lambda x, y: TrainingText(text="", n_predicted=0)

    @classmethod
    def from_llm(cls, llm: LLM, run_dir: str, prompts_file: str = DB_DEFAULT_FILENAME):
        """
        Create a ReplayLLM instance from an existing LLM and a SQLite database file.

        Args:
            cls (Type): The class to instantiate.
            llm (LLM): The original LLM instance.
            run_dir (str): The directory where the SQLite database file is located.
            prompts_file (str, optional): The name of the SQLite database file. Defaults to DB_DEFAULT_FILENAME.

        Returns:
            (ReplayLLM): An instance of ReplayLLM initialized with the LLM calls from the SQLite database.

        Raises:
            AssertionError: If the SQLite database file does not exist at the specified path.
        """
        sqlite_fpath = os.path.join(run_dir, prompts_file)
        assert os.path.exists(sqlite_fpath), f"Sqlite not found: {sqlite_fpath}"
        llm_calls = retrieve_all_llm_calls(sqlite_fpath)
        replay_llm = ReplayLLM(
            llm_calls=llm_calls,
            model_name=llm.tokenizer_name or llm.model_name,
            context_size=llm.context_size,
        )
        replay_llm.tokenizer = llm.tokenizer
        replay_llm.count_tokens_fn = llm.count_tokens
        replay_llm.make_training_text_fn = llm.make_training_text
        return replay_llm

    def model_post_init(self, __context: Any) -> None:
        dups = 0
        for llm_call in self.llm_calls:
            prompt_key = json.dumps(llm_call.prompt.messages, indent=2, ensure_ascii=False, sort_keys=True)
            output = llm_call.output.content or ""
            if prompt_key in self.outputs and output != self.outputs[prompt_key]:
                logger.debug(f"Output duplicate, using last value!\nOLD:{self.outputs[prompt_key]}\nNEW:{output}")
                dups += 1
            self.outputs[prompt_key] = output
        logger.info(f"Loaded {len(self.outputs)} outputs, {dups} duplicates")
        return super().model_post_init(__context)

    def generate(self, prompt: Prompt, **kwargs) -> LLMStream:
        """
        Generates an LLMStream based on the provided prompt.

        This method checks if the prompt has been previously processed and cached. If a cached output is found,
        it is returned. Otherwise, it attempts to find the closest known prompt and logs the differences. If no
        similar prompt is found, a FatalError is raised.

        Args:
            prompt (Prompt): The prompt object containing the messages to be processed.
            **kwargs (dict, optional): Additional keyword arguments.

        Returns:
            LLMStream: A stream of LLM events containing the generated output.

        Raises:
            FatalError: If the prompt is not found in the cache and no similar prompt is found.
        """

        def _implementation():
            prompt_key = json.dumps(prompt.messages, indent=2, ensure_ascii=False, sort_keys=True)
            if prompt_key in self.outputs:
                logger.debug(colored("prompt cache hit", "green"))
                output = self.outputs[prompt_key]
            else:
                logger.warning(
                    colored(f"prompt of size {len(prompt_key)} not found, checking similar ones..", "yellow")
                )
                known_prompts = list(self.outputs.keys())
                closest, score = closest_prompt(prompt_key, known_prompts)
                if score >= 0.7:
                    logger.warning(f"Closest prompt score {score:.3f}")
                    for i, (a, b) in enumerate(zip_longest(prompt.messages, json.loads(closest), fillvalue={})):
                        logger.warning(f"STEP{i}: {diff_strings(a.get('content', str(a)), b.get('content', str(b)))}\n")
                raise FatalError("prompt not found")
            yield LLMEvent(output=LLMOutput(content=output))

        return LLMStream(_implementation(), prompt=prompt)

    def make_training_text(self, prompt: Prompt, output: LLMOutput) -> TrainingText:
        """
        Generates training text based on the provided prompt and output.

        Args:
            prompt (Prompt): The input prompt to generate training text from.
            output (LLMOutput): The output generated by the language model.

        Returns:
            TrainingText: The generated training text.
        """
        return self.make_training_text_fn(prompt, output)

    def count_tokens(self, messages: list[dict] | str) -> int:
        """
        Counts the number of tokens in the given messages.

        Args:
            messages (Union[list[dict], str]): A list of message dictionaries or a single string message.

        Returns:
            int: The total number of tokens in the messages.
        """
        return self.count_tokens_fn(messages)


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


def trainable_llm_make_training_text(prompt: Prompt, output: LLMOutput, tokenizer) -> TrainingText:
    """
    Generates training text for LLM fine-tuning by combining prompt and output using tokenizer's chat template.

    Args:
        prompt (Prompt): The input prompt containing conversation messages.
        output (LLMOutput): The model's output/response.
        tokenizer (PreTrainedTokenizer): The tokenizer used to format the conversation.

    Returns:
        TrainingText: A dataclass containing:

            - text (str): The formatted conversation text
            - n_predicted (int): Length of the output text portion

    Note:
        - Uses tokenizer's chat template to format conversations
        - Removes BOS token if present in the beginning of the text
    """
    prompt_text = tokenizer.apply_chat_template(
        conversation=prompt.messages, tokenize=False, add_generation_prompt=True
    )
    text = tokenizer.apply_chat_template(
        prompt.messages + [{"role": "assistant", "content": output.content}],
        tokenize=False,
    )
    output_text = text[len(prompt_text) :]

    if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
        text = text[len(tokenizer.bos_token) :]

    return TrainingText(text=text, n_predicted=len(output_text))
