from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Generator

import litellm
import openai
import requests
from Levenshtein import ratio
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from termcolor import colored

from .config import DB_DEFAULT_FILENAME
from .core import LLMOutput, Prompt, TrainingText
from .observe import LLMCall, observe_llm_call, retrieve_all_llm_calls
from .utils import FatalError, diff_strings

requests.packages.urllib3.disable_warnings()  # type: ignore
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

TAPEAGENTS_LLM_TOKEN = "TAPEAGENTS_LLM_TOKEN"


class LLMEvent(BaseModel):
    chunk: str | None = None
    output: LLMOutput | None = None


class LLMStream:
    """Wrapper around LLM generator to allow for fast-tracking to the complete message."""

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
        for event in self:
            if event.output:
                return event.output
        raise ValueError("LLM did not produce an output")

    def get_text(self) -> str:
        o = self.get_output()
        if not o.role == "assistant" or o.content is None:
            raise ValueError("LLM did not produce an assistant message")
        return o.content


class LLM(BaseModel, ABC):
    model_name: str
    parameters: dict = {}
    context_size: int = 32000
    tokenizer_name: str = ""
    tokenizer: Any = None
    
    token_count: int = 0
    _log: list = []

    @abstractmethod
    def generate(self, prompt: Prompt, **kwargs) -> LLMStream:
        pass

    @abstractmethod
    def count_tokens(self, messages: list[dict] | str) -> int:
        pass

    @abstractmethod
    def make_training_text(self, prompt: Prompt, output: LLMOutput) -> TrainingText:
        pass

    def log_output(self, prompt: Prompt, message: LLMOutput, cached: bool = False):
        llm_call = LLMCall(
            timestamp=datetime.datetime.now().isoformat(),
            prompt=prompt,
            output=message,
            prompt_length_tokens=self.count_tokens(prompt.messages),
            output_length_tokens=self.count_tokens(message.content) if message.content else 0,
            cached=cached,
        )
        self._log.append(llm_call.model_dump())
        observe_llm_call(llm_call)


# Use this variable to force all LLMs to use cache from the sqlite DB
# This is meant to be used for testing purposes only
_REPLAY_SQLITE: str = ""
# force replacement of the tokenizer during testing
_MOCK_TOKENIZER: str = ""


class CachedLLM(LLM):
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
        cnt = 0
        for log_data in self._log:
            key = self.get_prompt_key(Prompt.model_validate(log_data["prompt"]))
            self._add_to_cache(key, LLMEvent(output=LLMOutput.model_validate(log_data["output"])).model_dump())
            cnt += 1
        logger.info(f"Reindexed {cnt} log entries")

    def _add_to_cache(self, key: str, event_dict: dict):
        if not self.use_cache:
            return
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
    def count_tokens(self, messages: list[dict] | str) -> int:
        if isinstance(messages, str):
            return litellm.token_counter(model=self.model_name, text=messages)
        else:
            return litellm.token_counter(model=self.model_name, messages=messages)

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
            except openai.APITimeoutError as e:
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
                        raise NotImplementedError(f"TODO: streaming with function calls not implemented yet")
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
        raise NotImplementedError()


class TrainableLLM(CachedLLM):
    """Talk to TGI or VLLM endpoints serving HF based model (Llama, Mistral, etc.) using OpenAI API.

    # TODO: use OpenAI Python client when the certificate issue is resolved.
    # TODO: consider using litellm
    """

    base_url: str
    api_token: str = Field(default="", exclude=True)

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
        r = requests.post(
            url=f"{self.base_url}/v1/chat/completions",
            json=data | self.parameters,
            headers=headers,
            stream=self.stream,
            verify=False,
        )
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
                if not content:
                    logger.warning(f"Empty completion {data}")
                output = LLMOutput(content=content)
            except Exception as e:
                logger.exception(f"Failed to parse llm response: {r}")
                raise e
        self.log_output(prompt, output)
        yield LLMEvent(output=output)

    def load_tokenizer(self):
        if self.tokenizer is None:
            import transformers

            name = _MOCK_TOKENIZER if _MOCK_TOKENIZER else (self.tokenizer_name or self.model_name)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(name)

    def make_training_text(self, prompt: Prompt, output: LLMOutput) -> TrainingText:
        self.load_tokenizer()
        return trainable_llm_make_training_text(prompt, output, self.tokenizer)

    def get_log_probs_complete(self, prompt: str, output: str) -> list[float]:
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
        url = f"{self.base_url}/v1/completions"
        logger.debug(f"POST request to {url}")
        r = requests.post(url, json=generation_args, headers=headers, verify=False)
        r.raise_for_status()  # raise exception if status code is not in the 200s
        try:
            response = r.json()
            log_probs = response["choices"][0]["logprobs"]["token_logprobs"]
            prompt_encoded = self.tokenizer.encode(prompt, add_special_tokens=True)
            prompt_completion_encoded = self.tokenizer.encode(prompt + output, add_special_tokens=True)
            log_probs = log_probs[len(prompt_encoded) : len(prompt_completion_encoded)]
            tokens = response["choices"][0]["logprobs"]["tokens"]
            tokens = tokens[len(prompt_encoded) : len(prompt_completion_encoded)]
            assert "".join(tokens) == output, f"Tokens do not match completion: {''.join(tokens)} != {output}"
        except Exception as e:
            raise RuntimeError(f"Generation API wrong response: {r.text}", e)
        return log_probs

    def get_log_probs_chat_complete(self, prompt: Prompt, output: LLMOutput) -> list[float]:
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers |= {"Authorization": f"Bearer {self.api_token}"}

        time_t0 = time.time()
        prompt_text = self.tokenizer.apply_chat_template(prompt.messages, tokenize=False)
        completion = output.content or ""
        messages = prompt.messages + [output.model_dump()]
        prompt_text = self.tokenizer.apply_chat_template(prompt.messages, tokenize=False, add_generation_prompt=True)
        prompt_completion_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        if self.tokenizer.bos_token and prompt_text.startswith(self.tokenizer.bos_token):
            prompt_text = prompt_text[len(self.tokenizer.bos_token) :]
            prompt_completion_text = prompt_completion_text[len(self.tokenizer.bos_token) :]

        prompt_encoded = self.tokenizer.encode(prompt_text, add_special_tokens=True)
        prompt_completion_encoded = self.tokenizer.encode(prompt_completion_text, add_special_tokens=True)

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
        r = requests.post(
            url=f"{self.base_url}/v1/chat/completions",
            json=generation_args,
            headers=headers,
            verify=False,
        )
        r.raise_for_status()

        try:
            response = r.json()
            log_probs = []
            decoded_tokens = []
            for log_prob in response["prompt_logprobs"]:
                if log_prob:
                    token_key = next(iter(log_prob))
                    token_info = log_prob[token_key]
                    log_probs.append(token_info["logprob"])
                    decoded_tokens.append(token_info["decoded_token"])
                else:
                    log_probs.append(0.0)
                    decoded_tokens.append("")

            log_probs = log_probs[len(prompt_encoded) : len(prompt_completion_encoded)]
            decoded_tokens = decoded_tokens[len(prompt_encoded) : len(prompt_completion_encoded)]
            reconstructed_completion = "".join(decoded_tokens)
            if self.tokenizer.eos_token in reconstructed_completion:
                reconstructed_completion = reconstructed_completion[: -len(self.tokenizer.eos_token)]
            assert (
                reconstructed_completion == completion
            ), f"Tokens do not match completion: {reconstructed_completion} != {completion}"
        except Exception as e:
            raise RuntimeError(f"Generation API wrong response: {r.text}", e)

        logger.debug(f"Log likelihood calculation took {time.time() - time_t0:.2f} seconds")
        logger.debug(f"Tokens per second: {len(log_probs) / (time.time() - time_t0):.2f}")

        return log_probs

    def get_log_probs(self, prompt: str | Prompt, output: str | LLMOutput) -> list[float]:
        if isinstance(prompt, str) and isinstance(output, str):
            return self.get_log_probs_complete(prompt=prompt, output=output)
        elif isinstance(prompt, Prompt) and isinstance(output, LLMOutput):
            return self.get_log_probs_chat_complete(prompt=prompt, output=output)
        else:
            raise ValueError("Invalid input types")

    def count_tokens(self, messages: list[dict] | str) -> int:
        self.load_tokenizer()
        if isinstance(messages, str):
            return len(self.tokenizer(messages).input_ids)
        else:
            return len(self.tokenizer.apply_chat_template(messages))


class ReplayLLM(LLM):
    outputs: dict[str, str] = Field(default_factory=dict)
    llm_calls: list[LLMCall]
    count_tokens_fn: Callable = lambda x: 0
    make_training_text_fn: Callable = lambda x, y: TrainingText(text="", n_predicted=0)

    @classmethod
    def from_llm(cls, llm: LLM, run_dir: str, prompts_file: str = DB_DEFAULT_FILENAME):
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
                    logger.warning(f"Closest prompt score {score:.3f}:\n{diff_strings(closest, prompt_key)}")
                raise FatalError("prompt not found")
            yield LLMEvent(output=LLMOutput(content=output))

        return LLMStream(_implementation(), prompt=prompt)

    def make_training_text(self, prompt: Prompt, output: LLMOutput) -> TrainingText:
        return self.make_training_text_fn(prompt, output)

    def count_tokens(self, messages: list[dict] | str) -> int:
        return self.count_tokens_fn(messages)


def closest_prompt(prompt_key: str, known_prompts: list[str]) -> tuple[str, float]:
    ratios = [(k, ratio(prompt_key, k, score_cutoff=0.5)) for k in known_prompts]
    if not len(ratios):
        return "", 0.0
    ratios = sorted(ratios, key=lambda x: x[1], reverse=True)
    closest, score = sorted(ratios, key=lambda x: x[1], reverse=True)[0]
    return closest, score


class MockLLM(LLM):
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
    prompt_text = tokenizer.apply_chat_template(
        conversation=prompt.messages, tokenize=False, add_generation_prompt=True
    )
    text = tokenizer.apply_chat_template(
        prompt.messages + [{"role": "assistant", "content": output.content}], tokenize=False
    )
    output_text = text[len(prompt_text) :]

    return TrainingText(text=text, n_predicted=len(output_text))
