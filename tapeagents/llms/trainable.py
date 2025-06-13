import asyncio
import json
import logging
import os
import pprint
import time
from typing import Any, Generator

import aiohttp
import litellm
import requests
import transformers
from omegaconf import DictConfig, OmegaConf
from pydantic import Field
from tenacity import retry, stop_after_attempt, wait_exponential

from tapeagents.core import LLMOutput, Prompt, TokenLogprob, TrainingText
from tapeagents.llms.base import LLMEvent, LLMStream
from tapeagents.llms.cached import CachedLLM
from tapeagents.utils import get_step_schemas_from_union_type

# force replacement of the tokenizer during testing
_MOCK_TOKENIZER: str = ""
TAPEAGENTS_LLM_TOKEN = "TAPEAGENTS_LLM_TOKEN"
logger = logging.getLogger(__name__)


def trainable_llm_make_training_text(prompt: Prompt, output: LLMOutput, tokenizer) -> TrainingText:
    """
    Generates training text for LLM fine-tuning by combining prompt and output using tokenizer's chat template.

    Args:
        prompt (Prompt): The input prompt containing conversation messages.
        output (LLMOutput): The model's output/response.
        tokenizer (PreTrainedTokenizer): The tokenizer used to format the conversation.

    Returns:
        TrainingText: A dataclass containing:

            - text (str): The formatted conversation text (prompt + output)
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


class TrainableLLM(CachedLLM):
    """
    Class for interacting with trainable language models through OpenAI-compatible API endpoints.

    This class implements functionality for both inference and training-related operations with
    language models served via Text Generation Inference (TGI) or vLLM endpoints that expose
    an OpenAI-compatible API interface. It supports streaming non-streaming and async modes
    and includes methods for token counting and log probability calculations.

    Attributes:
        base_url (str): Base URL of the API endpoint
        api_token (str): Authentication token for API access
    """

    # TODO: use OpenAI Python client when the certificate issue is resolved.
    # TODO: consider using litellm

    base_url: str = "https://api.openai.com"
    api_token: str = Field(default="", exclude=True)
    collect_logprobs: bool = False
    use_litellm_tokenizer_fallback: bool = False
    max_parallel_requests: int = 32
    max_retries: int = 5
    base_delay: float = 0.5
    _semaphore: asyncio.Semaphore

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.api_token = os.getenv(TAPEAGENTS_LLM_TOKEN, "") or os.getenv("OPENAI_API_KEY", "")
        self._semaphore = asyncio.Semaphore(self.max_parallel_requests)

    def get_base_url(self) -> str:
        """
        Returns the base URL for the API endpoint.
        """
        return self.base_url.rstrip("/")

    def parse_completion_logprobs(self, completion_logprobs: list[dict]) -> list[TokenLogprob]:
        logprobs = []
        for logprob in completion_logprobs:
            if logprob:
                try:
                    # We assume that the server was launched with --return-tokens-as-token-ids
                    # and that the tokens are provided as: ['token_id:1271', 'token_id:1505', '
                    logprobs.append(
                        TokenLogprob(
                            token_id=int(logprob["token"].split(":")[-1]),
                            logprob=logprob["logprob"],
                        )
                    )
                except Exception as e:
                    logger.error(f"Failed to process logprobs: {logprob}")
                    logger.error(e)

        return logprobs

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2))
    def _generate(self, prompt: Prompt) -> Generator[LLMEvent, None, None]:
        self.load_tokenizer()
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers |= {"Authorization": f"Bearer {self.api_token}"}
        data = {
            "model": self.model_name,
            "messages": prompt.messages,
            "stream": self.stream,
            "tools": prompt.tools,
        }
        if self.collect_logprobs:
            data.update(
                {
                    "logprobs": 1,
                    "include_stop_str_in_output": True,
                    "skip_special_tokens": False,
                }
            )
        logger.debug(f"POST request to {self.base_url}/v1/chat/completions")
        start_send_request = time.time()
        for k, v in self.parameters.items():
            data[k] = OmegaConf.to_container(v) if isinstance(v, DictConfig) else v
        r = requests.post(
            url=f"{self.base_url}/v1/chat/completions",
            json=data,
            headers=headers,
            stream=self.stream,
            verify=False,
        )
        time_send_request = time.time() - start_send_request
        self._stats["time_send_request"].append(time_send_request)
        if not r.ok:
            logger.error(f"Failed to get completion: {r.text}")
            r.raise_for_status()
        parsed_logprobs = []
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
                tool_calls = data["choices"][0]["message"].get("tool_calls", [])
                if not content and not tool_calls:
                    logger.warning(f"Empty completion {data}")

                if self.collect_logprobs:
                    completion_logprobs = data["choices"][0]["logprobs"]["content"]
                    parsed_logprobs = self.parse_completion_logprobs(completion_logprobs)
            except Exception as e:
                logger.exception(f"Failed to parse llm response: {r}")
                raise e
            output = LLMOutput(content=content)
            if tool_calls:
                output.tool_calls = [litellm.ChatCompletionMessageToolCall(**tc) for tc in tool_calls]
        llm_call = self.log_output(prompt, output)
        llm_call.logprobs = parsed_logprobs
        yield LLMEvent(output=output, llm_call=llm_call)

    def get_step_schema(self, cls):
        return get_step_schemas_from_union_type(cls)

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
            try:
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(name)
            except Exception as e:
                if not self.use_litellm_tokenizer_fallback:
                    raise e

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

    def get_logprobs_token_ids(self, prompt_token_ids: list[int], completion_token_ids: list[int]) -> dict[str, Any]:
        if not self.tokenizer:
            self.load_tokenizer()

        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers |= {"Authorization": f"Bearer {self.api_token}"}

        generation_args = {
            "model": self.model_name,
            "prompt": prompt_token_ids + completion_token_ids,
            "temperature": 0.0,
            "max_tokens": 0,
            "logprobs": 0,
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
        except Exception as e:
            raise RuntimeError(f"Generation API wrong response: {r.text}", e)
        logprobs = []
        completion_logprobs = response["choices"][0]["prompt_logprobs"][-len(completion_token_ids) :]
        for lp in completion_logprobs:
            if lp:
                for k, v in lp.items():
                    v.update({"generated": 0, "token_id": k})
                    logprobs.append(v)
        return {"content": logprobs}

    def get_batch_logprobs_token_ids(
        self, prompt_token_ids: list[list[int]], completion_token_ids: list[list[int]]
    ) -> list[dict[str, Any]]:
        if not self.tokenizer:
            self.load_tokenizer()
        batch_size = len(prompt_token_ids)

        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers |= {"Authorization": f"Bearer {self.api_token}"}

        generation_args = {
            "model": self.model_name,
            "prompt": [pids + cids for pids, cids in zip(prompt_token_ids, completion_token_ids)],
            "temperature": 0.0,
            "max_tokens": 0,
            "logprobs": 0,
            "echo": True,
            "include_stop_str_in_output": True,  # self.include_stop_str_in_output,
            "skip_special_tokens": False,
            "n": 1,  # number of completions to generate
            "stream": False,  # return a single completion and not a stream of lines
        }
        url = f"{self.base_url}/v1/completions"
        logger.debug(f"POST request to {url}")
        r = requests.post(url, json=generation_args, headers=headers, verify=False)
        r.raise_for_status()

        try:
            response = r.json()
        except Exception as e:
            raise RuntimeError(f"Generation API wrong response: {r.text}", e)

        all_logprobs = []
        for i in range(batch_size):
            logprobs = []
            for lp in response["choices"][i]["prompt_logprobs"][-len(completion_token_ids[i]) :]:
                if lp:
                    for k, v in lp.items():
                        v.update({"generated": 0, "token_id": k})
                        logprobs.append(v)
            all_logprobs.append({"content": logprobs})
        return all_logprobs

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
        base_url = self.get_base_url()
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
        base_url = self.get_base_url()
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
    def get_logprobs(self, prompt: str | Prompt | list[int], output: str | LLMOutput | list[int]) -> dict[str, Any]:
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
        elif isinstance(prompt, list) and isinstance(output, list):
            return self.get_logprobs_token_ids(prompt_token_ids=prompt, completion_token_ids=output)
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
        try:
            self.load_tokenizer()
            if isinstance(messages, str):
                return len(self.tokenizer(messages).input_ids)
            else:
                add_generation_prompt = False if messages[-1]["role"] == "assistant" else True
                return len(self.tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt))
        except Exception as e:
            if self.use_litellm_tokenizer_fallback:
                logger.warning("Failed to count tokens with tokenizer, fallback to litellm counter")
                if isinstance(messages, str):
                    return litellm.token_counter(model=self.model_name, text=messages)  # type: ignore
                else:
                    return litellm.token_counter(model=self.model_name, messages=messages)  # type: ignore
            else:
                raise e

    async def agenerate(self, prompt: Prompt, session: aiohttp.ClientSession, **kwargs) -> LLMStream:
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers |= {"Authorization": f"Bearer {self.api_token}"}
        params = {"model": self.model_name, "messages": prompt.messages, "tools": prompt.tools} | kwargs
        for k, v in self.parameters.items():
            params[k] = OmegaConf.to_container(v) if isinstance(v, DictConfig) else v
        if self.collect_logprobs:
            params.update(
                {
                    "logprobs": 1,
                    "include_stop_str_in_output": True,
                    "skip_special_tokens": False,
                }
            )

        logger.debug(
            f"POST request to {self.base_url}/v1/chat/completions with params: {pprint.pformat(params, width=120)}"
        )
        async with self._semaphore:  # type: ignore
            retry_count = 0
            while True:
                try:
                    async with session.post(
                        url=f"{self.base_url}/v1/chat/completions", json=params, headers=headers, ssl=False
                    ) as response:
                        if not response.ok:
                            error_text = await response.text()
                            logger.error(f"Failed to get completion: {error_text}")
                            response.raise_for_status()
                        data = await response.json()
                        break
                except asyncio.TimeoutError as e:
                    logger.exception("API Timeout, retrying in 1 sec")
                    retry_count += 1
                    if retry_count > self.max_retries:
                        raise e
                    delay = self.base_delay * (2 ** (retry_count - 1))
                    logger.warning(
                        f"API Timeout, retrying in {delay:.2f} seconds (attempt {retry_count}/{self.max_retries})"
                    )
                    await asyncio.sleep(delay)
                except aiohttp.ClientError as e:
                    logger.error(f"Connection error for {self.base_url}/v1/chat/completions: {e}")
                    retry_count += 1
                    if retry_count > self.max_retries:
                        raise e
                    delay = self.base_delay * (2 ** (retry_count - 1))
                    logger.warning(f"Retrying in {delay:.2f} seconds (attempt {retry_count}/{self.max_retries})")
                    await asyncio.sleep(delay)

        try:
            content = data["choices"][0]["message"]["content"]
            tool_calls = data["choices"][0]["message"].get("tool_calls", [])
            if not content and not tool_calls:
                logger.warning(f"Empty completion {data}")

            parsed_logprobs = []
            if self.collect_logprobs:
                completion_logprobs = data["choices"][0]["logprobs"]["content"]
                for logprob in completion_logprobs:
                    if logprob:
                        try:
                            # We assume that the server was launched with --return-tokens-as-token-ids
                            # and that the tokens are provided as: ['token_id:1271', 'token_id:1505', '
                            parsed_logprobs.append(
                                TokenLogprob(
                                    token_id=int(logprob["token"].split(":")[-1]),
                                    logprob=logprob["logprob"],
                                )
                            )
                        except Exception as e:
                            logger.error(f"Failed to process logprobs: {logprob}")
                            logger.error(e)
        except Exception as e:
            logger.exception(f"Failed to parse llm response: {data}")
            raise e

        prompt_tokens = data["usage"]["prompt_tokens"]
        completion_tokens = data["usage"]["completion_tokens"]

        output = LLMOutput(content=content or "")
        if tool_calls:
            output.tool_calls = [litellm.ChatCompletionMessageToolCall(**tc) for tc in tool_calls]
        llm_call = self.log_output(
            prompt,
            output,
            prompt_length_tokens=prompt_tokens,
            output_length_tokens=completion_tokens,
            count_tokens=False,
        )
        assert llm_call is not None, "llm_call is None"
        llm_call.logprobs = parsed_logprobs

        def _gen():
            yield LLMEvent(llm_call=llm_call, output=output)

        return LLMStream(generator=_gen(), prompt=prompt)
