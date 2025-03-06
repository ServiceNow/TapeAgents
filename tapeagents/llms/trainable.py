import json
import logging
import os
import time
from typing import Any, Generator

import requests
import transformers
from pydantic import Field
from tenacity import retry, stop_after_attempt, wait_exponential

from tapeagents.core import LLMCall, LLMOutput, Prompt, TokenLogprob, TrainingText
from tapeagents.llms.base import LLMEvent
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

    base_url: str
    api_token: str = Field(default="", exclude=True)
    collect_logprobs: bool = False

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.api_token = os.getenv(TAPEAGENTS_LLM_TOKEN, "")

    def get_base_url(self) -> str:
        """
        Returns the base URL for the API endpoint.
        """
        return self.base_url.rstrip("/")

    def make_llm_call_logprobs(
        self, prompt_token_ids: list[int], completion_logprobs: list[dict]
    ) -> list[TokenLogprob]:
        logprobs = []
        for id in prompt_token_ids:
            logprobs.append(
                TokenLogprob(
                    token_id=id,
                    logprob=0.0,
                    generated=0,
                )
            )
        for logprob in completion_logprobs:
            if logprob:
                try:
                    # We assume that the server was launched with --return-tokens-as-token-ids
                    # and that the tokens are provided as: ['token_id:1271', 'token_id:1505', '
                    logprobs.append(
                        TokenLogprob(
                            token_id=int(logprob["token"].split(":")[-1]),
                            logprob=logprob["logprob"],
                            generated=1,
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
        r = requests.post(
            url=f"{self.base_url}/v1/chat/completions",
            json=data | self.parameters,
            headers=headers,
            stream=self.stream,
            verify=False,
        )
        time_send_request = time.time() - start_send_request
        self._stats["time_send_request"].append(time_send_request)
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

                logprobs = None
                if self.collect_logprobs:
                    prompt_token_ids = self.tokenizer.apply_chat_template(
                        prompt.messages, add_special_tokens=True, add_generation_prompt=True
                    )
                    # prompt_decoded = self.tokenizer.decode(prompt_token_ids, skip_special_tokens=False)
                    completion_logprobs = data["choices"][0]["logprobs"]["content"]
                    logprobs = self.make_llm_call_logprobs(prompt_token_ids, completion_logprobs)
                    # <end_of_turn> is the end of message for Gemma2B, eos_token is wrong for this model
                    for eos_str in [self.tokenizer.eos_token, "<end_of_turn>"]:
                        if content.endswith(eos_str):
                            # the eos was added in the case where self.collect_logprobs is True
                            # TapeAgents is not expecting the eos token in the completion
                            content = content[: -len(eos_str)]
            except Exception as e:
                logger.exception(f"Failed to parse llm response: {r}")
                raise e
        output = LLMOutput(content=content)
        llm_call = self.log_output(prompt, output)
        llm_call.logprobs = logprobs
        yield LLMEvent(output=output, llm_call=llm_call)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2))
    def batch_generate(self, prompts: list[Prompt]) -> list[LLMCall]:
        self.load_tokenizer()
        if self.stream:
            raise NotImplementedError()

        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers |= {"Authorization": f"Bearer {self.api_token}"}

        prompt_token_ids = [
            p.token_ids
            if p.token_ids
            else self.tokenizer.apply_chat_template(p.messages, add_special_tokens=True, add_generation_prompt=True)
            for p in prompts
        ]
        data = {
            "model": self.model_name,
            "prompt": prompt_token_ids,
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
        logger.debug(f"POST request to {self.base_url}/v1/completions")
        start_send_request = time.time()
        r = requests.post(
            url=f"{self.base_url}/v1/completions",
            json=data | self.parameters,
            headers=headers,
            stream=self.stream,
            verify=False,
        )
        self._stats["time_send_request"].append(time.time() - start_send_request)
        if not r.ok:
            logger.error(f"Failed to get completion: {r.text}")
            r.raise_for_status()
        data = r.json()
        result = []
        start_postprocess_time = time.time()
        for i in range(len(prompts)):
            try:
                content = data["choices"][i]["text"]
                if not content:
                    logger.warning(f"Empty completion {data}")

                logprobs = None
                if self.collect_logprobs:
                    completion_logprobs = data["choices"][i]["logprobs"]
                    # /v1/completions returns logprobs in a format different to /v1/chat/completions
                    # Before calling self.process_logprobs, we need to convert the logprobs to a
                    # list of dicts format similar to /v1/chat/completions

                    chat_completion_logprobs = [
                        {"token": completion_logprobs["tokens"][j], "logprob": completion_logprobs["token_logprobs"][j]}
                        for j in range(len(completion_logprobs["tokens"]))
                    ]
                    logprobs = self.make_llm_call_logprobs(prompt_token_ids[i], chat_completion_logprobs)
                    # <end_of_turn> is the end of message for Gemma2B, eos_token is wrong for this model
                    for eos_str in [self.tokenizer.eos_token, "<end_of_turn>"]:
                        if content.endswith(eos_str):
                            # the eos was added in the case where self.collect_logprobs is True
                            # TapeAgents is not expecting the eos token in the completion
                            content = content[: -len(eos_str)]
            except Exception as e:
                logger.exception(f"Failed to parse llm response: {r}")
                raise e
            output = LLMOutput(content=content)
            # if logprobs is not None, we will directly take the token counts from vLLM
            # otherwise, we will count the tokens in the output using the tokenizer (which is sometimes inaccurate)
            if logprobs:
                llm_call = self.log_output(prompts[i], output, count_tokens=False)
                llm_call.prompt_length_tokens = len(prompt_token_ids[i])
                llm_call.output_length_tokens = len(chat_completion_logprobs)
                self._stats["prompt_length_tokens"].append(llm_call.prompt_length_tokens)
                self._stats["output_length_tokens"].append(llm_call.output_length_tokens)
                assert (
                    llm_call.output_length_tokens <= self.parameters["max_tokens"]
                ), f"output_length_tokens: {llm_call.output_length_tokens}, max_tokens: {self.parameters['max_tokens']}"
            else:
                llm_call = self.log_output(prompts[i], output, count_tokens=True)
                # do not assert token count since the tokenizer may not be accurate
            llm_call.logprobs = logprobs
            result.append(llm_call)
        self._stats["time_postprocess_llm_response"].append(time.time() - start_postprocess_time)
        return result

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
        self.load_tokenizer()
        if isinstance(messages, str):
            return len(self.tokenizer(messages).input_ids)
        else:
            add_generation_prompt = False if messages[-1]["role"] == "assistant" else True
            return len(self.tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt))
