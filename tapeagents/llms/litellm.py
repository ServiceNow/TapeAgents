import logging
import time
from typing import Generator

import litellm
import requests
from omegaconf import DictConfig, OmegaConf

from tapeagents.core import Prompt, TrainingText
from tapeagents.llms.base import LLMEvent, LLMOutput
from tapeagents.llms.cached import CachedLLM
from tapeagents.utils import get_step_schemas_from_union_type

requests.packages.urllib3.disable_warnings()  # type: ignore
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class LiteLLM(CachedLLM):
    """A LiteLLM implementation of the LLM interface.

    This class provides integration with the LiteLLM library for making LLM API calls.
    It supports both streaming and non-streaming responses, token counting, and handles API timeouts with retries.
    Streaming responses are handled by yielding chunks of text as they arrive.
    Non-streaming responses return complete messages.

    Note:
        Function calling during streaming is not yet implemented and will raise NotImplementedError.
    """

    simple_schemas: bool = True

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

    def get_step_schema(self, cls):
        return get_step_schemas_from_union_type(cls, self.simple_schemas)

    def _generate(
        self,
        prompt: Prompt,
        max_retries: int = 5,
        retry_count: int = 0,
        base_delay: float = 0.5,
        **kwargs,
    ) -> Generator[LLMEvent, None, None]:
        while True:
            for k, v in self.parameters.items():
                if isinstance(v, DictConfig):
                    kwargs[k] = OmegaConf.to_container(v)
                else:
                    kwargs[k] = v
            try:
                response = litellm.completion(
                    model=self.model_name,
                    messages=prompt.messages,
                    tools=prompt.tools,
                    stream=self.stream,
                    **kwargs,
                )
                break
            except litellm.RateLimitError as e:
                retry_count += 1
                if retry_count > max_retries:
                    logger.error(f"Rate limit exceeded after {max_retries} retries")
                    raise e
                delay = base_delay * (2 ** (retry_count - 1))
                logger.warning(f"Rate limit hit, retrying in {delay:.2f} seconds (attempt {retry_count}/{max_retries})")
                time.sleep(delay)
            except litellm.Timeout as e:
                logger.exception("API Timeout, retrying in 1 sec")
                retry_count += 1
                if retry_count > max_retries:
                    raise e
                delay = base_delay * (2 ** (retry_count - 1))
                logger.warning(f"API Timeout, retrying in {delay:.2f} seconds (attempt {retry_count}/{max_retries})")
                time.sleep(delay)
            except litellm.BadRequestError as e:
                logger.exception(e)
                retry_count += 1
                if retry_count > max_retries:
                    raise e
                delay = base_delay * (2 ** (retry_count - 1))
                logger.warning(f"Bad request, retrying in {delay:.2f} seconds (attempt {retry_count}/{max_retries})")
                time.sleep(delay)
            except tuple(litellm.LITELLM_EXCEPTION_TYPES) as e:
                logger.exception(e)
                raise e
        if self.stream:
            buffer = []
            for part in response:
                assert isinstance(part, (litellm.ModelResponse, litellm.ModelResponseStream))
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
