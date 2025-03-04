import time
from typing import Generator

import anthropic
from omegaconf import DictConfig, OmegaConf

from tapeagents.core import Prompt
from tapeagents.llms.base import LLMEvent, LLMOutput
from tapeagents.llms.cached import CachedLLM
from tapeagents.llms.litellm import logger
from tapeagents.utils import get_step_schemas_from_union_type, resize_base64_message


class Claude(CachedLLM):
    max_tokens: int = 4096

    def _generate(
        self,
        prompt: Prompt,
        max_retries: int = 5,
        retry_count: int = 0,
        base_delay: float = 0.5,
        **kwargs,
    ) -> Generator[LLMEvent, None, None]:
        for k, v in self.parameters.items():
            if isinstance(v, DictConfig):
                kwargs[k] = OmegaConf.to_container(v)
            else:
                kwargs[k] = v
        system_message, messages = self.separate_system_message(prompt.messages)
        messages = self.update_image_messages_format(messages)
        while True:
            try:
                response: anthropic.types.Message = anthropic.Anthropic().messages.create(
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    system=system_message,
                    messages=messages,
                    **kwargs,
                )
                break
            except anthropic.RateLimitError as e:
                retry_count += 1
                if retry_count > max_retries:
                    logger.error(f"Rate limit exceeded after {max_retries} retries")
                    raise e
                delay = base_delay * (2 ** (retry_count - 1))
                logger.warning(f"Rate limit hit, retrying in {delay:.2f} seconds (attempt {retry_count}/{max_retries})")
                time.sleep(delay)
            except anthropic.APIConnectionError as e:
                logger.exception(e)
                retry_count += 1
                if retry_count > max_retries:
                    raise e
                delay = base_delay * (2 ** (retry_count - 1))
                logger.warning(f"api error, retrying in {delay:.2f} seconds (attempt {retry_count}/{max_retries})")
                time.sleep(delay)
            except Exception as e:
                logger.exception(e)
                raise e

        for content_block in response.content:
            if content_block.type == "text":
                output = LLMOutput(content=content_block.text)
                yield LLMEvent(output=output)
            elif content_block.type == "tool_use":
                output = LLMOutput(tool_calls=[content_block])
                yield LLMEvent(output=output)
            elif content_block.type == "thinking":
                output = LLMOutput(content=content_block.text)
                yield LLMEvent(output=output)

    def separate_system_message(self, messages: list) -> tuple[str, list]:
        system_message = None
        messages_without_system = []
        for message in messages:
            if message["role"] == "system" and not system_message:
                system_message = message["content"]
            else:
                messages_without_system.append(message)
        return system_message, messages_without_system

    def update_image_messages_format(self, messages: list) -> list:
        # from OAI {"type": "image_url", "image_url": {"url": f"data:{content_type};base64,{base64_image}"}}
        # to Anthropic {"type": "image", "source": {"type": "base64", "media_type": content_type, "data": base64_image}}
        cleaned_messages = []
        for message in messages:
            if isinstance(message["content"], list):
                clean_message = {"role": message["role"], "content": []}
                texts = []
                for submessage in message["content"]:
                    if submessage["type"] == "image_url":
                        submessage = resize_base64_message(submessage)
                        url = submessage["image_url"]["url"]
                        content_type, base64_image = url.split(";base64,")
                        img_message = {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": content_type[5:],
                                "data": base64_image,
                            },
                        }
                        clean_message["content"].append(img_message)
                    else:
                        texts.append(submessage)
                clean_message["content"] += texts  # text should be at the end
                cleaned_messages.append(clean_message)
            else:
                cleaned_messages.append(message)
        return cleaned_messages

    def count_tokens(self, messages: list) -> int:
        system_message, messages = self.separate_system_message(messages)
        messages = self.update_image_messages_format(messages)
        response = anthropic.Anthropic().messages.count_tokens(
            system=system_message,
            model=self.model_name,
            messages=messages,
        )
        return response.input_tokens

    def make_training_text(self, prompt, output):
        return super().make_training_text(prompt, output)

    def get_step_schema(self, cls):
        return get_step_schemas_from_union_type(cls)
