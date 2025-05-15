import logging
import os

import aiohttp
import litellm
import transformers

from tapeagents.core import LLMOutput, Prompt, TokenLogprob, TrainingText
from tapeagents.llms.base import LLM, LLMEvent, LLMStream
from tapeagents.llms.trainable import trainable_llm_make_training_text

logger = logging.getLogger(__name__)


class AsyncLLM(LLM):
    base_url: str = "https://api.openai.com"
    api_token: str = os.environ.get("OPENAI_API_KEY", "")
    collect_logprobs: bool = False

    async def generate(self, prompt: Prompt, session: aiohttp.ClientSession, **kwargs) -> LLMStream:
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers |= {"Authorization": f"Bearer {self.api_token}"}
        params = {"model": self.model_name, "messages": prompt.messages} | self.parameters | kwargs
        if self.collect_logprobs:
            params.update(
                {
                    "logprobs": 1,
                    "include_stop_str_in_output": True,
                    "skip_special_tokens": False,
                }
            )

        logger.debug(f"POST request to {self.base_url}/v1/chat/completions with params: {params}")

        async with session.post(
            url=f"{self.base_url}/v1/chat/completions", json=params, headers=headers, ssl=False
        ) as response:
            if not response.ok:
                error_text = await response.text()
                logger.error(f"Failed to get completion: {error_text}")
                response.raise_for_status()
            data = await response.json()

        try:
            content = data["choices"][0]["message"]["content"]
            if not content:
                logger.warning(f"Empty completion {data}")

            logprobs = []
            if self.collect_logprobs:
                self.load_tokenizer()
                prompt_token_ids = self.tokenizer.apply_chat_template(
                    prompt.messages, add_special_tokens=True, add_generation_prompt=True
                )
                completion_logprobs = data["choices"][0]["logprobs"]["content"]
                logprobs = self.make_llm_call_logprobs(prompt_token_ids, completion_logprobs)
                # <end_of_turn> is the end of message for Gemma2B, eos_token is wrong for this model
                for eos_str in [self.tokenizer.eos_token, "<end_of_turn>"]:
                    if content.endswith(eos_str):
                        # the eos was added in the case where self.collect_logprobs is True
                        # TapeAgents is not expecting the eos token in the completion
                        content = content[: -len(eos_str)]
        except Exception as e:
            logger.exception(f"Failed to parse llm response: {data}")
            raise e

        prompt_tokens = data["usage"]["prompt_tokens"]
        completion_tokens = data["usage"]["completion_tokens"]

        output = LLMOutput(content=content or "")
        logger.info(f"LLM content: {content}")
        llm_call = self.log_output(
            prompt,
            output,
            prompt_length_tokens=prompt_tokens,
            output_length_tokens=completion_tokens,
            count_tokens=False,
        )
        assert llm_call is not None, "llm_call is None"
        llm_call.logprobs = logprobs

        def _gen():
            yield LLMEvent(llm_call=llm_call, output=output)

        return LLMStream(generator=_gen(), prompt=prompt)

    def count_tokens(self, messages: list[dict] | str) -> int:
        try:
            self.load_tokenizer()
            if isinstance(messages, str):
                return len(self.tokenizer(messages).input_ids)
            else:
                add_generation_prompt = False if messages[-1]["role"] == "assistant" else True
                return len(self.tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt))
        except Exception:
            logger.warning("Failed to count tokens with tokenizer, fallback to litellm counter")
            if isinstance(messages, str):
                return litellm.token_counter(model=self.model_name, text=messages)  # type: ignore
            else:
                return litellm.token_counter(model=self.model_name, messages=messages)  # type: ignore

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

    def load_tokenizer(self):
        name = self.tokenizer_name or self.model_name
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(name)

    def make_training_text(self, prompt: Prompt, output: LLMOutput) -> TrainingText:
        return trainable_llm_make_training_text(prompt, output, self.tokenizer)
