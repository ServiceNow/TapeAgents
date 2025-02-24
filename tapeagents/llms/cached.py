import hashlib
import json
import logging
import os
import threading
from abc import abstractmethod
from typing import Generator

from termcolor import colored

from tapeagents.config import common_cache_dir
from tapeagents.core import LLMOutput, Prompt
from tapeagents.llms.base import LLM, LLMEvent, LLMStream, closest_prompt
from tapeagents.observe import retrieve_all_llm_calls
from tapeagents.utils import diff_strings

logger = logging.getLogger(__name__)
# Use this variable to force all LLMs to use cache from the sqlite DB
# This is meant to be used for testing purposes only
_REPLAY_SQLITE: str = ""


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
        param_hash = self._key(json.dumps({k: v for k, v in self.parameters.items() if k != "token"}))
        name = self.model_name.replace("/", "__")
        prefix = f"llm_cache_{name}_{param_hash}."
        cache_dir = common_cache_dir()
        self._cache_file = os.path.join(cache_dir, f"{prefix}{os.getpid()}.{threading.get_native_id()}.jsonl")
        if os.path.exists(cache_dir):
            for fname in os.listdir(cache_dir):
                if not fname.startswith(prefix):
                    continue
                with open(os.path.join(cache_dir, fname)) as f:
                    for line in f:
                        key, event_dict = json.loads(line)
                        if key not in self._cache:
                            self._cache[key] = []
                        self._cache[key].append(event_dict)
            logger.info(f"Loaded {len(self._cache)} llm calls from cache {cache_dir}")
        else:
            logger.info(f"Cache dir {cache_dir} does not exist")

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
                logger.debug(colored(f"LLM cache hit, {len(self._cache[key])} events", "green"))
                for event_dict in self._cache[key]:
                    event = LLMEvent.model_validate(event_dict)
                    if event.output is not None:
                        self.log_output(prompt, event.output, cached=True)
                    yield event
            else:
                if _REPLAY_SQLITE:
                    closest, score = closest_prompt(key, list(self._cache.keys()))
                    logger.error(
                        f"LLM cache miss, closest in cache has score {score:.3f}\nNEW:\n{key}\nCLOSEST OLD:\n{closest}\nDIFF:\n{diff_strings(key, closest)}"
                    )
                    raise ValueError(f"LLM cache miss not allowed. Prompt key: {key}")
                toks = self.count_tokens(prompt.messages)
                self.token_count += toks
                logger.debug(f"{toks} prompt tokens, total: {self.token_count}")
                for event in self._generate(prompt, **kwargs):
                    self._add_to_cache(key, event.model_dump())
                    # note: the underlying LLM will log the output
                    yield event

        return LLMStream(_implementation(), prompt)

    def quick_response(self, text_prompt: str) -> str:
        prompt = Prompt(messages=[{"role": "user", "content": text_prompt}])
        outputs = []
        for e in self.generate(prompt):
            if e.output:
                outputs.append(e.output.content)
        return "".join(outputs)

    @abstractmethod
    def _generate(self, prompt: Prompt, **kwargs) -> Generator[LLMEvent, None, None]:
        pass
