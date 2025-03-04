import json
import logging
import os
from itertools import zip_longest
from typing import Any, Callable

from pydantic import Field
from termcolor import colored

from tapeagents.config import DB_DEFAULT_FILENAME
from tapeagents.core import LLMOutput, Prompt, TrainingText
from tapeagents.llms.base import LLM, LLMCall, LLMEvent, LLMStream, closest_prompt
from tapeagents.observe import retrieve_all_llm_calls
from tapeagents.utils import FatalError, diff_strings

logger = logging.getLogger(__name__)


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
    _get_step_schema: Callable = lambda: None

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
        replay_llm._get_step_schema = llm.get_step_schema
        return replay_llm

    def get_step_schema(self, *args, **kwargs):
        return self._get_step_schema(*args, **kwargs)

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
            elif len(prompt_key) < 20000:
                logger.warning(
                    colored(f"prompt of size {len(prompt_key)} not found, checking similar ones..", "yellow")
                )
                known_prompts = list(self.outputs.keys())
                closest, score = closest_prompt(prompt_key, known_prompts)
                if score >= 0.7:
                    logger.warning(f"Closest prompt score {score:.3f}")
                    for i, (a, b) in enumerate(zip_longest(prompt.messages, json.loads(closest), fillvalue={})):
                        aa = a.get("content", str(a))
                        bb = b.get("content", str(b))
                        if aa == bb:
                            continue
                        if len(aa) < 300 and len(bb) < 300:
                            logger.warning(f"STEP{i} A:\n{aa}\nSTEP{i} B:\n{bb}")
                        else:
                            logger.warning(f"STEP{i}: {diff_strings(aa, bb)}\n")
                raise FatalError("prompt not found")
            else:
                messages_previews = []
                for m in prompt.messages:
                    try:
                        if isinstance(m["content"], list):
                            msg = "[text,img]"
                        else:
                            m_dict = json.loads(m["content"])
                            msg = list(m_dict.keys())
                        messages_previews.append({"role": m["role"], "content": msg})
                    except Exception:
                        messages_previews.append({"role": m["role"], "content": "text"})
                logger.warning(
                    f"prompt with {len(prompt.messages)} messages {messages_previews}, {len(prompt_key)} chars not found, skipping.."
                )
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
