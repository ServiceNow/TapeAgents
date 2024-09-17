import traceback
import json
import logging
import os
import sys
import time
from io import StringIO
from typing import Any, Tuple

from Levenshtein import ratio
from termcolor import colored

from tapeagents.core import Prompt
from tapeagents.llms import LLM
from tapeagents.utils import FatalError, diff_strings

from .calculator import eval_expression_on_dict
from .document_converters import FileConverter
from .gym_browser import GymBrowser
from .python_interpreter import BASE_PYTHON_TOOLS, evaluate_python_code
from .simple_browser import SimpleTextBrowser

logger = logging.getLogger(__name__)


class BasicToolbox:
    """
    A toolbox for basic operations:
    - web search
    - web page reading and navigation
    - reading local documents: pdf, xlsx, pptx, txt and other supported by converter
    - python code execution
    - calculator
    - image to text conversion
    """

    def __init__(
        self,
        vision_lm: LLM | None = None,
        use_web_cache: bool = True,
        only_cached_webpages: bool = False,
        safe_calculator: bool = True,
        browser_viewport_size: int = 32000,
        interactive_browser: bool = False,
        interactive_browser_headless: bool = True,
        exp_path: str | None = None,
    ) -> None:
        def img2text(messages: list[dict]) -> str:
            assert vision_lm
            for event in vision_lm.generate(Prompt(messages=messages)):
                if event.completion and event.completion.content:
                    logger.debug("Image caption", event.completion.content)
                    return event.completion.content
            raise Exception("No answer from vision model")

        mlm_client = img2text if vision_lm else None
        self.converter = FileConverter(mlm_client=mlm_client)
        self.simple_browser = SimpleTextBrowser(viewport_size=browser_viewport_size, converter=self.converter)
        self._interactive_browser = None
        if interactive_browser:
            try:
                self._interactive_browser = GymBrowser(
                    viewport_size=browser_viewport_size, headless=interactive_browser_headless, log_path=exp_path
                )
            except Exception as e:
                logger.warning(f"Failed to initialize interactive browser: {e}")

        self.safe_calculator = safe_calculator
        self.use_web_cache = use_web_cache
        self.only_cached_webpages = only_cached_webpages
        self._cache = {}
        self._log = {}
        self._cache_filename = "web_cache.json"
        if os.path.exists(self._cache_filename):
            with open(self._cache_filename) as f:
                self._cache = json.load(f)
            logger.info(f"Loaded {len(self._cache)} web results from cache")

    @property
    def interactive_browser(self) -> GymBrowser:
        if self._interactive_browser is None:
            raise FatalError("Interactive browser is not available")
        return self._interactive_browser

    def set_web_cache(self, cache: dict) -> None:
        self._cache = cache

    def _add_to_cache(self, k: str, value: Any) -> None:
        self._cache[k] = value
        self._log[k] = value
        with open(self._cache_filename, "w") as f:
            json.dump(self._cache, f)

    def websearch(self, query: str, source: str = "") -> list[dict]:
        if "wiki" in source:
            query = f"site:wikipedia.org {query}"
        if self.use_web_cache and query in self._cache:
            print(colored(f"Cache hit for search {query}", "green"))
            self._log[query] = self._cache[query]
            return self._cache[query]
        if self.only_cached_webpages:
            ratios = [(k, ratio(query, k, score_cutoff=0.5)) for k in self._cache.keys()]
            ratios = sorted(ratios, key=lambda x: x[1], reverse=True)
            closest = ratios[0][0]
            score = ratios[0][1]
            raise FatalError(f'No cache for "{query}"\nClosest is "{closest}"\nWith score {score}')
        result = self.simple_browser.get_search_results(query)
        self._add_to_cache(query, result)
        return result

    def get_page(self, url: str) -> tuple[str, int, int]:
        """
        Load web page and return content of its first viewport (first screen), current page number and total number of pages.
        """
        if url.startswith("/"):
            # in case of a local file
            url = f"file://{url}"
        if self.use_web_cache and url in self._cache:
            logger.info(colored(f"Cache hit {url}", "green"))
            self._log[url] = self._cache[url]
            content, title = self._cache[url]
            self.simple_browser.history.append((url, time.time()))
            self.simple_browser.page_title = title
            self.simple_browser._set_page_content(content)
            self.simple_browser.viewport_current_page = 0
        elif self.only_cached_webpages:
            ratios = [(k, ratio(url, k, score_cutoff=0.7)) for k in self._cache.keys()]
            closest, score = sorted(ratios, key=lambda x: x[1], reverse=True)[0]
            if score >= 0.7:
                logger.warning(diff_strings(url, closest))
                logger.warning(f"Closest url score: {score:.3f}")
            raise FatalError(f"Page {url} not in cache")
        else:
            logger.info(colored(f"Page {url} not in cache", "yellow"))
            self.simple_browser.page_title = ""
            self.simple_browser.visit_page(url)
            self._add_to_cache(url, (self.simple_browser.page_content, self.simple_browser.page_title))
        return (
            self.simple_browser.page_with_title(),
            self.simple_browser.viewport_current_page + 1,
            len(self.simple_browser.viewport_pages),
        )

    def get_next_page(self) -> tuple[str, int, int]:
        if self.simple_browser.viewport_current_page + 1 == len(self.simple_browser.viewport_pages):
            raise ValueError("No more pages to read.")
        self.simple_browser.page_down()
        return (
            self.simple_browser.page_with_title(),
            self.simple_browser.viewport_current_page + 1,
            len(self.simple_browser.viewport_pages),
        )

    def get_whole_document(self, url: str) -> str:
        try:
            self.get_page(url)
        except Exception as e:
            raise Exception(f"Failed to load page {url}.\nError: {e}")
        return self.simple_browser.page_content

    def calculate(self, expression: str, facts: dict) -> str:
        if self.safe_calculator:
            result = eval_expression_on_dict(expression, facts)
        else:
            result = evaluate_python_code(expression, state=facts.copy(), tools=BASE_PYTHON_TOOLS)
        try:
            str_result = json.dumps(result)
        except Exception:
            str_result = str(result)
        return str_result

    def run_python_code(self, code: str, facts: dict, safe: bool=True) -> Tuple[str, str, str]:
        # run code and capture stdout, stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = mystdout = StringIO()
        sys.stderr = mystderr = StringIO()
        
        result = None
        try:
            if safe:
                result = evaluate_python_code(code, state=facts.copy(), tools=BASE_PYTHON_TOOLS)
            else:
                exec(code, facts.copy())
        except Exception:
            traceback.print_exc()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
        try:
            str_result = json.dumps(result)
        except Exception:
            str_result = str(result)
        return str_result, mystdout.getvalue(), mystderr.getvalue()