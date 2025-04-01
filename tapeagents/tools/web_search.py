import json
import logging
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

import requests
from pydantic import BaseModel, ConfigDict, Field
from termcolor import colored

from tapeagents.core import Action, Observation, Prompt
from tapeagents.llms import LLM
from tapeagents.tools.base import Tool
from tapeagents.tools.browser import Fetcher
from tapeagents.tools.simple_browser import SimpleTextBrowser
from tapeagents.tools.tool_cache import cached_tool
from tapeagents.utils import FatalError

logger = logging.getLogger(__name__)


def web_search_tool(query: str, max_results: int = 5, retry_pause: int = 5, attempts: int = 3) -> list[dict]:
    """
    Search the web for a given query, return a list of search result dictionaries.
    """
    return _web_search(query, max_results=max_results, retry_pause=retry_pause, attempts=attempts)


@cached_tool
def _web_search(query: str, max_results: int = 5, retry_pause: int = 5, attempts: int = 3) -> list[dict]:
    try:
        results = web_search(query, max_results=max_results, retry_pause=retry_pause, attempts=attempts)
    except Exception as e:
        logger.warning(f"Failed to fetch search results: {e}")
    return results


def web_search(query: str, max_results: int = 5, retry_pause: int = 2, attempts: int = 3) -> list[dict]:
    results = []
    while not results and attempts > 0:
        attempts -= 1
        results = serper_search(query, max_results=max_results)
        if not results:
            logger.warning(f"Empty search results, retrying in {retry_pause} seconds")
            time.sleep(retry_pause)
    if not results:
        raise Exception("Failed to get search results, try to use browser to access the search engine instead")
    return results


def serper_search(query: str, max_results: int = 5) -> list[dict]:
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        raise FatalError("SERPER_API_KEY env var is not set")
    topic = "videos" if "site:youtube.com" in query else "search"
    payload = json.dumps({"q": query, "location": "United States", "num": max_results})
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    try:
        response = requests.request("POST", f"https://google.serper.dev/{topic}", headers=headers, data=payload)
        response_dict = response.json()
    except Exception as e:
        logger.exception(f"Serper API error: {e}")
        raise FatalError(f"Failed to get search results: {e}")
    organic = response_dict.get("organic", [])
    videos = response_dict.get("videos", [])
    news = response_dict.get("news", [])
    results = organic + videos + news
    logger.info(f"Search response for query '{query}': code {response.status_code}, {len(results)} results")
    return [{"title": r["title"], "url": r["link"], "snippet": r.get("snippet", "")} for r in results[:max_results]]


class SearchAction(Action):
    """
    Action that provides parameters for a search function call.
    Could search in the web, wikipedia or youtube.
    Search results will be ordered by relevance from top to bottom.
    """

    kind: Literal["search_action"] = "search_action"
    source: str = Field(description="source to search in, could be web, wiki or youtube")
    query: str = Field(description="search query")


class SearchResultsObservation(Observation):
    kind: Literal["search_results_observation"] = "search_results_observation"
    query: str
    serp: list[dict[str, str]]
    error: str | None = None

    def short_view(self):
        view = self.llm_dict()
        for i, result in enumerate(view["serp"]):
            if "content" in result:
                view["serp"][i]["content"] = result["content"][:100] + "..."
        short = json.dumps(view, indent=2, ensure_ascii=False)
        logger.info(f"SearchResultsObservation long view was {len(self.llm_view())} chars, short view is {len(short)}")
        return short


class WebSearch(Tool):
    """
    Performs a search in the web, wikipedia or youtube
    """

    action: type[Action] = SearchAction
    observation: type[Observation] = SearchResultsObservation
    cached: bool = True

    def execute_action(self, action: SearchAction) -> SearchResultsObservation:
        if action.source == "wiki":
            query = f"site:wikipedia.org {action.query}"
        elif action.source == "youtube":
            query = f"site:youtube.com {action.query}"
        else:
            query = action.query
        error = None
        results = []
        try:
            results = web_search(query)
        except Exception as e:
            logger.exception(f"Failed to search the web: {e}")
            error = str(e)
        return SearchResultsObservation(query=action.query, serp=results, error=error)


class SuperSearch(WebSearch):
    """
    Performs a search in the web or wikipedia.
    Retrieves the whole content of each page.
    """

    enriched_results: int = 3
    page_viewport_size: int = 32000

    def model_post_init(self, __context):
        self._browser = SimpleTextBrowser(viewport_size=self.page_viewport_size)
        return super().model_post_init(__context)

    def execute_action(self, action: SearchAction) -> SearchResultsObservation:
        obs = super().execute_action(action)
        for i, result in enumerate(obs.serp[: self.enriched_results]):
            text, total_pages, error = self._browser.get_page(result["url"])
            logger.info(f"Fetched {len(text)} chars from the search result {i}")
            if not error:
                obs.serp[i]["content"] = text
            else:
                logger.warning(f"Failed to fetch page {result['url']}: {error}")
        return obs


class SearchTask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    section: str
    facts_to_discover: str
    queries: list[str]


class SearchAndExtract(Action):
    kind: Literal["search_and_extract"] = "search_and_extract"
    main_task: str
    instructions: str
    tasks: list[SearchTask]


class WebPageData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str
    title: str
    content: str
    prompt_id: str


class ExtractedFactsObservation(Observation):
    kind: Literal["extracted_facts_observation"] = "extracted_facts_observation"
    page_facts: dict[str, list[WebPageData]]

    def llm_view(self, indent=2):
        task_facts = []
        for task, pages in self.page_facts.items():
            prefix = f"Topic: {task}:\n<FACTS>"
            page_strs: list[str] = []
            for page in pages:
                page_strs.append(f"Page [{page.title}][{page.url}]:\n{page.content}\n--------")
            task_facts.append(prefix + "\n\n".join(page_strs) + "\n</FACTS>")
        return "\n\n".join(task_facts)

    def short_view(self):
        tasks = "\n".join(self.page_facts.keys())
        return f"Extracted facts for tasks:\n{tasks}"


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: int
    query_id: int
    n: int
    title: str
    url: str
    snippet: str
    text: str = ""


class SearchExtract(Tool):
    """
    Performs a search in the web and retrieves the whole content of each page. Then extracts the relevant information.
    """

    llm: LLM
    action: type[Action] = SearchAndExtract
    observation: type[Observation] = ExtractedFactsObservation
    cached: bool = True
    page_viewport_size: int = 64000
    top_k: int = 3
    max_workers: int = 20
    search_timeout: int = 30
    fetch_timeout: int = 60
    extract_timeout: int = 60
    extract_prefix: str = "Your should extract all relevant information from the page.\n\nTASK: "

    def model_post_init(self, __context):
        self._search_tool = WebSearch()
        return super().model_post_init(__context)

    def execute_action(self, action: SearchAndExtract) -> ExtractedFactsObservation:
        search_results = self.search(action)
        fetch_results = self.fetch(search_results)
        extracted_facts = self.extract(action, fetch_results)
        logger.info(f"Extracted facts from {sum([len(p) for p in extracted_facts.values()])} pages.")
        return ExtractedFactsObservation(page_facts=extracted_facts)

    def search(self, action: SearchAndExtract) -> list[SearchResult]:
        def search_query(i: int, j: int, query: str) -> list[SearchResult]:
            serp = self._search_tool.run(SearchAction(source="web", query=query)).serp[: self.top_k]
            results = []
            for n, r in enumerate(serp):
                results.append(
                    SearchResult(task_id=i, query_id=j, n=n, title=r["title"], url=r["url"], snippet=r["snippet"])
                )
            return results

        search_tasks = [
            (task_id, query_id, query)
            for task_id, search_task in enumerate(action.tasks)
            for query_id, query in enumerate(search_task.queries)
        ]
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(search_query, *s) for s in search_tasks]
            results: list[SearchResult] = []
            try:
                for future in as_completed(futures, timeout=self.search_timeout):
                    results += future.result()
            except Exception as e:
                logger.error(f"Error occurred while processing web search future: {e}")
        logger.info(f"Got {len(results)} search results for {len(search_tasks)} queries.")
        return results

    def fetch(self, search_results: list[SearchResult]) -> list[SearchResult]:
        fetcher = Fetcher()
        texts = {}
        urls = list(set([result.url for result in search_results]))
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            start_t = time.perf_counter()
            futures = [executor.submit(fetcher.fetch_for_llm, url) for url in urls]
            try:
                for future in as_completed(futures, timeout=self.fetch_timeout):
                    url, text = future.result()
                    texts[url] = text
                    logger.info(f"Fetched {len(texts)} out of {len(urls)} pages")
            except Exception as e:
                logger.error(f"Error occurred while processing fetch future: {e}")
                dt = time.perf_counter() - start_t
                if dt >= self.fetch_timeout:
                    logger.warning("Stopping all remained futures")
                    executor.shutdown(wait=False, cancel_futures=True)
        logger.info(f"Fetched {len(texts)} pages")
        for i in range(len(search_results)):
            search_results[i].text = texts.get(search_results[i].url, "")
        return search_results

    def extract(self, action: SearchAndExtract, fetch_results: list[SearchResult]) -> dict[str, list[WebPageData]]:
        extract_tasks = []
        for fr in fetch_results:
            task = action.tasks[fr.task_id].section
            facts_to_discover = action.tasks[fr.task_id].facts_to_discover
            query = action.tasks[fr.task_id].queries[fr.query_id]
            prefix = f"{self.extract_prefix}{task} (part of higher-level task {action.main_task})\nFacts to discover: {facts_to_discover}\nSearch query that led to the page: {query}\n\nPage content:\n\n"
            postfix = f"\n\nData extraction instructions:\n{action.instructions}\nIf the page is empty or contains only message about blocking the access, return only one word ERROR and nothing else."
            msg = f"{prefix}{fr.text}{postfix}"
            logger.info(f"Page: {fr.url}, prompt length {len(msg)} chars")
            prompt = Prompt(messages=[{"role": "user", "content": msg}])
            extract_tasks.append((task, fr.url, fr.title, prompt))

        def extract_page_data(task: str, url: str, title: str, prompt: Prompt) -> tuple[str, WebPageData]:
            page_data_content = self.llm.generate(prompt).get_text()
            if page_data_content.startswith("ERROR"):
                logger.warning(f"Page {url} empty or blocked")
                return "", None
            logger.info(colored(f"Completed extraction for page: {url}\nFacts output: {page_data_content}", "green"))
            return task, WebPageData(url=url, title=title, content=page_data_content, prompt_id=prompt.id)

        data_per_task = defaultdict(list)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(extract_page_data, *et) for et in extract_tasks]
            extracted = 0
            try:
                for future in as_completed(futures, timeout=self.extract_timeout):
                    task, page_data = future.result()
                    extracted += 1
                    logger.info(f"Extracted {extracted} out of {len(extract_tasks)} pages")
                    if not task:
                        continue
                    data_per_task[task].append(page_data)
            except Exception as e:
                logger.error(f"Error occurred while processing extract future: {e}")
        return data_per_task
