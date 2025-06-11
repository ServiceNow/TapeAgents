import json
import logging
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Literal

import requests
from pydantic import BaseModel, ConfigDict, Field
from termcolor import colored

from tapeagents.core import Action, Observation, Prompt
from tapeagents.llms import LLM
from tapeagents.steps import ExplainableAction
from tapeagents.tools.base import Tool
from tapeagents.tools.browser import Fetcher
from tapeagents.tools.simple_browser import SimpleTextBrowser
from tapeagents.tools.tool_cache import cached_tool
from tapeagents.utils import FatalError, response_format

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
        logger.exception(f"Failed to search the web: {e}")
        raise e
    return results


def web_search(
    query: str, time_interval: str = "", max_results: int = 5, retry_pause: int = 2, attempts: int = 3
) -> list[dict]:
    results = []
    while not results and attempts > 0:
        attempts -= 1
        results = serper_search(query, max_results=max_results, time_interval=time_interval)
        if not results:
            logger.warning(f"Empty search results, retrying in {retry_pause} seconds")
            time.sleep(retry_pause)
    if not results:
        raise Exception("Failed to get search results, try to use browser to access the search engine instead")
    return results


def serper_search(query: str, max_results: int = 5, time_interval: str = "") -> list[dict]:
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        raise FatalError("SERPER_API_KEY env var is not set")
    topic = "videos" if "site:youtube.com" in query else "search"
    payload = {"q": query, "location": "United States", "num": max_results}
    if time_interval:
        payload["tbs"] = f"qdr:{time_interval}"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    try:
        response = requests.request(
            "POST", f"https://google.serper.dev/{topic}", headers=headers, data=json.dumps(payload)
        )
        response_dict = response.json()
    except Exception as e:
        logger.exception(f"Serper API error: {e}")
        raise FatalError(f"Failed to get search results: {e}")
    organic = response_dict.get("organic", [])
    videos = response_dict.get("videos", [])
    news = response_dict.get("news", [])
    results = organic + videos + news
    logger.info(f"Search response for query '{query}': code {response.status_code}, {len(results)} results")
    if response.status_code != 200 and len(results) == 0:
        logger.error(f"Search response: {response_dict}")
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
    time_interval: str = Field(description="time interval for the search, could be h, d, w, m or empty", default="")


class SafeSearchAction(SearchAction):
    kind: Literal["safe_search_action"] = "safe_search_action"
    private_context: list[str] = Field(default_factory=list, exclude=True)


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


class SafeSearchResultsObservation(SearchResultsObservation):
    kind: Literal["safe_search_results_observation"] = "safe_search_results_observation"
    safe_search: bool = False
    safe_query: str = ""
    risk_level: int | None = None
    query_rewritten: bool = False
    details: str = ""


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


DEFAULT_SAFE_SEARCH_PROMPT = """You are the “Confidential-Data Guardian”.

Confidentiality rules (memorize, never disclose):
• CONFIDENTIAL → unreleased features, codenames/launch dates, internal roadmaps or strategy, private financials, proprietary tech, customer/partner lists, employee PII, security or legal/internal M&A material, anything labeled “Confidential.”
• NOT CONFIDENTIAL → information already public (press releases, launched products, disclosed financials, open-source code, published patents, public testimonials, job-site head-count, industry-standard best practices).

Tasks
1. From the private context, extract ≤ 5 bullet-point “protected_facts” that match the CONFIDENTIAL list.
2. Decide if answering the query verbatim would reveal any protected_fact → set needs_rewrite true/false.
3. If true, create a safe “result_query” that preserves intent but omits protected facts; otherwise return the original query.

Return **only** this JSON (no extra text):

{{
  "protected_facts": ["fact 1", "fact 2", …],   // may be []
  "needs_rewrite": true|false,
  "result_query": "…rewritten or original query…"
}}
"""


class SafeWebSearch(WebSearch):
    observation: type[Observation] = SafeSearchResultsObservation
    llm: LLM
    prompt: str = DEFAULT_SAFE_SEARCH_PROMPT
    max_private_context_len: int = 300000  # characters

    def execute_action(self, action: SafeSearchAction) -> SafeSearchResultsObservation:
        assert isinstance(action.private_context, list)
        query = action.query
        try:
            # Render the prompt
            private_context_str = "\n".join(action.private_context)

            if len(private_context_str) > self.max_private_context_len:
                logger.warning(
                    f"Private context is too long ({len(private_context_str)} chars), truncating to {self.max_private_context_len} chars"
                )
                private_context_str = private_context_str[: self.max_private_context_len]

            content = f"CONFIDENTIAL DATA:\n{private_context_str}\n\nORIGINAL SEARCH QUERY: {query}"
            prompt = Prompt(messages=[{"role": "system", "content": self.prompt}, {"role": "user", "content": content}])
            privacy_mitigator_output = self.llm.generate(prompt).get_text()

            # Get privacy level
            level = None
            match = re.search(r"PRIVACY RISK:\s*(.+)", privacy_mitigator_output)
            if match:
                level_str = match.group(1).strip()[:1]
                if level_str in ["1", "2", "3", "4", "5"]:
                    level = int(level_str)

            # Get query
            new_query = None
            match = re.search(r"NEW SEARCH QUERY:\s*(.+)", privacy_mitigator_output)
            if match:
                new_query = match.group(1).strip()
                if new_query.lower() == "none":
                    new_query = None

            # If both are present and level >= 3, keep the new query
            if level and new_query and level >= 3:
                query_rewritten = True
                logger.warning(
                    f'\nSAFE_SEARCH: Risk Level {level} Rewriting old query to new query: "{query}" -> "{new_query}"'
                )
            else:
                # If no new query is provided, keep the original query
                query_rewritten = False
                new_query = query
                logger.warning(f'\nSAFE_SEARCH: Risk Level {level} Keeping original query "{query}"')

            results = web_search(new_query, time_interval=action.time_interval)
            result_obs = SafeSearchResultsObservation(
                safe_search=True,
                safe_query=new_query,
                query_rewritten=query_rewritten,
                query=query,
                risk_level=level,
                details=privacy_mitigator_output,
                serp=results,
            )
        except Exception as e:
            logger.exception(f"Failed to search the web: {e}")
            result_obs = SafeSearchResultsObservation(query=action.query, safe_search=True, error=str(e), serp=[])

        return result_obs


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


class SearchExtractAction(
    ExplainableAction,
    SearchTask,
):
    """
    Action that provides parameters for a search function call.
    Could search in the web, wikipedia or youtube.
    Search results will be ordered by relevance from top to bottom.
    """

    kind: Literal["search_extract_action"] = "search_extract_action"  # type: ignore


class SearchAndExtract(Action):
    kind: Literal["search_and_extract"] = "search_and_extract"
    main_task: str
    instructions: str
    tasks: list[SearchTask]
    time_interval: str
    skip_urls: list[str]
    private_context: list[str] = Field(default_factory=list, exclude=True)  # hide so it's not dumped


class PageFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")
    publication_date: str = Field(description="Date in YYYY-MM-DD format, empty string if not available")
    facts: list[str]


class WebPageData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str
    title: str
    content: str | list[str]
    prompt_id: str
    fetch_date_time: str
    date: str | None = None


class ExtractedFactsObservation(Observation):
    kind: Literal["extracted_facts_observation"] = "extracted_facts_observation"
    page_facts: dict[str, list[WebPageData]]

    def llm_view(self, indent=2):
        task_facts = []
        for task, pages in self.page_facts.items():
            prefix = f"Topic: {task}:\n<FACTS>"
            page_strs: list[str] = []
            for page in pages:
                page_facts = (
                    "\n".join([f"- {f}" for f in page.content]) if isinstance(page.content, list) else page.content
                )
                page_strs.append(
                    f"Page [{page.title}][{page.url}]:\nPublication date: {page.date or 'Unknown'}. Extraction date: {page.fetch_date_time}. Facts:\n{page_facts}\n----"
                )
            task_facts.append(prefix + "\n\n".join(page_strs) + "\n</FACTS>")
        facts = "\n\n".join(task_facts)
        return f"<FACTS_COLLECTION>Extracted facts:\n{facts}\n</FACTS_COLLECTION>"

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

    # Additional fields for safe search
    query: str = ""
    safe_query: str = ""
    safe_search: bool = False
    query_rewritten: bool = False
    risk_level: int | None = None
    details: str = ""


class MultiSearchExtract(Tool):
    """
    Performs a search in the web and retrieves the whole content of each page. Then extracts the relevant information.
    """

    llm: LLM
    action: type[Action] = SearchAndExtract
    observation: type[Observation] = ExtractedFactsObservation
    cached: bool = True
    top_k: int = 3
    max_workers: int = 20
    search_timeout: int = 10
    fetch_timeout: int = 30
    extract_timeout: int = 60
    extract_prefix: str = "Your should extract all relevant information from the page.\n\nTASK: "
    safe_search: bool = False
    safe_search_prompt: str = DEFAULT_SAFE_SEARCH_PROMPT

    def model_post_init(self, __context):
        if self.safe_search:
            self._search_tool = SafeWebSearch(llm=self.llm, cached=self.cached, prompt=self.safe_search_prompt)
        else:
            self._search_tool = WebSearch(cached=self.cached)
        return super().model_post_init(__context)

    def execute_action(self, action: SearchAndExtract) -> ExtractedFactsObservation:
        search_results = self.search(action)
        fetch_results = self.fetch(search_results)
        extracted_facts = self.extract(action, fetch_results)
        logger.info(f"Extracted facts from {sum([len(p) for p in extracted_facts.values()])} pages.")
        return ExtractedFactsObservation(page_facts=extracted_facts)

    def search(self, action: SearchAndExtract) -> list[SearchResult]:
        def search_query(i: int, j: int, query: str) -> list[SearchResult]:
            results = []
            if self.safe_search:
                results_obs: SafeSearchResultsObservation = self._search_tool.run(
                    SafeSearchAction(
                        source="web",
                        query=query,
                        private_context=action.private_context,
                        time_interval=action.time_interval,
                    )
                )  # type: ignore
                serp = results_obs.serp[: self.top_k]
                for n, r in enumerate(serp):
                    results.append(
                        SearchResult(
                            task_id=i,
                            query_id=j,
                            query=query,
                            safe_query=results_obs.safe_query,
                            safe_search=True,
                            n=n,
                            title=r["title"],
                            url=r["url"],
                            snippet=r["snippet"],
                        )
                    )
            else:
                serp = self._search_tool.run(
                    SearchAction(source="web", query=query, time_interval=action.time_interval)
                ).serp[: self.top_k]  # type: ignore
                for n, r in enumerate(serp):
                    if r["url"] in action.skip_urls:
                        logger.info(f"Skipping URL {r['url']}")
                        continue
                    results.append(
                        SearchResult(
                            task_id=i,
                            query_id=j,
                            query=query,
                            n=n,
                            title=r["title"],
                            url=r["url"],
                            snippet=r["snippet"],
                        )
                    )
            return results

        search_tasks = [
            (task_id, query_id, query)
            for task_id, search_task in enumerate(action.tasks)
            for query_id, query in enumerate(search_task.queries)
        ]
        executor = ThreadPoolExecutor(max_workers=self.max_workers)
        futures = [executor.submit(search_query, *s) for s in search_tasks]
        results: list[SearchResult] = []
        try:
            for future in as_completed(futures, timeout=self.search_timeout):
                results += future.result()
        except Exception as e:
            logger.error(f"Error occurred while processing web search future: {e}")
        executor.shutdown(wait=False)
        logger.info(f"Got {len(results)} search results for {len(search_tasks)} queries.")
        return results

    def fetch(self, search_results: list[SearchResult]) -> list[SearchResult]:
        fetcher = Fetcher()
        texts = {}
        urls = list(set([result.url for result in search_results]))
        executor = ThreadPoolExecutor(max_workers=self.max_workers)
        futures = [executor.submit(fetcher.fetch_for_llm, url) for url in urls]
        try:
            for future in as_completed(futures, timeout=self.fetch_timeout):
                url, text = future.result()
                texts[url] = text
                logger.info(f"Fetched {len(texts)} out of {len(urls)} pages")
        except Exception as e:
            logger.error(f"Error occurred while processing fetch future: {e}")
        executor.shutdown(wait=False)
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
            postfix = f"\n\nData extraction instructions:\n{action.instructions}\nIf you found the creation date of the document, put it in the first line. If the page is empty or contains only message about blocking the access, return only one word ERROR and nothing else."
            msg = f"{prefix}{fr.text}{postfix}"
            logger.info(f"Page: {fr.url}, prompt length {len(msg)} chars")
            prompt = Prompt(messages=[{"role": "user", "content": msg}])
            extract_tasks.append((task, fr.url, fr.title, prompt))

        def extract_page_data(task: str, url: str, title: str, prompt: Prompt) -> tuple[str, WebPageData]:
            page_data_content = self.llm.generate(prompt).get_text()
            if page_data_content.startswith("ERROR"):
                logger.warning(f"Page {url} empty or blocked")
                return "", None

            msg = f"Convert the following text into a structured JSON object:\n\n{page_data_content}\n\nIf there is no date in the first line of the text, leave date field empty. Do not split the tables into multiple facts, put every table into single fact string."
            page_data_json = self.llm.generate(
                Prompt(messages=[{"role": "user", "content": msg}], response_format=response_format(PageFacts))
            ).get_text()
            try:
                page_facts = PageFacts.model_validate_json(page_data_json)
            except Exception as e:
                logger.exception(f"Failed to produce structured list of facts: {e}")
                return None, None  # type: ignore
            logger.info(
                colored(
                    f"Completed extraction for page: {url}\nDate {page_facts.publication_date}, facts: {page_facts.facts}",
                    "green",
                )
            )
            return task, WebPageData(
                url=url,
                title=title,
                content=page_facts.facts,
                prompt_id=prompt.id,
                date=page_facts.publication_date,
                fetch_date_time=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            )

        data_per_task = defaultdict(list)
        executor = ThreadPoolExecutor(max_workers=self.max_workers)
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
        executor.shutdown(wait=False)
        return data_per_task


EXTRACT_INSTRUCTIONS = """
Create markdown document with all the facts and claims from the provided document.
First try to find the creation date of the document in its content. If it is not available, skip it and not mention.
Then collect all facts, claims events and definitions mentioned in the document and produce a complete list.
Things that are relevant to the current task should come first.
If there are any tables, use markdown to represent them.
Fact is any numerical or categorical information about a person, place, event, product, object or a company.
Claim is any statement made by a person that is cited in the web page. Claims should be attributed to the person who made them and represented as verbatim quotes.
When providing quotes, ensure that the quote includes enough context to be understood.
Do not append the word "Fact" or "Claim" to the beginning of sentences.
Important! Respond with the plain text, do not include any JSON or code.
Do not output anything besides what I asked in this message.
"""


class SearchExtract(MultiSearchExtract):
    """
    Performs a search in the web for multiple queries and retrieves the whole content of each page.
    Then extracts the relevant information.
    """

    action: type[Action] = SearchExtractAction
    observation: type[Observation] = ExtractedFactsObservation

    def execute_action(self, action: SearchExtractAction) -> ExtractedFactsObservation:
        multi_action = SearchAndExtract(
            main_task="perform web search and extract facts",
            instructions=EXTRACT_INSTRUCTIONS,
            tasks=[action],
            time_interval="",
            skip_urls=[],
            private_context=[],
        )
        search_results = self.search(multi_action)
        fetch_results = self.fetch(search_results)
        extracted_facts = self.extract(multi_action, fetch_results)
        logger.info(f"Extracted facts from {sum([len(p) for p in extracted_facts.values()])} pages.")
        return ExtractedFactsObservation(page_facts=extracted_facts)
