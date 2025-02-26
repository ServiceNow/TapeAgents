import json
import logging
import os
import time
from typing import Literal

import requests
from pydantic import Field

from tapeagents.core import Action, Observation
from tapeagents.tools.base import Tool
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
    return [{"title": r["title"], "url": r["link"], "content": r.get("snippet", "")} for r in results[:max_results]]


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
