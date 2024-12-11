import json
import logging
import os
import threading
import time
from typing import Literal

import requests
from pydantic import Field

from tapeagents.core import Action, Observation
from tapeagents.tools.base import Tool
from tapeagents.utils import FatalError, acquire_timeout

logger = logging.getLogger(__name__)

search_lock = threading.Lock()


def web_search(query: str, max_results: int = 5, timeout_sec: int = 5) -> list[dict]:
    with acquire_timeout(search_lock, timeout_sec):
        results = []
        attempts = 3
        while not results and attempts > 0:
            attempts -= 1
            try:
                results = serper_search(query, max_results=max_results)
            except Exception as e:
                logger.warning(f"Failed to fetch search results: {e}")
            time.sleep(1)
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
        raise FatalError(f"Failed to get search results: {e}")
    results = response_dict.get("organic", []) + response_dict.get("videos", []) + response_dict.get("news", [])
    for item in response_dict.get("knowledgeGraph", []):
        results.append({"title": item["title"], "linqk": item.get("website", ""), "snippet": item["description"]})
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


class Search(Tool):
    """
    Tool that performs a search in the web, wikipedia or youtube
    """

    action: type[Action] = SearchAction
    observation: type[Observation] = SearchResultsObservation
    cached: bool = True

    def run(self, action: SearchAction) -> SearchResultsObservation:
        if action.source == "wiki":
            query = f"site:wikipedia.org {action.query}"
        elif action.source == "youtube":
            query = f"site:youtube.com {action.query}"
        else:
            query = action.query
        return SearchResultsObservation(query=action.query, serp=web_search(query))
