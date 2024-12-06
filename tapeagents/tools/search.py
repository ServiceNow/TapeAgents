import json
import logging
import os
import threading
import time

import requests

from tapeagents.utils import FatalError, acquire_timeout

logger = logging.getLogger(__name__)

search_lock = threading.Lock()


def web_search(query: str, max_results: int = 5, timeout_sec: int = 5) -> list[dict]:
    with acquire_timeout(search_lock, timeout_sec):
        results = []
        attempts = 2
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
    topic = "videos" if query.startswith("site:youtube.com") else "search"
    payload = json.dumps({"q": query})
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    response = requests.request("POST", f"https://google.serper.dev/{topic}", headers=headers, data=payload)
    results = response.json()["organic"][:max_results]
    return [{"title": r["title"], "url": r["link"], "content": r.get("snippet", "")} for r in results]
