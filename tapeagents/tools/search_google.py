import logging
import os
import re
from time import sleep

from bs4 import BeautifulSoup
from googlesearch import SearchResult, _req

logger = logging.getLogger(__name__)

cache_dir = ".cache/googlesearch"
os.makedirs(cache_dir, exist_ok=True)


# Drop-in replacement for the search function from googlesearch library
def search(
    term,
    num_results=10,
    lang="en",
    proxy=None,
    advanced=False,
    sleep_interval=5,
    timeout=5,
    safe="active",
    ssl_verify=None,
):
    """Search the Google search engine"""

    term

    # Proxy
    proxies = None
    if proxy:
        if proxy.startswith("https"):
            proxies = {"https": proxy}
        else:
            proxies = {"http": proxy}

    # Fetch
    n_result = 10
    start = 0
    results = []
    while start <= num_results:
        # Check cache
        file_path = f"{cache_dir}/{normalize_filename(term)}_{start}.html"
        if os.path.exists(file_path):
            logger.info("Using cached search results")
            with open(file_path, "r") as file:
                html = file.read()
        else:
            # Send request
            resp = _req(term, n_result, lang, start, proxies, timeout, safe, ssl_verify)
            # Save html for caching
            html = resp.text
            with open(file_path, "w") as file:
                file.write(html)
        # Parse html
        results += parse_search_results(html)
        start += n_result
        sleep(sleep_interval)
    return results


def parse_search_results(html):
    results = []
    # Parse
    soup = BeautifulSoup(html, "html.parser")

    # parse video results before standard results
    result_block = soup.find_all(
        lambda tag: tag.name == "a" and tag.get("class") == ["xMqpbd"] and tag.has_attr("href")
    )
    logger.debug(f"Found {len(result_block)} video results")
    for result in result_block:
        aria_label = result.get("aria-label")
        if aria_label:
            parts = aria_label.split(" by ")
            title = parts[0]
            description = parts[1] if len(parts) > 1 else ""
        link = result.get("href")
        if link and title and description:
            search_result = SearchResult(link, title, description)
            results.append(SearchResult(link, title, description))

    # parse standard results
    result_block = soup.find_all("div", attrs={"class": "g"})
    logger.debug(f"Found {len(result_block)} standard results")
    for result in result_block:
        # Find link, title, description
        link = result.find("a", href=True)
        if link:
            link = link["href"]
        else:
            logger.warning(f"No link found: {result}")
        title = result.find("h3")
        if title:
            title = title.text
        else:
            logger.warning(f"No title found: {result}")
        print(title)
        print(link)
        description = result.find("div", {"style": "-webkit-line-clamp:2"})  # standard
        if description is None:
            description = result.find("div", {"class": "ITZIwc"})  # video description
            if description is None:
                description = result.find("span", {"class": "hgKElc"})  # feature snippet
        if description:
            description = description.text
        else:
            logger.warning(f"No description found: {result}")
        if link and title and description:
            search_result = SearchResult(link, title, description)
            results.append(search_result)

    return results


def normalize_filename(text, max_length=255, extension=None):
    # Remove special characters
    text = re.sub(r'[\/:*?"<>|]', "", text)
    # Replace spaces with underscores
    text = re.sub(r"\s+", "_", text.strip())
    # Truncate to max_length
    text = text[:max_length]
    # Add extension if provided
    if extension:
        text = f"{text}.{extension.strip('.')}"
    return text
