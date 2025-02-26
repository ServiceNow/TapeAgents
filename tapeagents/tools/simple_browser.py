# Modified from the original source: https://github.com/microsoft/autogen/blob/gaia_multiagent_v01_march_1st/autogen/browser_utils.py
# MIT License

# Copyright (c) Microsoft Corporation.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.


import json
import logging
import mimetypes
import os
import pathlib
import re
import threading
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from urllib.parse import unquote, urljoin, urlparse

import pathvalidate
import requests
from Levenshtein import ratio
from pydantic import Field
from termcolor import colored

from tapeagents.config import common_cache_dir, force_cache
from tapeagents.core import Action, Observation
from tapeagents.tools.base import StatefulTool
from tapeagents.utils import FatalError, diff_strings

from .converters import (
    FileConversionException,
    FileConverter,
    UnsupportedFormatException,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_CACHE_PREFIX = "web_cache"


class SimpleTextBrowser:
    """A minimal text-based web browser designed for AI agent use."""

    user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"

    def __init__(
        self,
        start_page: Optional[str] = None,
        viewport_size: Optional[int] = 32000,
        downloads_folder: str = "/tmp/agent_browser_downloads",
        use_web_cache: bool = True,
        request_kwargs: Optional[Union[Dict[str, Any], None]] = None,
        converter_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.start_page: str = start_page if start_page else "about:blank"
        self.viewport_size = viewport_size  # Applies only to the standard uri types
        self.downloads_folder = downloads_folder
        os.makedirs(self.downloads_folder, exist_ok=True)
        self.history: List[Tuple[str, float]] = list()
        self.page_title: Optional[str] = None
        self.viewport_current_page = 0
        self.viewport_pages: List[Tuple[int, int]] = list()
        self.set_address(self.start_page)
        self.request_kwargs = request_kwargs or {"headers": {"User-Agent": self.user_agent}}
        self.request_kwargs["headers"] = self.request_kwargs.get("headers", {})

        self._mdconvert = FileConverter()

        self._page_content: str = ""
        self._page_error: int = 0
        self.converter_kwargs = converter_kwargs or {}

        self._find_on_page_query: Union[str, None] = None
        self._find_on_page_last_result: Union[int, None] = None  # Location of the last result

        self.use_web_cache = use_web_cache
        self._cache = {}
        self._log = []
        self._cache_buffer = []
        self.load_cache()

    def load_cache(self):
        cache_dir = common_cache_dir()
        if os.path.exists(cache_dir):
            for fname in os.listdir(cache_dir):
                if not fname.startswith(_CACHE_PREFIX):
                    continue
                with open(os.path.join(cache_dir, fname)) as f:
                    for line in f:
                        data = json.loads(line)
                        self._cache[data["k"]] = data["v"]
        logger.info(f"Loaded {len(self._cache)} web results from cache {cache_dir}")

    @property
    def address(self) -> str:
        """Return the address of the current page."""
        return self.history[-1][0]

    def set_address(self, uri_or_path: str) -> None:
        """Update the address, visit the page, and set the content of the viewport."""
        self.history.append((uri_or_path, time.time()))

        # Handle special URIs
        if uri_or_path == "about:blank":
            self._set_page_content("")
        else:
            if (
                not uri_or_path.startswith("http:")
                and not uri_or_path.startswith("https:")
                and not uri_or_path.startswith("file:")
            ):
                if len(self.history) > 1:
                    prior_address = self.history[-2][0]
                    uri_or_path = urljoin(prior_address, uri_or_path)
                    # Update the address with the fully-qualified path
                    self.history[-1] = (uri_or_path, self.history[-1][1])
            self._fetch_page(uri_or_path)

        self.viewport_current_page = 0
        self.find_on_page_query = None
        self.find_on_page_viewport = None

    @property
    def viewport(self) -> str:
        """Return the content of the current viewport."""
        bounds = self.viewport_pages[self.viewport_current_page]
        return self.page_content[bounds[0] : bounds[1]]

    @property
    def page_content(self) -> str:
        """Return the full contents of the current page."""
        return self._page_content

    def _set_page_content(self, content: str) -> None:
        """Sets the text content of the current page."""
        self._page_content = content
        self._split_pages()
        if self.viewport_current_page >= len(self.viewport_pages):
            self.viewport_current_page = len(self.viewport_pages) - 1

    def page_down(self) -> None:
        self.viewport_current_page = min(self.viewport_current_page + 1, len(self.viewport_pages) - 1)

    def page_up(self) -> None:
        self.viewport_current_page = max(self.viewport_current_page - 1, 0)

    def find_on_page(self, query: str) -> Union[str, None]:
        """Searches for the query from the current viewport forward, looping back to the start if necessary."""

        # Did we get here via a previous find_on_page search with the same query?
        # If so, map to find_next
        if query == self._find_on_page_query and self.viewport_current_page == self._find_on_page_last_result:
            return self.find_next()

        # Ok it's a new search start from the current viewport
        self._find_on_page_query = query
        viewport_match = self._find_next_viewport(query, self.viewport_current_page)
        if viewport_match is None:
            self._find_on_page_last_result = None
            return None
        else:
            self.viewport_current_page = viewport_match
            self._find_on_page_last_result = viewport_match
            return self.viewport

    def find_next(self) -> str | None:
        """Scroll to the next viewport that matches the query"""

        if self._find_on_page_query is None:
            return None

        starting_viewport = self._find_on_page_last_result
        if starting_viewport is None:
            starting_viewport = 0
        else:
            starting_viewport += 1
            if starting_viewport >= len(self.viewport_pages):
                starting_viewport = 0

        viewport_match = self._find_next_viewport(self._find_on_page_query, starting_viewport)
        if viewport_match is None:
            self._find_on_page_last_result = None
            return None
        else:
            self.viewport_current_page = viewport_match
            self._find_on_page_last_result = viewport_match
            return self.viewport

    def _find_next_viewport(self, query: str, starting_viewport: int) -> Union[int, None]:
        """Search for matches between the starting viewport looping when reaching the end."""

        if query is None:
            return None

        # Normalize the query, and convert to a regular expression
        nquery = re.sub(r"\*", "__STAR__", query)
        nquery = " " + (" ".join(re.split(r"\W+", nquery))).strip() + " "
        nquery = nquery.replace(" __STAR__ ", "__STAR__ ")  # Merge isolated stars with prior word
        nquery = nquery.replace("__STAR__", ".*").lower()

        if nquery.strip() == "":
            return None

        idxs = list()
        idxs.extend(range(starting_viewport, len(self.viewport_pages)))
        idxs.extend(range(0, starting_viewport))

        for i in idxs:
            bounds = self.viewport_pages[i]
            content = self.page_content[bounds[0] : bounds[1]]

            # TODO: Remove markdown links and images
            ncontent = " " + (" ".join(re.split(r"\W+", content))).strip().lower() + " "
            if re.search(nquery, ncontent):
                return i

        return None

    def _split_pages(self) -> None:
        # Do not split search results
        if self.address.startswith("search:"):
            self.viewport_pages = [(0, len(self._page_content))]
            return

        # Handle empty pages
        if len(self._page_content) == 0:
            self.viewport_pages = [(0, 0)]
            return

        # Break the viewport into pages
        self.viewport_pages = []
        start_idx = 0
        while start_idx < len(self._page_content):
            end_idx = min(start_idx + self.viewport_size, len(self._page_content))  # type: ignore[operator]
            # Adjust to end on a space
            while end_idx < len(self._page_content) and self._page_content[end_idx - 1] not in [" ", "\t", "\r", "\n"]:
                end_idx += 1
            self.viewport_pages.append((start_idx, end_idx))
            start_idx = end_idx

    def _fetch_page(self, url: str) -> None:
        download_path = ""
        response = None
        try:
            if url.startswith("file://"):
                download_path = os.path.normcase(os.path.normpath(unquote(url[7:])))
                res = self._mdconvert.convert_local(download_path, **self.converter_kwargs)
                self.page_title = res.title
                self._set_page_content(res.text_content)
            else:
                # Prepare the request parameters
                request_kwargs = self.request_kwargs.copy() if self.request_kwargs is not None else {}
                request_kwargs["stream"] = True

                response = requests.get(url, **request_kwargs)
                response.raise_for_status()

                content_type = response.headers.get("content-type", "")
                # Text or HTML
                if "text/" in content_type.lower():
                    res = self._mdconvert.convert_response(response, **self.converter_kwargs)
                    self.page_title = res.title
                    self._set_page_content(res.text_content)
                # A download
                else:
                    # Try producing a safe filename
                    fname = None
                    download_path = None
                    try:
                        fname = pathvalidate.sanitize_filename(os.path.basename(urlparse(url).path)).strip()
                        download_path = os.path.abspath(os.path.join(self.downloads_folder, fname))

                        suffix = 0
                        while os.path.exists(download_path) and suffix < 1000:
                            suffix += 1
                            base, ext = os.path.splitext(fname)
                            new_fname = f"{base}__{suffix}{ext}"
                            download_path = os.path.abspath(os.path.join(self.downloads_folder, new_fname))
                    except NameError:
                        pass

                    # No suitable name, so make one
                    if fname is None:
                        extension = mimetypes.guess_extension(content_type)
                        if extension is None:
                            extension = ".download"
                        fname = str(uuid.uuid4()) + extension
                        download_path = os.path.abspath(os.path.join(self.downloads_folder, fname))

                    # Open a file for writing
                    if not download_path:
                        raise ValueError("Could not determine a suitable download path.")

                    with open(download_path, "wb") as fh:
                        for chunk in response.iter_content(chunk_size=512):
                            fh.write(chunk)

                    # Render it
                    local_uri = pathlib.Path(download_path).as_uri()
                    self.set_address(local_uri)

        except UnsupportedFormatException as e:
            logger.error(colored(f"UnsupportedFormatException: {e}", "red"))
            self.page_title = "Unsupported Format"
            self._set_page_content(f"Unsupported Format File: {e}")
            self._page_error = 1
        except FileConversionException as e:
            logger.error(colored(f"FileConversionException: {e}", "red"))
            self.page_title = "Failed to read file"
            self._set_page_content(f"Error: {e}")
            self._page_error = 2
        except FileNotFoundError:
            self.page_title = "Error 404"
            self._set_page_content(f"## Error 404\n\nFile not found: {download_path}")
            self._page_error = 404
        except requests.exceptions.RequestException as e:
            if response is None:
                self._set_page_content(f"## Error {e}")
                self._page_error = 3
            else:
                self.page_title = f"Error {response.status_code}"
                self._page_error = response.status_code
                # If the error was rendered in HTML we might as well render it
                content_type = response.headers.get("content-type", "")
                if content_type is not None and "text/html" in content_type.lower():
                    res = self._mdconvert.convert(response, **self.converter_kwargs)
                    self.page_title = f"Error {response.status_code}"
                    self._set_page_content(f"## Error {response.status_code}\n\n{res.text_content[:500]}")
                else:
                    text = ""
                    for chunk in response.iter_content(chunk_size=512, decode_unicode=True):
                        text += chunk
                    self.page_title = f"Error {response.status_code}"
                    self._set_page_content(f"## Error {response.status_code}\n\n{text[:500]}")
        except Exception as e:
            self._page_error = -1
            self.page_title = "Failed to fetch page"
            self._set_page_content(str(e))

    def page_with_title(self) -> str:
        if self._page_error:
            header = (
                f"Failed to load page, Error {self._page_error}\nTitle: {self.page_title}\n=======================\n"
            )
        else:
            header = f"Title: {self.page_title}\n=======================\n" if self.page_title else ""
        return header + self.viewport.strip()

    def _add_to_cache(self, key: str, value: Any) -> None:
        self._cache[key] = value
        self._log.append({"k": key, "v": value})
        self._cache_buffer.append({"k": key, "v": value})
        self.flush_cache()

    def flush_cache(self):
        fname = os.path.join(common_cache_dir(), f"{_CACHE_PREFIX}.{os.getpid()}.{threading.get_native_id()}.jsonl")
        with open(fname, "a") as f:
            for item in self._cache_buffer:
                f.write(json.dumps(item) + "\n")
        self._cache_buffer = []

    def flush_log(self, exp_dir: str):
        os.makedirs(os.path.join(exp_dir, "browser_log"), exist_ok=True)
        browser_log_path = os.path.join(exp_dir, f"browser_log/{os.getpid()}.{threading.get_native_id()}.jsonl")
        if len(self._log):
            with open(browser_log_path, "a") as wf:
                for line in self._log:
                    wf.write(json.dumps(line) + "\n")
            self._log = []

    def get_page(self, url: str) -> tuple[str, int, int]:
        """
        Loads a web page and returns a tuple containing:
        - Content of its first viewport (first screen)
        - Current page number
        - Total number of pages
        """
        self._page_error = 0
        if url.startswith("/"):
            # in case of a local file
            url = f"file://{url}"
        if self.use_web_cache and url in self._cache:
            logger.info(colored(f"Cache hit {url}", "green"))
            self._log.append({"k": url, "v": self._cache[url]})
            content, title = self._cache[url]
            self.history.append((url, time.time()))
            self.page_title = title
            self._set_page_content(content)
            self.viewport_current_page = 0
        elif force_cache():
            ratios = [(k, ratio(url, k, score_cutoff=0.7)) for k in self._cache.keys()]
            closest, score = sorted(ratios, key=lambda x: x[1], reverse=True)[0]
            if score >= 0.7:
                logger.warning(diff_strings(url, closest))
                logger.warning(f"Closest url score: {score:.3f}")
            raise FatalError(f"Page {url} not in cache")
        else:
            logger.info(colored(f"Page {url} not in cache", "yellow"))
            self.page_title = ""
            self.set_address(url)
            self._add_to_cache(url, (self.page_content, self.page_title))
        error = self._page_error
        self._page_error = 0
        return (self.page_with_title(), len(self.viewport_pages), error)

    def get_next_page(self) -> tuple[str, int, int]:
        """
        Load next page of the document and return the current content of the viewport, current page number and total number of pages.
        """
        if self.viewport_current_page + 1 == len(self.viewport_pages):
            raise ValueError("No more pages to read.")
        self.page_down()
        return (
            self.page_with_title(),
            self.viewport_current_page + 1,
            len(self.viewport_pages),
        )

    def get_whole_document(self, url: str) -> str:
        try:
            self.get_page(url)
        except Exception as e:
            raise Exception(f"Failed to load page {url}.\nError: {e}")
        return self.page_content


class PageDownAction(Action):
    """
    Action that scrolls down to display the next page of the current document.
    """

    kind: Literal["page_down_action"] = "page_down_action"


class PageUpAction(Action):
    """
    Action that scrolls up to display the previous page of the current document.
    """

    kind: Literal["page_up_action"] = "page_up_action"


class ReadDocumentAction(Action):
    """
    Action that loads a document, file, image, or web page from the provided URL or file path.
    Returns the first page of its content. Use page_down_action to view subsequent pages.
    """

    kind: Literal["read_document_action"] = "read_document_action"
    url: str = Field(description="URL or file path of the document")
    fact_description: str = Field(description="Description of the fact to search for in the document")
    fact_name: str = Field(description="Name of the fact to search for in the document")


class PageObservation(Observation):
    kind: Literal["page_observation"] = "page_observation"
    text: str
    current_page: int
    total_pages: int
    error: int | str | None = None

    def short_view(self):
        view = self.llm_dict()
        view["text"] = view["text"][:100] + "..."
        return json.dumps(view, indent=2, ensure_ascii=False)


class SimpleBrowser(StatefulTool):
    """
    Simple text-based web browser for reading documents and web pages.
    Can load web page or document from a URL or file path and scroll through its content.
    """

    actions: tuple[type[Action], ...] = (ReadDocumentAction, PageDownAction)
    observations: tuple[type[Observation], ...] = (PageObservation,)
    exp_path: str
    kwargs: dict[str, Any]
    _browser: SimpleTextBrowser = None  # type: ignore

    def model_post_init(self, __context: Any):
        self._browser = SimpleTextBrowser(**self.kwargs)

    def execute_action(self, action: ReadDocumentAction | PageDownAction) -> PageObservation:
        if isinstance(action, ReadDocumentAction):
            text, total_pages, error = self._browser.get_page(action.url)
            obs = PageObservation(text=text, current_page=1, total_pages=total_pages, error=error or None)
        else:
            text, current_page, total_pages = self._browser.get_next_page()
            obs = PageObservation(
                text=text,
                current_page=current_page,
                total_pages=total_pages,
                error=self._browser._page_error if self._browser._page_error else None,
            )
        return obs

    def close(self) -> None:
        self._browser.flush_log(self.exp_path)
