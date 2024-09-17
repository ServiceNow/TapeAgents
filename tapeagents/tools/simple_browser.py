# based on Autogen project, MIT License: https://github.com/microsoft/autogen/blob/gaia_multiagent_v01_march_1st/autogen/browser_utils.py

import mimetypes
import os
import pathlib
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import unquote, urljoin, urlparse

import pathvalidate
import requests
from googlesearch import search
from tavily import TavilyClient
from termcolor import colored

from .document_converters import (
    FileConversionException,
    FileConverter,
    UnsupportedFormatException,
)


def get_tavily_key():
    if os.path.exists("TAVILY_API_KEY"):
        with open("TAVILY_API_KEY") as f:
            return f.read().strip()
    key = os.environ.get("TAVILY_API_KEY")
    if key is None:
        raise ValueError("TAVILY_API_KEY environment variable must be set.")


class SimpleTextBrowser:
    """An extremely simple text-based web browser suitable for Agentic use."""

    user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"

    def __init__(
        self,
        start_page: Optional[str] = None,
        viewport_size: Optional[int] = 1024 * 8,
        downloads_folder: str = "/tmp/agent_browser_downloads",
        use_tavily: bool = False,
        converter: Optional[FileConverter] = None,
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
        self._mdconvert = converter or FileConverter()
        self._page_content: str = ""
        self._page_error: int = 0
        self.tavily = TavilyClient(api_key=get_tavily_key()) if use_tavily else None
        self.converter_kwargs = converter_kwargs or {}

        self._find_on_page_query: Union[str, None] = None
        self._find_on_page_last_result: Union[int, None] = None  # Location of the last result

    @property
    def address(self) -> str:
        """Return the address of the current page."""
        return self.history[-1][0]

    def set_address(self, uri_or_path: str) -> None:
        # TODO: Handle anchors
        self.history.append((uri_or_path, time.time()))

        # Handle special URIs
        if uri_or_path == "about:blank":
            self._set_page_content("")
        elif uri_or_path.startswith("search:"):
            self._search(uri_or_path[len("search:") :].strip())
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

    def visit_page(self, path_or_uri: str) -> str:
        """Update the address, visit the page, and return the content of the viewport."""
        self._page_error = 0
        self.set_address(path_or_uri)
        return self.viewport

    def search(self, query: str) -> str:
        """Search for the query and return the content of the viewport."""
        self.set_address(f"search: {query}")
        return self.viewport

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

    def _search(self, query: str) -> None:
        results = self.get_search_results(query)

        def _prev_visit(url):
            for i in range(len(self.history) - 1, -1, -1):
                if self.history[i][0] == url:
                    # Todo make this more human-friendly
                    return f"You previously visited this page {round(time.time() - self.history[i][1])} seconds ago.\n"
            return ""

        web_snippets: List[str] = list()
        idx = 0
        for page in results:
            title = page["title"]
            _url = page["url"]
            description = page["content"]
            idx += 1
            web_snippets.append(f"{idx}. [{title}]({_url})\n{_prev_visit(_url)}{description}")

        self.page_title = f"{query} - Search"

        content = f"Search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(
            web_snippets
        )

        self._set_page_content(content)

    def get_search_results(self, query: str, max_results: int = 5) -> list[dict]:
        if self.tavily is not None:
            serp = self.tavily.search(query=query, search_depth="basic", max_results=max_results) or {"results": []}
            results = [{"title": r["title"], "url": r["url"], "content": r["content"][:200]} for r in serp["results"]]
        else:
            results = [
                {"title": r.title, "url": r.url, "content": r.description}
                for r in search(query, advanced=True, num_results=max_results)
            ]
        return results[:max_results]

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
            print(colored(f"UnsupportedFormatException: {e}", "red"))
            self.page_title = "Download complete."
            self._set_page_content(f"# Download complete\n\nSaved file to '{download_path}'")
        except FileConversionException as e:
            print(colored(f"FileConversionException: {e}", "red"))
            self.page_title = "Download complete."
            self._set_page_content(f"# Download complete\n\nSaved file to '{download_path}'")
        except FileNotFoundError:
            self.page_title = "Error 404"
            self._set_page_content(f"## Error 404\n\nFile not found: {download_path}")
            self._page_error = 404
        except requests.exceptions.RequestException as e:
            if response is None:
                self._set_page_content(f"## Error {e}")
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
            header = f"Failed to load page, HTTP Error {self._page_error}\n=======================\n"
        else:
            header = f"Title: {self.page_title}\n=======================\n" if self.page_title else ""
        return header + self.viewport.strip()
