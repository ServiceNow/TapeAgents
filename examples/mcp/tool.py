# Add lifespan support for startup/shutdown with strong typing
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from mcp.server.fastmcp import Context, FastMCP

from tapeagents.tools.simple_browser import SimpleTextBrowser


@dataclass
class AppContext:
    browser: SimpleTextBrowser


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with type-safe context"""
    # Initialize on startup
    browser = SimpleTextBrowser()
    try:
        yield AppContext(browser=browser)
    finally:
        # Save cache on shutdown
        browser.flush_cache()


# Pass lifespan to server
mcp = FastMCP("Simple Browser Server", lifespan=app_lifespan)


# Access type-safe lifespan context in tools
@mcp.tool()
def get_page(url: str, ctx: Context) -> str:
    """Tool that returns the text of the document at the given URL"""
    browser: SimpleTextBrowser = ctx.request_context.lifespan_context.browser
    text, pages, error = browser.get_page(url)
    if error:
        raise ValueError(f"Error fetching document: {error}")
    return text


@mcp.tool()
def echo_tool(message: str) -> str:
    """Echo a message as a tool"""
    return f"Tool echo: {message}"


if __name__ == "__main__":
    mcp.run()
