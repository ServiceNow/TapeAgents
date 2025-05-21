# Add lifespan support for startup/shutdown with strong typing

from mcp.server.fastmcp import FastMCP

from tapeagents.tools.web_search import _web_search

# Pass lifespan to server
mcp = FastMCP("Serper Web Search")


@mcp.tool()
def web_search_tool(query: str) -> list[dict]:
    """
    Search the web for a given query, return a list of search result dictionaries.
    """
    return _web_search(query, max_results=5, retry_pause=5, attempts=3)


if __name__ == "__main__":
    mcp.run()
