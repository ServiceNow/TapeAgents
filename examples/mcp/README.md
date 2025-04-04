Test tool server:
```bash
uv run mcp dev examples/mcp/tool.py
```

Test resource server:
```bash
uv run mcp dev examples/mcp/resource.py
```

Test the agent that uses 5 mcp servers to solve 1 task from gaia benchmark:
```bash
uv run examples/mcp/agent_with_mcp.py
```

MCP Inspector will be available at http://127.0.0.1:6274