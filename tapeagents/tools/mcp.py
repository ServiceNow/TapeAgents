import asyncio
import json
import logging
import os
import uuid
from contextlib import AsyncExitStack
from typing import Any, Literal

import nest_asyncio
from mcp import ClientSession, StdioServerParameters, Tool, stdio_client
from mcp.types import CallToolResult
from pydantic import Field

from tapeagents.core import Action, Observation
from tapeagents.environment import ToolCollectionEnvironment
from tapeagents.tool_calling import FunctionSpec

nest_asyncio.apply()
logger = logging.getLogger(__name__)


class MCPClient:
    def __init__(self, config_path: str):
        self.servers = self.load_config(config_path)
        self.sessions: dict[str, ClientSession] = {}
        self.exit_stacks: dict[str, AsyncExitStack] = {}
        self.tools: dict[str, Tool] = {}
        self.tool_to_server: dict[str, str] = {}
        asyncio.run(self.start_servers())

    async def start_servers(self):
        for name, server_params in self.servers.items():
            logger.info(f"Starting MCP server '{name}'")
            await self.connect_to_server(name, server_params)
        logger.info(f"Started {len(self.servers)} MCP servers")

    def load_config(self, config_path) -> dict[str, StdioServerParameters]:
        assert os.path.exists(config_path), f"Config path {config_path} does not exist"
        self.config_path = config_path

        try:
            with open(config_path, "r") as f:
                self.config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse {config_path}, invalid json: {e}")
        try:
            server_configs: dict[str, dict] = self.config["mcpServers"]
            assert isinstance(server_configs, dict), "mcpServers must be a dict"
            assert len(server_configs) > 0, "mcpServers dict is empty"
        except Exception as e:
            raise ValueError(f"Failed to get MCP server configs from {config_path}: {e}")

        servers: dict[str, StdioServerParameters] = {}
        for server_name, server_config_dict in server_configs.items():
            try:
                server_params = StdioServerParameters.model_validate(server_config_dict)
            except Exception as e:
                raise ValueError(f"Failed to parse server config {server_config_dict}: {e}")
            servers[server_name] = server_params
        logger.info(f"Loaded {len(servers)} MCP server configs from {config_path}")
        return servers

    async def connect_to_server(self, server_name: str, server_params: StdioServerParameters):
        try:
            exit_stack = AsyncExitStack()
            stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
            session = await exit_stack.enter_async_context(ClientSession(*stdio_transport))
            await session.initialize()
        except Exception as e:
            logger.exception(f"Failed to start MCP server {server_name} with config {server_params.model_dump()}: {e}")
            raise e

        # List available tools
        response = await session.list_tools()
        for tool in response.tools:
            if tool.name in self.tools:
                raise Exception(
                    f"Tools conflict! Tool {tool.name} already provided by server '{self.tool_to_server[tool.name]}'"
                )

            self.tools[tool.name] = tool
            self.tool_to_server[tool.name] = server_name
        logger.info(f"Connected to MCP server '{server_name}' with tools: {[tool.name for tool in response.tools]}")
        self.sessions[server_name] = session
        self.exit_stacks[server_name] = exit_stack

    async def call_tool(self, tool_name: str, tool_args: dict[str, Any]) -> CallToolResult:
        try:
            server_name = self.tool_to_server[tool_name]
        except KeyError:
            raise Exception(f"Tool {tool_name} not found in any MCP server")

        try:
            session = self.sessions[server_name]
            result = await session.call_tool(tool_name, tool_args)
        except Exception as e:
            logger.exception(f"Error calling tool {tool_name}: {e}")
            raise e
        return result

    def close(self):
        # TODO: close all sessions and exit stacks
        pass


class MCPToolCall(Action):
    kind: Literal["mcp_tool_call"] = "mcp_tool_call"  # type: ignore
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    input: dict[str, Any]


class MCPToolResult(Observation):
    kind: Literal["mcp_tool_result"] = "mcp_tool_result"  # type: ignore
    tool_use_id: str
    result: CallToolResult


class MCPEnvironment(ToolCollectionEnvironment):
    client: MCPClient
    tools: dict[str, FunctionSpec]

    def __init__(self, config_path: str = "", client: MCPClient | None = None) -> None:
        self.client = client or MCPClient(config_path)
        self.tools = {
            tool.name: FunctionSpec(name=tool.name, description=tool.description or "", parameters=tool.inputSchema)
            for tool in self.client.tools.values()
        }

    def actions(self) -> tuple[type[Action] | FunctionSpec, ...]:
        return tuple(self.tools.values())

    def tools_description(self) -> str:
        return "\n".join(f"{spec.name} - {spec.description}" for spec in self.tools.values())

    def step(self, action: MCPToolCall) -> MCPToolResult:
        result = asyncio.run(self.client.call_tool(action.name, action.input))
        return MCPToolResult(tool_use_id=action.id, result=result)

    def close(self) -> None:
        self.client.close()
