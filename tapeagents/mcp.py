import asyncio
import json
import logging
import os
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Any, Optional

import nest_asyncio
from mcp import ClientSession, StdioServerParameters, Tool as MCPTool, stdio_client
from mcp.types import CallToolResult, TextContent

from tapeagents.config import force_cache
from tapeagents.core import Action, LLMOutputParsingFailureAction, Observation
from tapeagents.environment import ToolCollectionEnvironment
from tapeagents.tool_calling import FunctionSpec, ToolCallAction, ToolResult, ToolSpec
from tapeagents.tools.base import BaseTool
from tapeagents.tools.tool_cache import add_to_cache, get_from_cache
from tapeagents.utils import FatalError

nest_asyncio.apply()
logger = logging.getLogger(__name__)


class NoTool(Exception):
    """Raised when a tool is not found in the MCP client"""

    pass


class MCPClient:
    def __init__(self, config_path: str, use_cache: bool = False) -> None:
        self.servers = self.load_config(config_path)
        self.sessions: dict[str, ClientSession] = {}
        self.exit_stacks: dict[str, AsyncExitStack] = {}
        self.tools: dict[str, MCPTool] = {}
        self.tool_to_server: dict[str, str] = {}
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.use_cache = use_cache
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
                server_config_dict = self.prepare_env_vars(server_config_dict)
                server_params = StdioServerParameters.model_validate(server_config_dict)
            except Exception as e:
                raise ValueError(f"Failed to parse server config {server_config_dict}: {e}")
            servers[server_name] = server_params
        logger.info(f"Loaded {len(servers)} MCP server configs from {config_path}")
        return servers

    def prepare_env_vars(self, server_config_dict: dict) -> dict:
        if server_env := server_config_dict.get("env"):
            for env_var, env_value in server_env.items():
                if env_var in os.environ and not env_value:  # reuse existing env var value if not set in config
                    logger.info(f"Set mcp server env var {env_var} from current environment")
                    server_config_dict["env"][env_var] = os.environ[env_var]
        return server_config_dict

    async def connect_to_server(self, server_name: str, server_params: StdioServerParameters):
        try:
            exit_stack = AsyncExitStack()
            stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
            session = await exit_stack.enter_async_context(
                ClientSession(*stdio_transport, read_timeout_seconds=timedelta(seconds=8))
            )
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
        server_name = self.check_tool_exists(tool_name)
        # Current implementation of cache assumes tool calls are deterministic and do not alter state
        if self.use_cache:
            tool_name_key = f"mcp.{server_name}.{tool_name}"
            result = get_from_cache(tool_name_key, args=(), kwargs=tool_args)
            if not result and force_cache():
                raise FatalError(f"Cache is forced but no cache entry found for {tool_name_key}({tool_args})")
            if not result:
                result = await self._call_tool(server_name, tool_name, tool_args)
                add_to_cache(tool_name_key, args=(), kwargs=tool_args, result=result.model_dump(exclude_none=True))
            else:
                result = CallToolResult(**result)
        else:
            result = await self._call_tool(server_name, tool_name, tool_args)

        return result

    async def _call_tool(self, server_name: str, tool_name: str, tool_args: dict[str, Any]) -> CallToolResult:
        try:
            session = self.sessions[server_name]
            result = await session.call_tool(tool_name, tool_args)
        except Exception as e:
            logger.exception(f"Error calling tool {tool_name}: {e}")
            raise e
        return result

    def check_tool_exists(self, tool_name):
        try:
            server_name = self.tool_to_server[tool_name]
        except KeyError:
            raise NoTool(f"Tool {tool_name} not found in any of the MCP servers")
        return server_name

    async def close(self) -> None:
        """Clean up resources."""
        async with self._cleanup_lock:
            for server_name, exit_stack in self.exit_stacks.items():
                try:
                    await exit_stack.aclose()
                    del self.sessions[server_name]
                except Exception:
                    pass


class MCPEnvironment(ToolCollectionEnvironment):
    client: MCPClient | None
    tools: list[BaseTool]

    def __init__(
        self,
        tools: Optional[list[BaseTool]] = None,
        config_path: str = "",
        use_cache: bool = False,
        client: MCPClient | None = None,
    ) -> None:
        super().__init__(tools=tools or [])
        self.client = client or (MCPClient(config_path=config_path, use_cache=use_cache) if config_path else None)
        if not self.client and not self.tools:
            raise ValueError("Tools or MCP client config_path must be provided")
        if self.client:
            self.client.use_cache = use_cache
            self.tools.extend(
                [
                    ToolSpec(
                        function=FunctionSpec(
                            name=tool.name, description=tool.description or "", parameters=tool.inputSchema
                        )
                    )
                    for tool in self.client.tools.values()
                ]
            )

    def actions(self) -> tuple[type[Action] | ToolSpec, ...]:
        actions = super().actions()
        tool_specs = [t for t in self.tools if isinstance(t, ToolSpec)]
        return actions + tuple(tool_specs)

    def step(self, action: Action) -> Observation:
        if not isinstance(action, ToolCallAction):
            return super().step(action)
        if isinstance(action, LLMOutputParsingFailureAction):
            return ToolResult(tool_call_id="", content="Try again")
        try:
            assert self.client is not None, "MCPClient is not initialized"
            result = asyncio.run(self.client.call_tool(action.function.name, action.function.arguments))
        except NoTool:
            logger.exception(f"Tool {action.function.name} not found in MCP client")
            result = CallToolResult(
                content=[TextContent(type="text", text=f"Tool {action.function.name} not found")], isError=True
            )
        except KeyError as e:
            logger.exception(f"KeyError when executing MCP tool call: {e}")
            result = CallToolResult(
                content=[TextContent(type="text", text=f"Error executing tool {action.function.name}: KeyError {e}")],
                isError=True,
            )
        except Exception as e:
            logger.exception(f"Error executing MCP tool call: {e}")
            result = CallToolResult(
                content=[TextContent(type="text", text=f"Error executing tool {action.function.name}: {str(e)}")],
                isError=True,
            )
        return ToolResult(tool_call_id=action.id, content=result)

    def close(self) -> None:
        if self.client is not None:
            try:
                asyncio.run(self.client.close())
            except Exception:
                pass
