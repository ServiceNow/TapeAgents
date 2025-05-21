import asyncio
import json
import logging
import os
from typing import Any, Optional

import nest_asyncio
from mcp import ClientSessionGroup, StdioServerParameters, Tool as MCPTool
from mcp.types import CallToolResult, TextContent

from tapeagents.config import force_cache
from tapeagents.core import Action, LLMOutputParsingFailureAction, Observation
from tapeagents.environment import ToolCollectionEnvironment
from tapeagents.tool_calling import FunctionSpec, ToolCallAction, ToolResult, ToolSpec
from tapeagents.tools.base import BaseTool
from tapeagents.tools.tool_cache import add_to_cache_async, get_from_cache
from tapeagents.utils import FatalError

logger = logging.getLogger(__name__)


class NoTool(Exception):
    """Raised when a tool is not found in the MCP client"""

    pass


class MCPClient:
    def __init__(self, config_path: str, use_cache: bool = False, read_timeout_seconds: int = 10) -> None:
        self.config_name = os.path.basename(config_path)
        self.servers = self.load_config(config_path)
        self.tools: dict[str, MCPTool] = {}
        self.use_cache = use_cache
        self.read_timeout_seconds = read_timeout_seconds

    async def start_servers(self):
        self.client_session_group = await ClientSessionGroup().__aenter__()
        for name, server_params in self.servers.items():
            logger.info(f"Starting MCP server '{name}'")
            await self.client_session_group.connect_to_server(server_params)
        self.tools = self.client_session_group.tools
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

    async def call_tool(self, tool_name: str, tool_args: dict[str, Any]) -> CallToolResult:
        self.client_session_group._tool_to_session[tool_name]
        # Current implementation of cache assumes tool calls are deterministic and do not alter state
        if self.use_cache:
            tool_name_key = f"mcp.{self.config_name}.{tool_name}"
            result = get_from_cache(tool_name_key, args=(), kwargs=tool_args)
            if not result and force_cache():
                raise FatalError(f"Cache is forced but no cache entry found for {tool_name_key}({tool_args})")
            if not result:
                result = await self.client_session_group.call_tool(tool_name, tool_args)
                await add_to_cache_async(
                    tool_name_key, args=(), kwargs=tool_args, result=result.model_dump(exclude_none=True)
                )
            else:
                result = CallToolResult(**result)
        else:
            result = await self.client_session_group.call_tool(tool_name, tool_args)

        return result

    async def close(self) -> None:
        try:
            await self.client_session_group.__aexit__(None, None, None)
        except Exception:
            pass
        logger.info("Closed all MCP servers")


class MCPEnvironment(ToolCollectionEnvironment):
    def __init__(
        self,
        config_path: str = "",
        other_tools: Optional[list[BaseTool]] = None,
        use_cache: bool = False,
        read_timeout_seconds: int = 10,
        client: MCPClient | None = None,
        async_mode: bool = False,
    ) -> None:
        super().__init__(tools=other_tools or [])
        logger.info(f"Initializing MCPEnvironment with config_path: {config_path}")
        self.client = client or MCPClient(
            config_path=config_path, use_cache=use_cache, read_timeout_seconds=read_timeout_seconds
        )

        if not self.client and not self.tools:
            raise ValueError("Tools or MCP client config_path must be provided")
        self.async_mode = async_mode
        self.client.use_cache = use_cache
        self.loop = asyncio.get_event_loop()
        if not async_mode:
            nest_asyncio.apply()
            self.loop.run_until_complete(self.client.start_servers())
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

    async def initialize(self) -> None:
        await self.client.start_servers()
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
            result = self.loop.run_until_complete(
                self.client.call_tool(action.function.name, action.function.arguments)
            )
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
        super().close()
        if self.client is not None:
            try:
                self.loop.run_until_complete(self.client.close())
            except Exception as e:
                logger.warning(f"Failed to close MCP client properly: {e}")

    async def astep(self, action: Action) -> Observation:
        if not isinstance(action, ToolCallAction):
            return await super().astep(action)
        if isinstance(action, LLMOutputParsingFailureAction):
            return ToolResult(tool_call_id="", content="Try again")
        try:
            assert self.client is not None, "MCPClient is not initialized"
            result = await self.client.call_tool(action.function.name, action.function.arguments)
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

    async def aclose(self) -> None:
        await super().aclose()
        await self.client.close()
