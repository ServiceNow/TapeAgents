from typing import Any

from litellm import ChatCompletionMessageToolCall
from mcp import Tool
from mcp.types import CallToolResult, TextContent

from tapeagents.agent import Agent
from tapeagents.core import Tape
from tapeagents.dialog_tape import UserStep
from tapeagents.llms import MockLLM
from tapeagents.nodes import StandardNode
from tapeagents.tools.mcp import MCPClient, MCPEnvironment, MCPToolCall, MCPToolResult

MOCK_TOOLS = {
    "server1": [
        Tool(
            name="calculator",
            description="Performs basic math operations",
            inputSchema={
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["add", "subtract"]},
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["operation", "a", "b"],
            },
        ),
        Tool(
            name="greeter",
            description="Greets a person",
            inputSchema={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        ),
    ],
    "server2": [
        Tool(
            name="echo",
            description="Echoes back the input",
            inputSchema={"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]},
        )
    ],
}


class MockMCPClient(MCPClient):
    def __init__(self, config_path: str):
        self.tools = {}
        self.tool_to_server = {}

        # Register mock tools
        for server_name, tools in MOCK_TOOLS.items():
            for tool in tools:
                self.tools[tool.name] = tool
                self.tool_to_server[tool.name] = server_name

    async def call_tool(self, tool_name: str, tool_args: dict[str, Any]) -> CallToolResult:
        self.check_tool_exists(tool_name)
        # Simulate tool behavior
        if tool_name == "calculator":
            result = (
                tool_args["a"] + tool_args["b"] if tool_args["operation"] == "add" else tool_args["a"] - tool_args["b"]
            )
            return CallToolResult(content=[TextContent(type="text", text=str(result))])
        elif tool_name == "greeter":
            return CallToolResult(content=[TextContent(type="text", text=f"Hello, {tool_args['name']}!")])
        elif tool_name == "echo":
            return CallToolResult(content=[TextContent(type="text", text=tool_args["message"])])
        else:
            raise Exception(f"Tool {tool_name} not implemented")


def test_mcp_env():
    env = MCPEnvironment(client=MockMCPClient("dummy_config.json"))

    obs = env.step(MCPToolCall(name="calculator", input={"operation": "add", "a": 5, "b": 3}))
    assert obs.kind == "mcp_tool_result"
    assert obs.tool_use_id is not None
    assert isinstance(obs.result.content[0], TextContent)
    assert obs.result.content[0].text == "8"

    obs = env.step(MCPToolCall(name="greeter", input={"name": "Alice"}))
    assert obs.kind == "mcp_tool_result"
    assert obs.tool_use_id is not None
    assert isinstance(obs.result.content[0], TextContent)
    assert obs.result.content[0].text == "Hello, Alice!"

    obs = env.step(MCPToolCall(name="echo", input={"message": "Hello, World!"}))
    assert obs.kind == "mcp_tool_result"
    assert obs.tool_use_id is not None
    assert isinstance(obs.result.content[0], TextContent)
    assert obs.result.content[0].text == "Hello, World!"


def test_wrong_tool():
    env = MCPEnvironment(client=MockMCPClient("dummy_config.json"))
    obs = env.step(MCPToolCall(name="non_existent_tool", input={}))
    assert obs.kind == "mcp_tool_result"
    assert obs.tool_use_id is not None
    assert isinstance(obs.result.content[0], TextContent)
    assert obs.result.isError
    assert obs.result.content[0].text == "Tool non_existent_tool not found"


def test_incorrect_tool_args():
    env = MCPEnvironment(client=MockMCPClient("dummy_config.json"))
    obs = env.step(MCPToolCall(name="calculator", input={"operator": "add", "a": 5, "b": 3}))
    assert obs.kind == "mcp_tool_result"
    assert obs.tool_use_id is not None
    assert isinstance(obs.result.content[0], TextContent)
    assert obs.result.isError
    assert obs.result.content[0].text == "Error executing tool calculator: KeyError 'operation'"

    obs = env.step(MCPToolCall(name="calculator", input={"operation": "add", "a": 5}))
    assert obs.kind == "mcp_tool_result"
    assert obs.tool_use_id is not None
    assert isinstance(obs.result.content[0], TextContent)
    assert obs.result.isError
    assert obs.result.content[0].text == "Error executing tool calculator: KeyError 'b'"


def test_prompt_with_tool_calls():
    env = MCPEnvironment(client=MockMCPClient("dummy_config.json"))
    llm = MockLLM()
    node = StandardNode(use_known_actions=True, use_function_calls=True)
    agent = Agent.create(llms=llm, nodes=[node], known_actions=env.actions())
    tape = Tape(steps=[UserStep(content="test")])

    prompt = agent.make_prompt(tape)

    assert prompt.tools
    assert len(prompt.tools) == 3

    assert prompt.tools[0]["function"]["name"] == "calculator"
    assert prompt.tools[0]["function"]["description"] == "Performs basic math operations"
    assert prompt.tools[0]["function"]["parameters"] == {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["add", "subtract"]},
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "required": ["operation", "a", "b"],
    }

    assert prompt.tools[1]["function"]["name"] == "greeter"
    assert prompt.tools[1]["function"]["description"] == "Greets a person"
    assert prompt.tools[1]["function"]["parameters"] == {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    assert prompt.tools[2]["function"]["name"] == "echo"
    assert prompt.tools[2]["function"]["description"] == "Echoes back the input"
    assert prompt.tools[2]["function"]["parameters"] == {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    }


def test_tool_use_parsing():
    env = MCPEnvironment(client=MockMCPClient("dummy_config.json"))
    llm = MockLLM(
        mock_tool_calls=[
            ChatCompletionMessageToolCall(
                id="test1",
                function={
                    "name": "calculator",
                    "arguments": {"operation": "add", "a": 9, "b": 2},
                },
            )
        ]
    )
    node = StandardNode(use_known_actions=True, use_function_calls=True)
    agent = Agent.create(llms=llm, nodes=[node], known_actions=env.actions())
    tape = Tape(steps=[UserStep(content="test")])
    event = None
    for event in agent.run(tape):
        if event.final_tape:
            break
    assert event and event.final_tape
    agent_tape = event.final_tape
    assert len(agent_tape) == 3
    call_step = agent_tape[-1]

    assert isinstance(call_step, MCPToolCall)
    assert call_step.name == "calculator"
    assert len(call_step.input) == 3
    assert call_step.input["operation"] == "add"
    assert call_step.input["a"] == 9
    assert call_step.input["b"] == 2

    final_tape = env.react(agent_tape)
    assert len(final_tape) == 4
    obs = final_tape.steps[-1]
    assert isinstance(obs, MCPToolResult)
    assert obs.tool_use_id == call_step.id
    assert isinstance(obs.result.content[0], TextContent)
    assert obs.result.content[0].text == "11"
