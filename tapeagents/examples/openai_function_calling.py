from pprint import pprint
from typing import Generator

from litellm.utils import ChatCompletionMessageToolCall
from pydantic import TypeAdapter

from tapeagents.agent import Agent
from tapeagents.core import Prompt
from tapeagents.dialog import (
    AssistantStep,
    Dialog,
    DialogContext,
    DialogEvent,
    ToolCalls,
    ToolResult,
    ToolSpec,
    UserStep,
)
from tapeagents.environment import (
    ExternalObservationNeeded,
    MockToolEnvironment,
)
from tapeagents.llms import LiteLLM, LLMMessage, LLMStream
from tapeagents.runtime import main_loop


class FunctionCallingAgent(Agent[Dialog]):
    def make_prompt(self, tape: Dialog):
        steps = list(tape.steps)
        for i in range(len(steps)):
            if isinstance(steps[i], ToolResult):
                steps[i] = steps[i].model_copy()
                steps[i].content = str(steps[i].content)  # type: ignore
        assert tape.context
        return Prompt(tools=[t.model_dump() for t in tape.context.tools], messages=[s.llm_dict() for s in steps])

    def generate_steps(self, _, llm_stream: LLMStream):
        m = llm_stream.get_message()
        if m.content:
            yield AssistantStep(content=m.content)
        elif m.tool_calls:
            assert all(isinstance(tc, ChatCompletionMessageToolCall) for tc in m.tool_calls)
            yield ToolCalls(tool_calls=m.tool_calls)
        else:
            raise ValueError(f"don't know what to do with message {m}")


TOOL_SCHEMAS = TypeAdapter(list[ToolSpec]).validate_python(
    [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
)


def try_openai_function_calling():
    llm = LiteLLM(model_name="gpt-3.5-turbo")
    agent = FunctionCallingAgent.create(llm)
    dialog = Dialog(context=DialogContext(tools=TOOL_SCHEMAS), steps=[])

    for event in agent.run(dialog.append(UserStep(content="What's the weather like in San Francisco, Tokyo"))):
        if event.step:
            print(event.step)

    assert event.final_tape
    dialog = event.final_tape
    tool_call_step = dialog.steps[-1]
    assert isinstance(tool_call_step, ToolCalls)
    tool_calls = tool_call_step.tool_calls

    dialog = dialog.append(
        ToolResult(
            tool_call_id=tool_calls[0].id,
            content="Cloudy, 13C",
        )
    ).append(
        ToolResult(
            tool_call_id=tool_calls[1].id,
            content="Sunny, 23C",
        )
    )

    for event in agent.run(dialog):
        if event.step:
            print(event.step)


def try_openai_function_callling_with_environment():
    llm = LiteLLM(model_name="gpt-3.5-turbo")
    agent = FunctionCallingAgent.create(llm)
    dialog = Dialog(
        context=DialogContext(tools=TOOL_SCHEMAS),
        steps=[UserStep(content="What's the weather like in San Francisco, Tokyo")],
    )
    environment = MockToolEnvironment(llm)
    for s in dialog.steps:
        print("USER STEP")
        pprint(s.model_dump(exclude_none=True))
    try:
        for event in main_loop(agent, dialog, environment, max_loops=3):
            if ae := event.agent_event:
                if ae.step:
                    print("AGENT STEP")
                    pprint(ae.step.model_dump(exclude_none=True))
            elif event.observation:
                print("OBSERVATION")
                pprint(event.observation.model_dump(exclude_none=True))
            else:
                raise ValueError("Must be something in the event")
    except ExternalObservationNeeded as e:
        assert isinstance(e.action, AssistantStep)
    print("Stopping, next user message is needed")


if __name__ == "__main__":
    try_openai_function_calling()
    try_openai_function_callling_with_environment()
