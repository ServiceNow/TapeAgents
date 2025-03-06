import json
from pprint import pprint

from litellm.types.utils import ChatCompletionMessageToolCall
from pydantic import TypeAdapter

from tapeagents.agent import Agent
from tapeagents.core import Action, Prompt
from tapeagents.dialog_tape import (
    AssistantStep,
    DialogContext,
    DialogTape,
    UserStep,
)
from tapeagents.environment import (
    Environment,
    ExternalObservationNeeded,
)
from tapeagents.llms import LiteLLM, LLMStream
from tapeagents.orchestrator import main_loop
from tapeagents.prompting import tape_to_messages
from tapeagents.tool_calling import ToolCalls, ToolResult, ToolSpec

MOCK_TOOL_ENV_PROMPT_TEMPLATE = """You will generate result of the following function call

{function_call}

You will output JSON of the following structure:
{{
    "result": ...
}}

You will output just the JSON and nothing else. Go!
"""


class MockToolEnvironment(Environment):
    def __init__(self, llm: LiteLLM):
        self.llm = llm

    def react(self, tape: DialogTape) -> DialogTape:
        # TODO: move prompting to a separate agent?
        action = tape.steps[-1]
        assert isinstance(action, Action)
        if isinstance(action, ToolCalls):
            for tc in action.tool_calls:
                prompt_text = MOCK_TOOL_ENV_PROMPT_TEMPLATE.format(function_call=tc.function)
                messages = [{"role": "user", "content": prompt_text}]
                for event in self.llm.generate(Prompt(messages=messages)):
                    completion = event.output
                    if completion and not isinstance(completion, str):
                        completion = completion.content
                    if completion:
                        result_json = json.loads(completion)
                        if "result" not in result_json:
                            raise ValueError("Result JSON should have 'result' key")
                        observation = ToolResult(content=json.dumps(result_json["result"]), tool_call_id=tc.id)
                        tape = tape.append(observation)
        elif isinstance(action, AssistantStep):
            self.raise_external_observation_needed(action)
        else:
            self.raise_unexpected_action(action)
        return tape


class FunctionCallingAgent(Agent[DialogTape]):
    def make_prompt(self, tape: DialogTape):
        assert tape.context
        return Prompt(tools=[t.model_dump() for t in tape.context.tools], messages=tape_to_messages(tape))

    def generate_steps(self, _, llm_stream: LLMStream):
        o = llm_stream.get_output()
        if o.content:
            yield AssistantStep(content=o.content)
        elif o.tool_calls:
            assert all(isinstance(tc, ChatCompletionMessageToolCall) for tc in o.tool_calls)
            yield ToolCalls.from_llm_output(o)
        else:
            raise ValueError(f"don't know what to do with message {o}")


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
    tape = DialogTape(context=DialogContext(tools=TOOL_SCHEMAS), steps=[])

    for event in agent.run(tape.append(UserStep(content="What's the weather like in San Francisco, Tokyo"))):
        if event.step:
            print(event.step)

    assert event.final_tape
    tape = event.final_tape
    tool_call_step = tape.steps[-1]
    assert isinstance(tool_call_step, ToolCalls)
    tool_calls = tool_call_step.tool_calls

    tape = tape.append(
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

    for event in agent.run(tape):
        if event.step:
            print(event.step)


def try_openai_function_callling_with_environment():
    llm = LiteLLM(model_name="gpt-3.5-turbo")
    agent = FunctionCallingAgent.create(llm)
    tape = DialogTape(
        context=DialogContext(tools=TOOL_SCHEMAS),
        steps=[UserStep(content="What's the weather like in San Francisco, Tokyo")],
    )
    environment = MockToolEnvironment(llm)
    for s in tape.steps:
        print("USER STEP")
        pprint(s.model_dump(exclude_none=True))
    try:
        for event in main_loop(agent, tape, environment, max_loops=3):
            if ae := event.agent_event:
                if ae.step:
                    print("AGENT STEP")
                    pprint(ae.step.model_dump(exclude_none=True))
            elif event.observation:
                print("OBSERVATION")
                pprint(event.observation.model_dump(exclude_none=True))
    except ExternalObservationNeeded as e:
        assert isinstance(e.action, AssistantStep)
    print("Stopping, next user message is needed")


if __name__ == "__main__":
    try_openai_function_calling()
    try_openai_function_callling_with_environment()
