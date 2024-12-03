from tapeagents.demo import Demo
from tapeagents.dialog_tape import DialogContext, DialogTape
from tapeagents.llms import LiteLLM
from tapeagents.renderers.basic import BasicRenderer

from .openai_function_calling import TOOL_SCHEMAS, FunctionCallingAgent, MockToolEnvironment


def try_openai_function_calling_interactive_demo():
    llm = LiteLLM(model_name="gpt-3.5-turbo")
    agent = FunctionCallingAgent.create(llm)
    tape = DialogTape(context=DialogContext(tools=TOOL_SCHEMAS), steps=[])
    environment = MockToolEnvironment(llm)
    demo = Demo(agent, tape, environment, BasicRenderer())
    demo.launch()


if __name__ == "__main__":
    try_openai_function_calling_interactive_demo()
