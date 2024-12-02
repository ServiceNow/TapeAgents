import logging

from langchain_community.tools.tavily_search import TavilySearchResults

from tapeagents.demo import Demo
from tapeagents.dialog_tape import DialogContext, DialogTape
from tapeagents.environment import ToolEnvironment
from tapeagents.llms import LiteLLM
from tapeagents.renderers.basic import BasicRenderer

from .annotator import GroundednessAnnotator
from .openai_function_calling import FunctionCallingAgent

logging.basicConfig(level=logging.INFO)


def main():
    small_llm = LiteLLM(model_name="gpt-3.5-turbo")
    big_llm = LiteLLM(model_name="gpt-4-turbo")
    agent = FunctionCallingAgent.create(small_llm)
    environment = ToolEnvironment(tools=[TavilySearchResults()])
    init_dialog = DialogTape(context=DialogContext(tools=environment.get_tool_schemas()), steps=[])
    demo = Demo(agent, init_dialog, environment, BasicRenderer(), GroundednessAnnotator.create(big_llm))
    demo.launch()


if __name__ == "__main__":
    main()
