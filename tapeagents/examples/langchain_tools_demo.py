from langchain_community.tools.tavily_search import TavilySearchResults
from tapeagents.demo import Demo
from tapeagents.dialog import Dialog, DialogContext
from tapeagents.environment import LangchainToolEnvironment
from tapeagents.llms import LiteLLM
from tapeagents.rendering import BasicRenderer

from .openai_function_calling import FunctionCallingAgent


def try_langchain_tools_demo():
    llm = LiteLLM(model_name="gpt-3.5-turbo")
    agent = FunctionCallingAgent.create(llm)
    environment = LangchainToolEnvironment(tools=[TavilySearchResults()])
    init_dialog = Dialog(context=DialogContext(tools=environment.get_tool_schemas()), steps=[])
    demo = Demo(agent, init_dialog, environment, BasicRenderer())
    demo.launch()


if __name__ == "__main__":
    try_langchain_tools_demo()
