import logging
import sys

from tapeagents.develop import Develop
from tapeagents.dialog_tape import Dialog, SystemStep, UserStep
from tapeagents.llms import LLAMA
from tapeagents.rendering import PrettyRenderer

from .delegate_stack import EXAMPLE_TEXT, ExampleTape, make_analyze_text_chain
from .llama_agent import LLAMAChatBot

logging.basicConfig(level=logging.INFO)


def try_development_app_with_stack(llm):
    tape = ExampleTape(context=EXAMPLE_TEXT)
    agent = make_analyze_text_chain(llm)
    Develop(agent, tape, PrettyRenderer()).launch()


def try_development_app_with_chat(llm):
    tape = Dialog(
        context=None,
        steps=[
            SystemStep(
                content="Respond to the user using the style of Shakespeare books. Be very brief, 50 words max."
            ),
            UserStep(content="Hello, how are you?"),
        ],
    )
    agent = LLAMAChatBot.create(llm)
    Develop(agent, tape, PrettyRenderer()).launch()


if __name__ == "__main__":
    llm = LLAMA(
        base_url="https://api.together.xyz",
        model_name="meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
        parameters=dict(temperature=0.7, max_tokens=512),
    )
    if sys.argv[1] == "chat":
        try_development_app_with_chat(llm)
    elif sys.argv[1] == "stack":
        try_development_app_with_stack(llm)
    else:
        raise ValueError("Unknown mode")
