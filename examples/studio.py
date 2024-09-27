import logging
import sys

from tapeagents.studio import Studio
from tapeagents.dialog_tape import DialogTape, SystemStep, UserStep
from tapeagents.llms import TrainableLLM
from tapeagents.rendering import PrettyRenderer

from .delegate_stack import EXAMPLE_TEXT, ExampleTape, make_analyze_text_chain
from .llama_agent import LLAMAChatBot

logging.basicConfig(level=logging.INFO)


def try_studio_with_stack(llm):
    tape = ExampleTape(context=EXAMPLE_TEXT)
    agent = make_analyze_text_chain(llm)
    Studio(agent, tape, PrettyRenderer()).launch()


def try_studio_with_chat(llm):
    tape = DialogTape(
        context=None,
        steps=[
            SystemStep(
                content="Respond to the user using the style of Shakespeare books. Be very brief, 50 words max."
            ),
            UserStep(content="Hello, how are you?"),
        ],
    )
    agent = LLAMAChatBot.create(llm)
    Studio(agent, tape, PrettyRenderer()).launch()


if __name__ == "__main__":
    llm = TrainableLLM(
        base_url="https://api.together.xyz",
        model_name="meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
        parameters=dict(temperature=0.7, max_tokens=512),
    )
    if sys.argv[1] == "chat":
        try_studio_with_chat(llm)
    elif sys.argv[1] == "stack":
        try_studio_with_stack(llm)
    else:
        raise ValueError("Unknown mode")
