import logging
import sys

from tapeagents.dialog_tape import DialogTape, SystemStep, UserStep
from tapeagents.llms import LLM, TrainableLLM
from tapeagents.renderers.pretty import PrettyRenderer
from tapeagents.studio import Studio

from .delegate_stack import EXAMPLE_TEXT, ExampleTape, make_analyze_text_chain
from .llama_agent import LLAMAChatBot

logging.basicConfig(level=logging.INFO)


def try_studio_with_stack(llm: LLM):
    """
    Launches the studio with the stack of agents that analyze the text for nouns and irregular verbs.
    """
    tape = ExampleTape(context=EXAMPLE_TEXT)
    agent = make_analyze_text_chain(llm)
    Studio(agent, tape, PrettyRenderer()).launch()


def try_studio_with_chat(llm: LLM):
    """
    Launches the studio with the agent that responds to the user using the style of Shakespeare books.
    """
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


# Interactive Gradio demo of the agent that could be changed in runtime.
if __name__ == "__main__":
    llm = TrainableLLM(
        base_url="https://api.together.xyz",
        model_name="meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
        parameters=dict(temperature=0.7, max_tokens=512),
    )
    if len(sys.argv) < 2:
        try_studio_with_chat(llm)
    elif sys.argv[1] == "chat":
        try_studio_with_chat(llm)
    elif sys.argv[1] == "stack":
        try_studio_with_stack(llm)
    else:
        raise ValueError(f"Unknown mode, {sys.argv[1]}")
