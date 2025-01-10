import tempfile
from pathlib import Path

from tapeagents.batch import generate_tapes
from tapeagents.dialog_tape import AssistantStep, DialogTape, SystemStep, UserStep
from tapeagents.environment import EmptyEnvironment
from tapeagents.llms import LLM, TrainableLLM

from .llama_agent import LLAMAChatBot
from .llama_user import LLAMAUserModel


def try_continue_tapes(llm: LLM):
    agent = LLAMAChatBot.create(llm)
    follow_up_user = LLAMAUserModel(
        instruction="come up with a follow-up message that a user may write to their conversational assistant",
        llms={"default": llm},
    )
    change_topic_user = LLAMAUserModel(
        instruction="change the topic of the conversation to something completely different",
        llms={"default": llm},
    )

    tape = DialogTape(
        context=None,
        steps=[
            SystemStep(content="You are a helpful assistant."),
            UserStep(content="Tell me about ServiceNow"),
            AssistantStep(content="It is a very successful enterprise software company"),
        ],
    )

    layers_conf = [
        [(follow_up_user, 3)],
        [(follow_up_user, 3), (change_topic_user, 3)],
    ]
    path = Path(tempfile.mktemp())
    generate_tapes(
        agent,
        [tape],
        EmptyEnvironment(),
        layers_conf,  # type: ignore
        path,
    )

    print(f"Go check the output at {path}")


if __name__ == "__main__":
    try_continue_tapes(
        TrainableLLM(
            base_url="https://api.together.xyz",
            model_name="meta-llama/Llama-3-8b-chat-hf",
            parameters=dict(temperature=0.7, max_tokens=512),
        )
    )
