import tempfile
from pathlib import Path

from tapeagents.batch import generate_tapes
from tapeagents.dialog import AssistantStep, Dialog, SystemStep, UserStep
from tapeagents.environment import EmptyEnvironment
from tapeagents.llms import LLAMA, LLAMAConfig

from .llama_agent import LLAMAChatBot
from .llama_user import LLAMAUserModel


def try_continue_tapes(llama_config: LLAMAConfig):
    agent = LLAMAChatBot(LLAMA(llama_config))
    follow_up_user = LLAMAUserModel(
        "come up with a follow-up message that a user may write to their conversational assistant",
        LLAMA(llama_config),
    )
    change_topic_user = LLAMAUserModel(
        "change the topic of the conversation to something completely different",
        LLAMA(llama_config), 
    )

    tape = Dialog(
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
        layers_conf,
        path,
    )

    print(f"Go check the output at {path}")


if __name__ == "__main__":
    try_continue_tapes(
        LLAMAConfig(
            base_url="https://api.together.xyz",
            model_name="meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
            # base_url="http://localhost:8000",
            # model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            parameters=dict(
                temperature=0.7,
                max_tokens=512
            )
        )
    )
