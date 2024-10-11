import tempfile
from pathlib import Path

from tapeagents.batch import ObsLayerConfig, batch_add_observations
from tapeagents.dialog_tape import AssistantStep, DialogTape, SystemStep, UserStep
from tapeagents.io import stream_yaml_tapes
from tapeagents.llms import LLM, TrainableLLM

from .llama_user import LLAMAUserModel


def try_batch_add_observations(llm: LLM):
    user_model = LLAMAUserModel(
        instruction="come up with a follow-up message that a user may write to their conversational assistant",
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

    path = Path(tempfile.mktemp())
    path_user_tapes = Path(tempfile.mktemp())
    with stream_yaml_tapes(path) as dumper:
        with stream_yaml_tapes(path_user_tapes) as dumper_user:
            layer_conf = ObsLayerConfig(obs_makers=[(user_model, 3)])
            for new_tape, obs_maker_tape in batch_add_observations([tape], layer_conf):
                dumper.save(new_tape)
                dumper_user.save(obs_maker_tape)

    with open(path) as src:
        print(src.read())
    with open(path_user_tapes) as src:
        print(src.read())


if __name__ == "__main__":
    try_batch_add_observations(
        TrainableLLM(
            base_url="https://api.together.xyz",
            model_name="meta-llama/Llama-3-8b-chat-hf",
            parameters=dict(temperature=0.7, max_tokens=512),
        )
    )
