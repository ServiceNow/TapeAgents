import sys

from tapeagents.llms import LLM, TrainableLLM
from tapeagents.renderers.pretty import PrettyRenderer
from tapeagents.studio import Studio
from tapeagents.team import TeamAgent, TeamTape


def try_chat(llm: LLM, studio: bool):
    # equilavent of https://microsoft.github.io/autogen/docs/tutorial/introduction
    comedy_duo = TeamAgent.create_initiator(
        name="Joe",
        llm=llm,
        system_prompt="Your name is Joe and you are a part of a duo of comedians.",
        teammate=TeamAgent.create(
            name="Cathy", llm=llm, system_prompt="Your name is Cathy and you are a part of a duo of comedians."
        ),
        max_calls=3,
        init_message="Hey Cathy, tell me a joke",
    )
    if studio:
        Studio(comedy_duo, TeamTape(context=None, steps=[]), PrettyRenderer()).launch()
    else:
        for event in comedy_duo.run(TeamTape(context=None, steps=[])):
            print(event.model_dump_json(indent=2))


if __name__ == "__main__":
    llm = TrainableLLM(
        base_url="https://api.together.xyz",
        model_name="meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
        parameters=dict(temperature=0.7, max_tokens=512),
    )
    if len(sys.argv) == 2:
        if sys.argv[1] == "studio":
            try_chat(llm, studio=True)
        else:
            raise ValueError()
    elif len(sys.argv) == 1:
        try_chat(llm, studio=False)
    else:
        raise ValueError()
