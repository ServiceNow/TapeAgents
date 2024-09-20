import datetime
import logging
import sys

from tapeagents.autogen_prompts import AUTOGEN_ASSISTANT_SYSTEM_MESSAGE
from tapeagents.collective import CollectiveAgent, CollectiveTape
from tapeagents.container_executor import ContainerExecutor
from tapeagents.develop import Develop
from tapeagents.environment import CodeExecutionEnvironment
from tapeagents.llms import LLM, LiteLLM
from tapeagents.rendering import PrettyRenderer
from tapeagents.runtime import main_loop

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def try_chat(llm: LLM, develop: bool):
    # equilavent of https://microsoft.github.io/autogen/docs/tutorial/introduction
    org = CollectiveAgent.create_chat_initiator(
        name="UserProxy",
        llm=llm,
        system_prompt="",
        collective_manager=CollectiveAgent.create(
            name="Assistant",
            system_prompt=AUTOGEN_ASSISTANT_SYSTEM_MESSAGE,
            llm=llm,
        ),
        max_calls=3,
        init_message="compute 5 fibonacci numbers",
        execute_code=True,
    )
    start_tape = CollectiveTape(context=None, steps=[])
    now = f"{datetime.datetime.now():%Y%m%d%H%M%S}"
    env = CodeExecutionEnvironment(ContainerExecutor(work_dir=f"outputs/chat_code/{now}"))
    if develop:
        Develop(org, start_tape, PrettyRenderer(), env).launch()
    else:
        loop = main_loop(org, start_tape, env)
        for event in loop:
            if ae := event.agent_event:
                if ae.step:
                    print(ae.step.model_dump_json(indent=2))
            else:
                assert event.observation
                print(event.observation.model_dump_json(indent=2))


if __name__ == "__main__":
    llm = LiteLLM(model_name="gpt-4o")
    if len(sys.argv) == 2:
        if sys.argv[1] == "develop":
            try_chat(llm, develop=True)
        else:
            raise ValueError()
    elif len(sys.argv) == 1:
        try_chat(llm, develop=False)
    else:
        raise ValueError()
