import datetime
import logging

from tapeagents.core import FinalStep
from tapeagents.tools.container_executor import ContainerExecutor
from tapeagents.view import Call, Respond

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

import sys

from tapeagents.autogen_prompts import AUTOGEN_ASSISTANT_SYSTEM_MESSAGE
from tapeagents.collective import CollectiveAgent, CollectiveEnvironment, CollectiveTape
from tapeagents.develop import Develop
from tapeagents.llms import LLM, LiteLLM
from tapeagents.rendering import BasicRenderer, PrettyRenderer
from tapeagents.runtime import main_loop


def try_chat(develop: bool):
    llm = LiteLLM(model_name="gpt-4o", parameters={"timeout": 15.0}, use_cache=True)
    product_manager = CollectiveAgent.create(
        name="ProductManager",
        system_prompt="Creative in software product ideas.",
        llm=llm,
    )
    coder = CollectiveAgent.create(
        name="SoftwareEngineer",
        system_prompt=AUTOGEN_ASSISTANT_SYSTEM_MESSAGE,
        llm=llm,
    )
    code_executor = CollectiveAgent.create(
        name="CodeExecutor",
        llm=llm,
        execute_code=True,
    )
    team = CollectiveAgent.create_collective_manager(
        name="GroupChatManager",
        subagents=[product_manager, coder, code_executor],
        max_calls=15,
        llm=llm,
    )
    org = CollectiveAgent.create_chat_initiator(
        name="UserProxy",
        init_message="Find a latest paper about gpt-4 on arxiv and find its potential applications in software.",
        collective_manager=team,
    )
    start_tape = CollectiveTape(context=None, steps=[])
    now = f"{datetime.datetime.now():%Y%m%d%H%M%S}"
    env = CollectiveEnvironment(ContainerExecutor(work_dir=f"outputs/multi_chat_code/{now}"))
    if develop:
        renderers = {
            "messages": PrettyRenderer(filter_steps=(Call, Respond, FinalStep), render_llm_calls=False),
            "full": PrettyRenderer(),
        }
        Develop(org, start_tape, renderers, env).launch()
    else:
        _ = list(main_loop(org, start_tape, env))


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["develop"]:
            try_chat(develop=True)
        case []:
            try_chat(develop=False)
        case _:
            print("Usage: python multi_chat.py [develop]")
            sys.exit(1)
