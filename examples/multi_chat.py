import datetime
import logging
import sys

from tapeagents.autogen_prompts import DEFAULT_TEAM_AGENT_SYSTEM_MESSAGE
from tapeagents.team import TeamAgent, TeamTape
from tapeagents.container_executor import ContainerExecutor
from tapeagents.core import FinalStep
from tapeagents.studio import Studio
from tapeagents.environment import CodeExecutionEnvironment
from tapeagents.llms import LiteLLM
from tapeagents.rendering import PrettyRenderer
from tapeagents.runtime import main_loop
from tapeagents.view import Call, Respond

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def try_chat(studio: bool):
    llm = LiteLLM(model_name="gpt-4o", parameters={"timeout": 15.0}, use_cache=True)
    product_manager = TeamAgent.create(
        name="ProductManager",
        system_prompt="Creative in software product ideas.",
        llm=llm,
    )
    coder = TeamAgent.create(
        name="SoftwareEngineer",
        system_prompt=DEFAULT_TEAM_AGENT_SYSTEM_MESSAGE,
        llm=llm,
    )
    code_executor = TeamAgent.create(
        name="CodeExecutor",
        llm=llm,
        execute_code=True,
    )
    team = TeamAgent.create_team_manager(
        name="GroupChatManager",
        subagents=[product_manager, coder, code_executor],
        max_calls=15,
        llm=llm,
    )
    org = TeamAgent.create_initiator(
        name="UserProxy",
        init_message="Find a latest paper about gpt-4 on arxiv and find its potential applications in software.",
        teammate=team,
    )
    start_tape = TeamTape(context=None, steps=[])
    now = f"{datetime.datetime.now():%Y%m%d%H%M%S}"
    env = CodeExecutionEnvironment(ContainerExecutor(work_dir=f"outputs/multi_chat_code/{now}"))
    if studio:
        renderers = {
            "messages": PrettyRenderer(filter_steps=(Call, Respond, FinalStep), render_llm_calls=False),
            "full": PrettyRenderer(),
        }
        Studio(org, start_tape, renderers, env).launch()
    else:
        final_tape = main_loop(org, start_tape, env).get_final_tape()
        with open("final_tape.json", "w") as f:
            f.write(final_tape.model_dump_json(indent=2))


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["studio"]:
            try_chat(studio=True)
        case []:
            try_chat(studio=False)
        case _:
            print("Usage: python multi_chat.py [studio]")
            sys.exit(1)
