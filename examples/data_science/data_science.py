import datetime
import logging
import sys

from tapeagents.agent import Agent
from tapeagents.autogen_prompts import AUTOGEN_ASSISTANT_SYSTEM_MESSAGE
from tapeagents.team import TeamAgent, TeamTape
from tapeagents.container_executor import ContainerExecutor
from tapeagents.core import Action, FinalStep, Observation, Tape
from tapeagents.environment import CodeExecutionEnvironment, Environment
from tapeagents.llms import LLM, LiteLLM
from tapeagents.rendering import BasicRenderer, PrettyRenderer
from tapeagents.runtime import main_loop
from tapeagents.utils import run_in_tmp_dir_to_make_test_data
from tapeagents.view import Call, Respond

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def make_world(llm: LLM | None = None, env: Environment | None = None) -> tuple[Agent, Tape, Environment]:
    llm = llm or LiteLLM(model_name="gpt-4o", parameters={"timeout": 15.0})
    coder = TeamAgent.create(
        name="SoftwareEngineer",
        system_prompt=(
            AUTOGEN_ASSISTANT_SYSTEM_MESSAGE + "Only say TERMINATE when your code was successfully executed."
        ),
        llm=llm,
    )
    code_executor = TeamAgent.create(
        name="CodeExecutor",
        llm=llm,
        execute_code=True,
    )
    team = TeamAgent.create_team_manager(
        name="GroupChatManager",
        subagents=[coder, code_executor],
        max_calls=15,
        llm=llm,
    )
    org = TeamAgent.create_chat_initiator(
        name="UserProxy",
        init_message=(
            "Make a plot comparing the stocks of ServiceNow and Salesforce"
            " since beginning of 2024. Save it to a PNG file."
        ),
        teammate=team,
    )
    start_tape = TeamTape(context=None, steps=[])
    now = f"{datetime.datetime.now():%Y%m%d%H%M%S}"
    env = env or CodeExecutionEnvironment(ContainerExecutor(work_dir=f"outputs/data_science/{now}"))
    return org, start_tape, env


def make_renderers() -> dict[str, BasicRenderer]:
    return {
        "full": PrettyRenderer(),
        "calls_and_responses": PrettyRenderer(filter_steps=(Call, Respond, FinalStep), render_llm_calls=False),
        "actions_and_observations": PrettyRenderer(filter_steps=(Action, Observation), render_llm_calls=False),
    }


def main(studio: bool):
    agent, start_tape, env = make_world()
    if studio:
        from tapeagents.studio import Studio

        Studio(agent, start_tape, make_renderers(), env).launch()
    else:
        final_tape = main_loop(agent, start_tape, env).get_final_tape()
        with open("final_tape.json", "w") as f:
            f.write(final_tape.model_dump_json(indent=2))


if __name__ == "__main__":
    match sys.argv[1:]:
        case []:
            main(studio=False)
        case ["studio"]:
            main(studio=True)
        case ["make_test_data"]:
            with run_in_tmp_dir_to_make_test_data("data_science"):
                main(studio=False)
        case _:
            print("Usage: python -m examples.data_science.data_science [studio]")
            sys.exit(1)
