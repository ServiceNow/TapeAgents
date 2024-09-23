import datetime
import logging
import sys

from tapeagents.agent import Agent
from tapeagents.autogen_prompts import AUTOGEN_ASSISTANT_SYSTEM_MESSAGE
from tapeagents.collective import CollectiveAgent, CollectiveTape
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
    coder = CollectiveAgent.create(
        name="SoftwareEngineer",
        system_prompt=(
            AUTOGEN_ASSISTANT_SYSTEM_MESSAGE + "Only say TERMINATE when your code was successfully executed."
        ),
        llm=llm,
    )
    code_executor = CollectiveAgent.create(
        name="CodeExecutor",
        llm=llm,
        execute_code=True,
    )
    team = CollectiveAgent.create_collective_manager(
        name="GroupChatManager",
        subagents=[coder, code_executor],
        max_calls=15,
        llm=llm,
    )
    org = CollectiveAgent.create_chat_initiator(
        name="UserProxy",
        init_message=(
            "Make a plot comparing the stocks of ServiceNow and Salesforce"
            " since beginning of 2024. Save it to a PNG file."
        ),
        collective_manager=team,
    )
    start_tape = CollectiveTape(context=None, steps=[])
    now = f"{datetime.datetime.now():%Y%m%d%H%M%S}"
    env = env or CodeExecutionEnvironment(ContainerExecutor(work_dir=f"outputs/data_science/{now}"))
    return org, start_tape, env


def make_renderers() -> dict[str, BasicRenderer]:
    return {
        "full": PrettyRenderer(),
        "calls_and_responses": PrettyRenderer(filter_steps=(Call, Respond, FinalStep), render_llm_calls=False),
        "actions_and_observations": PrettyRenderer(filter_steps=(Action, Observation), render_llm_calls=False),
    }


def main(develop: bool):
    agent, start_tape, env = make_world()
    if develop:
        from tapeagents.studio import Develop

        Develop(agent, start_tape, make_renderers(), env).launch()
    else:
        events = list(main_loop(agent, start_tape, env))
        assert (ae := events[-1].agent_event) and ae.final_tape
        with open("final_tape.json", "w") as f:
            f.write(ae.final_tape.model_dump_json(indent=2))


if __name__ == "__main__":
    match sys.argv[1:]:
        case []:
            main(develop=False)
        case ["develop"]:
            main(develop=True)
        case ["make_test_data"]:
            with run_in_tmp_dir_to_make_test_data("data_science"):
                main(develop=False)
        case _:
            print("Usage: python -m examples.data_science [develop]")
            sys.exit(1)
