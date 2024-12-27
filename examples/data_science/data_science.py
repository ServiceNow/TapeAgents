import datetime
import logging
import sys

from tapeagents.agent import Agent
from tapeagents.core import Action, FinalStep, Observation, Tape
from tapeagents.environment import CodeExecutionEnvironment, Environment
from tapeagents.llms import LLM, LiteLLM
from tapeagents.orchestrator import main_loop
from tapeagents.renderers.basic import BasicRenderer
from tapeagents.renderers.camera_ready_renderer import CameraReadyRenderer
from tapeagents.renderers.pretty import PrettyRenderer
from tapeagents.team import TeamAgent, TeamTape
from tapeagents.tools.container_executor import ContainerExecutor
from tapeagents.view import Call, Respond

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

DEFAULT_TEAM_AGENT_SYSTEM_MESSAGE = """You are a helpful AI assistant.
Solve tasks using your coding and language skills.
In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute.
    1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
    2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
"""

SELECT_SPEAKER_MESSAGE_BEFORE_TEMPLATE = """You are in a role play game. The following roles are available:
{subagents}. Read the following conversation. Then select the next role from {subagents} to play. Only return the role.
"""

SELECT_SPEAKER_MESSAGE_AFTER_TEMPLATE = (
    """Read the above conversation. Then select the next role from {subagents} to play. Only return the role."""
)


def make_world(llm: LLM | None = None, env: Environment | None = None) -> tuple[Agent, Tape, Environment]:
    llm = llm or LiteLLM(model_name="gpt-4o", parameters={"timeout": 15.0})
    coder = TeamAgent.create(
        name="SoftwareEngineer",
        system_prompt=(
            DEFAULT_TEAM_AGENT_SYSTEM_MESSAGE
            + " Always start by installing packages your code need in a `install.sh` file. ."
            + " Always print in the code the filename of the generated files."
        ),
        llm=llm,
    )
    code_executor = TeamAgent.create(
        name="CodeExecutor",
        llm=llm,
        execute_code=True,
    )
    analyst = TeamAgent.create(
        name="AssetReviewer",
        system_prompt=(
            """As an asset reviewer, your role is to provide one-time feedback to enhance the generated assets. Only communicate your feedback, not a solution.
            Here is a list of best practices:
            - Compare stocks based on percentage changes, using a baseline at 0%.
            - Show the baseline in the plot.
            - Annotate the latest data points in the plot.
            Once your feedback has been implemented and the code runs successfully, simply respond with "TERMINATE".
            """
        ),
        llm=llm,
    )
    team = TeamAgent.create_team_manager(
        name="Manager",
        subagents=[coder, code_executor, analyst],
        max_calls=15,
        templates={
            "select_before": SELECT_SPEAKER_MESSAGE_BEFORE_TEMPLATE,
            "select_after": SELECT_SPEAKER_MESSAGE_AFTER_TEMPLATE,
        },
        llm=llm,
    )
    org = TeamAgent.create_initiator(
        name="Initiator",
        init_message=("Make a plot comparing the stocks of Google and Meta since beginning of 2024."),
        teammate=team,
    )
    start_tape = TeamTape(context=None, steps=[])
    now = f"{datetime.datetime.now():%Y%m%d%H%M%S}"
    env = env or CodeExecutionEnvironment(ContainerExecutor(work_dir=f"outputs/data_science/{now}"))
    return org, start_tape, env


def make_renderers() -> dict[str, BasicRenderer]:
    return {
        "camera-ready": CameraReadyRenderer(),
        "camera-ready_userview": CameraReadyRenderer(filter_steps=(Call, Respond, Action, Observation)),
        "camera-ready_nocontent": CameraReadyRenderer(show_content=False),
        "camera-ready_nocontent-nollmcalls": CameraReadyRenderer(show_content=False, render_llm_calls=False),
        "full": PrettyRenderer(),
        "calls_and_responses": PrettyRenderer(filter_steps=(Call, Respond, FinalStep), render_llm_calls=False),
        "actions_and_observations": PrettyRenderer(filter_steps=(Action, Observation), render_llm_calls=False),
    }


def main(studio: bool):
    now = f"{datetime.datetime.now():%Y%m%d%H%M%S}"
    work_dir = f"outputs/data_science/{now}"
    env = CodeExecutionEnvironment(ContainerExecutor(work_dir=work_dir, timeout=60 * 5))

    agent, start_tape, env = make_world(env=env)
    if studio:
        from tapeagents.studio import Studio

        Studio(agent, start_tape, make_renderers(), env).launch(static_dir=work_dir)
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
        case _:
            print("Usage: python -m examples.data_science.data_science [studio]")
            sys.exit(1)
