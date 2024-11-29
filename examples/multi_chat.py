import datetime
import logging
import sys

from tapeagents.core import FinalStep
from tapeagents.environment import CodeExecutionEnvironment
from tapeagents.llms import LiteLLM
from tapeagents.orchestrator import main_loop
from tapeagents.renderers.pretty import PrettyRenderer
from tapeagents.studio import Studio
from tapeagents.team import TeamAgent, TeamTape
from tapeagents.tools.container_executor import ContainerExecutor
from tapeagents.view import Call, Respond

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


# Prompts borrowed from https://github.com/microsoft/autogen, MIT License
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
        templates={
            "select_before": SELECT_SPEAKER_MESSAGE_BEFORE_TEMPLATE,
            "select_after": SELECT_SPEAKER_MESSAGE_AFTER_TEMPLATE,
        },
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
