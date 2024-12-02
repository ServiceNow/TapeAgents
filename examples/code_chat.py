import datetime
import logging
import sys

from tapeagents.environment import CodeExecutionEnvironment
from tapeagents.llms import LLM, LiteLLM
from tapeagents.orchestrator import main_loop
from tapeagents.renderers.pretty import PrettyRenderer
from tapeagents.studio import Studio
from tapeagents.team import TeamAgent, TeamTape
from tapeagents.tools.container_executor import ContainerExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Prompt borrowed from https://github.com/microsoft/autogen, MIT License
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


def try_chat(llm: LLM, studio: bool):
    # equilavent of https://microsoft.github.io/autogen/docs/tutorial/introduction
    org = TeamAgent.create_initiator(
        name="UserProxy",
        llm=llm,
        system_prompt="",
        teammate=TeamAgent.create(
            name="Assistant",
            system_prompt=DEFAULT_TEAM_AGENT_SYSTEM_MESSAGE,
            llm=llm,
        ),
        max_calls=3,
        init_message="compute 5 fibonacci numbers",
        execute_code=True,
    )
    start_tape = TeamTape(context=None, steps=[])
    now = f"{datetime.datetime.now():%Y%m%d%H%M%S}"
    env = CodeExecutionEnvironment(ContainerExecutor(work_dir=f"outputs/chat_code/{now}"))
    if studio:
        Studio(org, start_tape, PrettyRenderer(), env).launch()
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
        if sys.argv[1] == "studio":
            try_chat(llm, studio=True)
        else:
            raise ValueError()
    elif len(sys.argv) == 1:
        try_chat(llm, studio=False)
    else:
        raise ValueError()
