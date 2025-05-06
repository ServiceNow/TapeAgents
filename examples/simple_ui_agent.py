import json
import logging
from datetime import datetime
from typing import Any, TypeAlias

import litellm
from litellm import completion
from litellm.caching.caching import Cache
from pydantic import BaseModel

from tapeagents.core import FinalStep, Tape
from tapeagents.dialog_tape import MessageStep, UserStep
from tapeagents.mcp import MCPEnvironment, mcp_result_to_content
from tapeagents.tool_calling import ToolResult
from tapeagents.tools.base import StopTool

# os.environ["LITELLM_LOG"] = "DEBUG"
# litellm.set_verbose = True
litellm.cache = Cache()
logging.basicConfig(level=logging.INFO, force=True, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


UIStep: TypeAlias = UserStep | MessageStep | ToolResult | FinalStep


def _log_steps(steps: list[UIStep], length_limit: int = 500) -> None:
    for step in steps:
        step_str = step.model_dump_json(indent=2, exclude={"metadata"})
        if len(step_str) > length_limit:
            step_str = step_str[:length_limit] + f"...(truncated {len(step_str) - length_limit} chars)"
        logger.info(f"\x1b[3{2 if isinstance(step, ToolResult) else 5};20m{step_str}\x1b[0m")


class UITape(Tape[None, UIStep]):
    def to_llm_messages(self) -> list[dict]:
        return [m for step in self.steps for m in self._step_to_messages(step)]

    def _step_to_messages(self, step: UIStep) -> list[dict[str, Any]]:
        if isinstance(step, ToolResult):
            return mcp_result_to_content(step)
        elif isinstance(step, MessageStep):
            return [step.message.model_dump()]
        elif isinstance(step, UserStep):
            return [{"role": "user", "content": step.content}]
        else:
            return [{"role": "assistant", "content": step.llm_view()}]

    def __add__(self, steps: UIStep | list):
        if isinstance(steps, UIStep):
            steps = [steps]
        _log_steps(steps)
        return super().__add__(steps)  # type: ignore


class UIExpConfig(BaseModel):
    agent_name: str
    mcp_config_path: str
    task: str
    llm: str
    system_prompt: str
    plan_prompt: str
    select_action_prompt: str
    act_prompt: str
    reflection_prompt: str
    steps_limit: int


class Task(BaseModel):
    text: str

    def get_starting_tape(self) -> UITape:
        return UITape(steps=[UserStep(content=self.text)])


def run_agent(config: UIExpConfig) -> None:
    """
    Run the agent with the given config.
    """
    task = Task(text=config.task)
    # action space is defined by mcp server plus StopTool to stop the agent
    env = MCPEnvironment(config_path=config.mcp_config_path, tools=[StopTool()])

    def make_plan(tape: UITape) -> MessageStep:
        prompt = config.plan_prompt.format(tools=env.tools_description())
        step = _llm(tape, prompt)
        return step

    def select_action(tape: UITape) -> MessageStep:
        prompt = config.select_action_prompt.format(tools=env.tools_description())
        step = _llm(tape, prompt)
        return step

    def act(tape: UITape) -> MessageStep:
        step = _llm(tape, config.act_prompt, with_tools=True)
        return step

    def reflect(tape: UITape):
        step = _llm(tape, config.reflection_prompt)
        return step

    def task_complete(tape: UITape):
        return any(isinstance(step, FinalStep) for step in tape)

    def steps_limit(tape: UITape):
        return len(tape) >= config.steps_limit

    def _llm(tape: UITape, prompt: str, with_tools: bool = False) -> MessageStep:
        today = datetime.now().strftime("%Y-%m-%d")
        messages = [{"role": "system", "content": config.system_prompt.format(today=today)}]
        messages += tape.to_llm_messages()
        messages.append({"role": "user", "content": prompt})
        logger.debug(f"LLM input: {json.dumps(messages, indent=2)}")
        response = completion(
            model=config.llm,
            caching=True,
            messages=messages,
            tools=env.tool_specs() if with_tools else None,
        )
        message = response.choices[0].message  # type: ignore
        assert message is not None
        logger.debug(f"LLM output: {message}")
        step = MessageStep(message=message)
        step.metadata.agent = config.agent_name
        step.metadata.llm = config.llm
        return step

    # Main loop

    env.reset()
    tape = task.get_starting_tape()

    tape += make_plan(tape)
    while not task_complete(tape) and not steps_limit(tape):
        tape += select_action(tape)
        action = act(tape)
        tape += action
        tool_results = env.run_tools_from_message(action)
        tape += tool_results
        tape += reflect(tape)

    env.reset()


if __name__ == "__main__":
    config_dict = {
        "agent_name": "web_agent",
        "mcp_config_path": "conf/mcp/web.json",
        "task": "Find cheapest flight from Montreal to Miami for this Saturday",
        "llm": "gpt-4.1-2025-04-14",
        "steps_limit": 50,
        "system_prompt": "You are a web agent. Today is {today}.",
        "plan_prompt": "Make a plan of how to accomplish the task. Available tools: {tools}. Use google search if needed.",
        "select_action_prompt": "Describe which action to use next to move forward with the task. Available tools: {tools}.",
        "act_prompt": "Call the selected tool",
        "reflection_prompt": "Reflect on last tool results and a screenshot and how it affects the task.",
    }
    config = UIExpConfig(**config_dict)
    run_agent(config)
