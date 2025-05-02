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


class UITape(Tape[None, UIStep]):
    def to_llm_messages(self) -> list[dict]:
        return [message for step in self.steps for message in self._message(step)]

    def _message(self, step: UIStep) -> list[dict[str, Any]]:
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
        for step in steps:
            self.log_step(step)
        return super().__add__(steps)  # type: ignore

    def log_step(self, step: UIStep) -> None:
        step_str = step.model_dump_json(indent=2, exclude={"metadata"})
        if len(step_str) > 500:
            step_str = step_str[:500] + "..."
        logger.info(f"\x1b[3{2 if isinstance(step, ToolResult) else 5};20m{step_str}\x1b[0m")


class UIAgentConfig(BaseModel):
    agent_name: str
    mcp_config_path: str
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


class UIAgent(BaseModel):
    config: UIAgentConfig
    _env: MCPEnvironment = None  # type: ignore

    def model_post_init(self, __context) -> None:
        self._env = MCPEnvironment(config_path=self.config.mcp_config_path, tools=[StopTool()])

    def run(self, task: Task):
        self._env.reset()
        tape: UITape = task.get_starting_tape()

        tape += self.make_plan(tape)
        while not self.task_complete(tape) and not self.steps_limit(tape):
            tape += self.select_action(tape)
            action = self.act(tape)
            tape += action
            tool_results = self._env.run_tools_from_message(action.message)
            tape += tool_results
            tape += self.reflect(tape)

        self._env.reset()

    def make_plan(self, tape: UITape) -> MessageStep:
        prompt = self.config.plan_prompt.format(tools=self._env.tools_description())
        step = self._llm(tape, prompt)
        return step

    def select_action(self, tape: UITape) -> MessageStep:
        prompt = self.config.select_action_prompt.format(tools=self._env.tools_description())
        step = self._llm(tape, prompt)
        return step

    def act(self, tape: UITape) -> MessageStep:
        step = self._llm(tape, self.config.act_prompt, with_tools=True)
        return step

    def reflect(self, tape: UITape):
        step = self._llm(tape, self.config.reflection_prompt)
        return step

    def task_complete(self, tape: UITape):
        return any(isinstance(step, FinalStep) for step in tape)

    def steps_limit(self, tape: UITape):
        return len(tape) >= self.config.steps_limit

    def _llm(self, tape: UITape, prompt: str, with_tools: bool = False) -> MessageStep:
        messages = [{"role": "system", "content": self.config.system_prompt}]
        messages += tape.to_llm_messages()
        messages.append({"role": "user", "content": prompt})
        logger.debug(f"LLM input: {json.dumps(messages, indent=2)}")
        response = completion(
            model=self.config.llm,
            caching=True,
            messages=messages,
            tools=self._env.tool_specs() if with_tools else None,
        )
        message = response.choices[0].message  # type: ignore
        assert message is not None
        logger.debug(f"LLM output: {message}")
        step = MessageStep(message=message)
        step.metadata.agent = self.config.agent_name
        step.metadata.llm = self.config.llm
        return step


def main(config: UIAgentConfig, task_text: str):
    task = Task(text=task_text)
    agent = UIAgent(config=config)
    agent.run(task)


if __name__ == "__main__":
    config_dict = {
        "agent_name": "web_agent",
        "mcp_config_path": "conf/mcp/web.json",
        "llm": "gpt-4.1-2025-04-14",
        "system_prompt": f"You are a web agent. Today is {datetime.now().strftime('%Y-%m-%d')}.",
        "plan_prompt": "Make a plan of how to accomplish the task. Available tools: {tools}. Use google search if needed.",
        "select_action_prompt": "Describe which action to use next to move forward with the task. Available tools: {tools}.",
        "act_prompt": "Call two tools: the selected tool and the tool to make screenshot after that.",
        "reflection_prompt": "Reflect on last tool results and a screenshot and how it affects the task.",
        "steps_limit": 50,
    }
    config = UIAgentConfig(**config_dict)
    task_text = "Find cheapest flight from Montreal to Miami for this Saturday"
    main(config, task_text)
