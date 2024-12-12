import logging
import os

from tapeagents.core import Action, Tape
from tapeagents.environment import Environment
from tapeagents.tools.base import Multitool, Tool
from tapeagents.tools.code_executor import CodeExecutor
from tapeagents.tools.container_executor import ContainerExecutor
from tapeagents.tools.media_reader import VideoReader
from tapeagents.tools.search import Search
from tapeagents.tools.simple_browser import SimpleBrowser

logger = logging.getLogger(__name__)


class StepToolEnvironment(Environment):
    action_map: dict[type[Action], Tool | Multitool]

    def __init__(self, tools: list[Tool | Multitool]) -> None:
        super().__init__()
        self.tools = tools
        self.action_map = {tool.action: tool for tool in tools if isinstance(tool, Tool)}
        multitools = [tool for tool in tools if isinstance(tool, Multitool)]
        for multitool in multitools:
            self.action_map |= {action: multitool for action in multitool.actions}

    def react(self, tape: Tape) -> Tape:
        for action in self.last_actions(tape):
            action_type = type(action)
            if action_type not in self.action_map:
                raise Exception(f"Unknown action: {action_type}")
            tool = self.action_map[action_type]
            observation = tool.run(action)
            tape = tape.append(observation)
        return tape

    def close(self) -> None:
        for tool in self.tools:
            tool.close()

    def last_actions(self, tape: Tape) -> list[Action]:
        return [step for step in tape.steps[-tape.metadata.n_added_steps :] if isinstance(step, Action)]


def get_env(
    exp_path: str,
    code_sandbox: ContainerExecutor | None = None,
    **kwargs,
) -> StepToolEnvironment:
    return StepToolEnvironment(
        tools=[
            Search(),
            CodeExecutor(_sandbox=code_sandbox),
            VideoReader(attachment_dir=os.path.join(exp_path, "attachments")),
            SimpleBrowser(exp_path=exp_path, kwargs=kwargs),
        ]
    )
