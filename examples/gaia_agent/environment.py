import logging

from tapeagents.environment import StepToolEnvironment
from tapeagents.tools.code_executor import CodeExecutor
from tapeagents.tools.container_executor import ContainerExecutor
from tapeagents.tools.media_reader import VideoReader
from tapeagents.tools.search import Search
from tapeagents.tools.simple_browser import SimpleBrowser

logger = logging.getLogger(__name__)


def get_env(
    exp_path: str,
    code_sandbox: ContainerExecutor | None = None,
    **kwargs,
) -> StepToolEnvironment:
    return StepToolEnvironment(
        tools=[
            Search(),
            CodeExecutor(_sandbox=code_sandbox),
            VideoReader(exp_path=exp_path),
            SimpleBrowser(exp_path=exp_path, kwargs=kwargs),
        ]
    )
