import logging

from tapeagents.environment import ToolCollectionEnvironment
from tapeagents.tools.browser import Browser
from tapeagents.tools.code_executor import CodeExecutor
from tapeagents.tools.container_executor import ContainerExecutor
from tapeagents.tools.media_reader import VideoReader
from tapeagents.tools.search import Search
from tapeagents.tools.simple_browser import SimpleBrowser

logger = logging.getLogger(__name__)


def get_env(
    exp_path: str,
    simple_browser: bool = False,
    code_sandbox: ContainerExecutor | None = None,
    **kwargs,
) -> ToolCollectionEnvironment:
    browser = SimpleBrowser(exp_path=exp_path, kwargs=kwargs) if simple_browser else Browser(exp_path=exp_path)
    return ToolCollectionEnvironment(
        tools=[
            Search(),
            CodeExecutor(_sandbox=code_sandbox),
            VideoReader(exp_path=exp_path),
            browser,
        ]
    )
