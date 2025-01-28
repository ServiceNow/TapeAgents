import logging

from tapeagents.environment import ToolCollectionEnvironment
from tapeagents.tools.browser import Browser
from tapeagents.tools.code_executor import CodeExecutor
from tapeagents.tools.media_reader import VideoReader
from tapeagents.tools.simple_browser import SimpleBrowser
from tapeagents.tools.web_search import WebSearch

logger = logging.getLogger(__name__)


def get_env(
    exp_path: str,
    simple_browser: bool = False,
    **kwargs,
) -> ToolCollectionEnvironment:
    if simple_browser:
        logger.info("Using simple browser")
    return ToolCollectionEnvironment(
        tools=[
            WebSearch(),
            CodeExecutor(exp_path=exp_path),
            VideoReader(exp_path=exp_path),
            SimpleBrowser(exp_path=exp_path, kwargs=kwargs) if simple_browser else Browser(exp_path=exp_path, **kwargs),
        ]
    )
