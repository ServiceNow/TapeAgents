import logging

from tapeagents.environment import ToolCollectionEnvironment
from tapeagents.tools.browser import Browser
from tapeagents.tools.code_executor import CodeExecutor
from tapeagents.tools.document_converters import DocumentReader
from tapeagents.tools.media_reader import VideoReader
from tapeagents.tools.search import Search
from tapeagents.tools.simple_browser import SimpleBrowser

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
            DocumentReader(),
            Search(),
            CodeExecutor(exp_path=exp_path),
            VideoReader(exp_path=exp_path),
            SimpleBrowser(exp_path=exp_path, kwargs=kwargs) if simple_browser else Browser(exp_path=exp_path),
        ]
    )
