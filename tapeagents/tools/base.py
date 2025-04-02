import logging
import re

from pydantic import BaseModel

from tapeagents.config import force_cache
from tapeagents.core import Action, Observation
from tapeagents.steps import ActionExecutionFailure
from tapeagents.tools.tool_cache import add_to_cache, get_from_cache
from tapeagents.utils import FatalError

logger = logging.getLogger(__name__)


class BaseTool(BaseModel):
    def run(self, action: Action) -> Observation:
        raise NotImplementedError

    def execute_action(self, action: Action) -> Observation:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def close(self) -> None:
        """
        Perform any necessary cleanup actions.
        """
        pass

    def description(self) -> str:
        """
        Return a description of the tool.
        """
        doc = self.__doc__.replace("\n", " ").strip()
        doc = re.sub(r"\s+", " ", doc)
        return f"{self.__class__.__name__} - {doc}"


class Tool(BaseTool):
    """
    Tool is a base class for descriptions of a function that can be called.
    Defines the input action and the output observation.
    Implements caching of the results.

    Args:
        action: type[Action]: the type of the input action
        observation: type[Observation]: the type of the output observation
        cached: bool: whether to cache the results of the actions
    """

    action: type[Action]
    observation: type[Observation]
    cached: bool = False

    def run(self, action: Action) -> Observation:
        assert isinstance(action, self.action)
        tool_name = self.__class__.__name__
        if self.cached:
            obs_dict = get_from_cache(tool_name, args=(), kwargs=action.llm_dict())
            if obs_dict is not None:
                try:
                    return self.observation.model_validate(obs_dict)
                except Exception as e:
                    logger.error(f"Cache validation error: {e}, rerun tool")
            elif force_cache():
                raise FatalError(f"Cache is forced but no cache entry found for {tool_name}({action.llm_dict()})")
        try:
            observation = self.execute_action(action)
            if self.cached:
                add_to_cache(tool_name, args=(), kwargs=action.llm_dict(), result=observation.llm_dict())
        except FatalError:
            raise
        except Exception as e:
            logger.exception(f"Action failure: {e}")
            short_error = str(e)[:1000]
            observation = ActionExecutionFailure(error=short_error)
        assert isinstance(observation, (self.observation, ActionExecutionFailure))
        return observation


class StatefulTool(BaseTool):
    """
    Class that provides a set of functions performing
    explicitly defined set of actions operating on a shared stateful context.
    """

    actions: tuple[type[Action], ...]
    observations: tuple[type[Observation], ...]

    def run(self, action: Action) -> Observation:
        assert isinstance(action, self.actions)
        try:
            observation = self.execute_action(action)
        except FatalError:
            raise
        except Exception as e:
            logger.exception(f"Action failure: {e}")
            observation = ActionExecutionFailure(error=str(e))
        assert isinstance(observation, self.observations + (ActionExecutionFailure,))
        return observation
