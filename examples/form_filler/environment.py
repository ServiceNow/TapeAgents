import logging
from functools import lru_cache
from pathlib import Path

import yaml

from tapeagents.core import Error
from tapeagents.dialog_tape import AssistantStep
from tapeagents.environment import Environment, NoActionsToReactTo

from .schema import FunctionSchema
from .steps import CallFunction, Exit, FunctionCandidate, FunctionCandidates, InspectFunction, ResolveFunction
from .tape import FormFillerTape
from .types import FunctionName

logger = logging.getLogger(__name__)


@lru_cache
def load_schemas(path: Path) -> list[FunctionSchema]:
    logger.debug(f"Cache miss: loading schemas from {path}")
    result = []
    if path.is_dir():
        for file_path in path.resolve().glob("**/*.yaml"):
            result.extend(load_schemas(file_path))
    else:
        with open(path) as f:
            for obj in yaml.safe_load_all(f):
                result.append(FunctionSchema.model_validate(obj))
    return result


class FormFillerEnvironment(Environment[FormFillerTape]):
    def __init__(self, available_schemas: dict[FunctionName, FunctionSchema]):
        self.available_schemas: dict[FunctionName, FunctionSchema] = available_schemas

    def react(self, tape: FormFillerTape) -> FormFillerTape:
        action = tape.last_action
        assert action is not None, f"Last tape step should be an action for environment to react. Got {action}."
        if isinstance(action, ResolveFunction):
            # return all function candidates
            candidates: list[FunctionCandidate] = [
                FunctionCandidate(
                    function=f_name,
                    short_description=f_schema.description.split("\n")[0],
                )
                for f_name, f_schema in self.available_schemas.items()
            ]
            observation = FunctionCandidates(candidates=candidates)
            tape = tape.append(observation)
        elif isinstance(action, InspectFunction):
            # return the schema of the function
            observation = self.available_schemas.get(action.function)
            if observation is None:
                raise ValueError(f"Function schema not found for {action.function}")
            tape = tape.append(observation)
        elif isinstance(action, (AssistantStep, Exit)):
            self.raise_external_observation_needed(action)
        elif isinstance(action, CallFunction):
            # To be simulated with an LLM to obtain the results of a function call
            # tape = tape.append(FunctionResult(function=action.function, result={}))
            self.raise_external_observation_needed(action)
        elif isinstance(action, Error):
            # raise NoActionsToReactTo to stop orchestrator main_loop
            raise NoActionsToReactTo()
        else:
            self.raise_unexpected_action(action)
        return tape

    @staticmethod
    def from_spec(env_spec: str):
        return FormFillerEnvironment(available_schemas={f.name: f for f in load_schemas(Path(env_spec))})
