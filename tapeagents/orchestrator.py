"""
Module contains the main loops of the agent-environment interaction and replay functions.
"""

import asyncio
import enum
import logging
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Generator, Generic

import aiohttp
from hydra.utils import instantiate
from omegaconf import DictConfig
from pydantic import BaseModel, Field
from termcolor import colored

from tapeagents.agent import Agent
from tapeagents.config import is_debug_mode
from tapeagents.core import AgentEvent, Observation, Step, StopStep, TapeType
from tapeagents.environment import (
    AsyncEnvironment,
    Environment,
    ExternalObservationNeeded,
    NoActionsToReactTo,
    ToolCollectionEnvironment,
)
from tapeagents.remote_environment import AsyncRemoteEnvironment
from tapeagents.utils import FatalError, diff_dicts

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MainLoopStatus(enum.Enum):
    OK = "ok"
    FINISHED = "finished"
    NO_ACTIONS = "no_actions"
    EXTERNAL_INPUT_NEEDED = "external_input_needed"
    UNKNOWN_ACTION = "unknown_action"


class MainLoopEvent(BaseModel, Generic[TapeType]):
    # TODO: validate that event must have only either the agent or the environment data
    agent_event: AgentEvent[TapeType] | None = Field(default=None, description="Propagates all the agent events")
    agent_tape: TapeType | None = Field(default=None, description="The tape after agent.run()")
    observation: Observation | None = Field(
        default=None, description="One of the observations produced by environment.react()"
    )
    env_tape: TapeType | None = Field(default=None, description="The tape after environment.react()")
    status: MainLoopStatus = MainLoopStatus.OK


class MainLoopStream(Generic[TapeType]):
    def __init__(self, generator: Generator[MainLoopEvent[TapeType], None, None]):
        self.generator = generator

    def __bool__(self):
        return self.generator is not None

    def __iter__(self):
        if self.generator is None:
            raise ValueError("can't iterate a null stream")
        return self

    def __next__(self) -> MainLoopEvent:
        if self.generator is None:
            raise StopIteration
        return next(self.generator)

    def agent_events(self) -> Generator[AgentEvent[TapeType], None, None]:
        for event in self:
            if event.agent_event:
                yield event.agent_event

    def get_final_tape(self) -> TapeType:
        """Return the last tape by either the agent or the environment."""
        last_final_tape = None
        for event in self:
            if event.agent_tape:
                last_final_tape = event.agent_tape
            if event.env_tape:
                last_final_tape = event.env_tape
        if last_final_tape is not None:
            return last_final_tape
        raise ValueError("No tape by either the agent or the environment")


def get_agent_and_env_from_config(cfg: DictConfig) -> tuple[Agent, ToolCollectionEnvironment]:
    environment: ToolCollectionEnvironment = instantiate(cfg.environment)
    environment.initialize()
    logger.info(f"Environment tools: {environment.tools_description()}")
    agent: Agent = instantiate(
        cfg.agent, known_actions=environment.actions(), tools_description=environment.tools_description()
    )
    return agent, environment


async def run_agent_with_remote_env(cfg: DictConfig, tape: TapeType, session: aiohttp.ClientSession) -> TapeType:
    """
    Run the agent with a remote environment, using the provided tape as the starting point.
    For each tape, the following happens:

    1. Environment Acquisition: `async with environment.acontext(session, wait_for_env=True) as env:`

        An AsyncRemoteEnvironment instance is created.
        It acquires a session from the server via HTTP POST to /acquire.
        The server assigns an environment process to this session.

    2. Task Execution Loop: `await async_execute_agent(agent, tape, env, session)`

        The agent decides on actions based on observations.
        Actions are sent to the remote environment via HTTP POST to /step.
        Environment processes the action and returns observations.
        This continues until the agent produces a StopStep.

    3. Environment Release: end of the `async with ...` context manager

        When finished, the environment is released via HTTP POST to /release.
        The server resets the environment and makes it available for other tasks.
        Async Communication Workflow.

    Here's the communication flow between components:

    Client (orcherstrator.py)                   Environment Server (remote_environment.py)
      |                                           |
      |----- POST /acquire ---------------->      | (1. Acquire environment)
      |<---- session_id -------------------|      |
      |                                           |
      |----- POST /actions ---------------->      | (2. Get available actions)
      |<---- [action types] ---------------|      |
      |                                           |
      |----- POST /step ------------------->      | (3. Execute action)
      |<---- observation ------------------|      |
      |                                           |
      |      ... (repeat steps 3) ...             |
      |                                           |
      |----- POST /release ---------------->      | (4. Release environment)
      |<---- ok ---------------------------|      |

    :param cfg: Configuration for the agent and environment
    :param tape: Initial tape to start the agent with
    :param session: aiohttp session to use for the remote environment
    :return: Final tape after running the agent
    """
    environment: AsyncRemoteEnvironment = instantiate(cfg.environment)  # type: ignore
    async with environment.acontext(session, wait_for_env=True) as env:
        actions = await environment.a_actions()
        tools_description = await environment.a_tools_description()
        logger.info(f"Available tools: {tools_description}")
        agent: Agent = instantiate(cfg.agent, known_actions=actions, tools_description=tools_description)
        tape = await async_execute_agent(agent, tape, env, session)
        return tape


def main_loop(
    agent: Agent[TapeType],
    start_tape: TapeType,
    environment: Environment,
    max_loops: int = -1,
) -> MainLoopStream[TapeType]:
    """
    Main loop of the agent-environment interaction. The agent is run on the tape, then the environment reacts to the
    agent's tape, then the agent is run on the environment's tape, and so on.
    The loop stops when the agent emits a final step or the environment emits a final step,
    or the maximum number of loops is reached.

    :param agent: Agent object
    :param start_tape: initial tape
    :param environment: Environment object
    :param max_loops: maximum number of loops, -1 for infinite

    :return: generator of MainLoopEvent objects
    """

    def _implementation():
        if is_debug_mode():
            logger.setLevel(logging.DEBUG)
        n_loops = 0
        tape = start_tape
        event = None
        while n_loops < max_loops or max_loops == -1:
            # --- RUN THE AGENT ---
            for event in agent.run(tape):
                if event.step:
                    logger.debug(
                        colored(
                            f"{n_loops}:AGENT {event.step.metadata}: {event.step.llm_view()}",
                            "green",
                        )
                    )
                yield MainLoopEvent(agent_event=event)
                if event.final_tape:
                    break
            assert event and event.final_tape
            agent_tape = event.final_tape
            yield MainLoopEvent(agent_tape=agent_tape)

            # --- RUN THE ENVIRONMENT ---
            if any([isinstance(step, StopStep) for step in agent_tape.steps]):
                logger.debug(f"Agent emitted final step {agent_tape.steps[-1]}")
                yield MainLoopEvent(status=MainLoopStatus.FINISHED)
                return
            try:
                tape = environment.react(agent_tape)
            except NoActionsToReactTo:
                yield MainLoopEvent(status=MainLoopStatus.NO_ACTIONS)
                return
            except ExternalObservationNeeded:
                yield MainLoopEvent(status=MainLoopStatus.EXTERNAL_INPUT_NEEDED)
                return
            for observation in tape[len(agent_tape) :]:
                logger.debug(colored(f"{n_loops}:ENV {observation.metadata}: {observation.llm_view()}", "yellow"))
                yield MainLoopEvent(observation=observation)
                if isinstance(observation, StopStep):
                    logger.debug(f"Environment emitted final step {observation}")
                    yield MainLoopEvent[TapeType](env_tape=tape)
                    yield MainLoopEvent(status=MainLoopStatus.FINISHED)
                    return
            yield MainLoopEvent[TapeType](env_tape=tape)

            # --- REPEAT ---
            n_loops += 1

    return MainLoopStream(_implementation())


async def async_main_loop(
    agent: Agent[TapeType],
    start_tape: TapeType,
    environment: AsyncEnvironment,
    session: aiohttp.ClientSession,
    max_loops: int = -1,
):
    if is_debug_mode():
        logger.setLevel(logging.DEBUG)
    n_loops = 0
    tape = start_tape
    event = None
    while n_loops < max_loops or max_loops == -1:
        # --- RUN THE AGENT ---
        async for event in agent.arun(tape, session):
            yield MainLoopEvent(agent_event=event)
            if event.step:
                logger.info(f"Tape {tape.metadata.id} turn {n_loops} agent step")
                logger.debug(
                    colored(
                        f"{n_loops}:AGENT {event.step.metadata}: {event.step.llm_view()}",
                        "green",
                    )
                )
            if event.final_tape:
                break
        assert event and event.final_tape
        agent_tape = event.final_tape
        yield MainLoopEvent(agent_tape=agent_tape)

        # --- RUN THE ENVIRONMENT ---
        if any([isinstance(step, StopStep) for step in agent_tape.steps]):
            logger.debug(f"Agent emitted final step {agent_tape.steps[-1]}")
            yield MainLoopEvent(status=MainLoopStatus.FINISHED)
            return
        try:
            tape = await environment.areact(agent_tape)
        except NoActionsToReactTo:
            yield MainLoopEvent(status=MainLoopStatus.NO_ACTIONS)
            return
        except ExternalObservationNeeded:
            yield MainLoopEvent(status=MainLoopStatus.EXTERNAL_INPUT_NEEDED)
            return
        for observation in tape[len(agent_tape) :]:
            logger.info(f"Tape {tape.metadata.id} turn {n_loops} env step")
            logger.debug(colored(f"{n_loops}:ENV {observation.metadata}: {observation.llm_view()}", "yellow"))
            yield MainLoopEvent(observation=observation)
            if isinstance(observation, StopStep):
                logger.debug(f"Environment emitted final step {observation}")
                yield MainLoopEvent[TapeType](env_tape=tape)
                yield MainLoopEvent(status=MainLoopStatus.FINISHED)
                return
        yield MainLoopEvent[TapeType](env_tape=tape)

        # --- REPEAT ---
        n_loops += 1


def execute_agent(
    agent: Agent[TapeType], start_tape: TapeType, environment: Environment, max_loops: int = 50
) -> TapeType:
    """
    Execute the agent on the tape, then the environment reacts to the agent's tape, then the agent is run on the
    environment's tape, and so on. The loop stops when the agent emits a final step or the environment emits a final
    step, or the maximum number of loops is reached.

    :param agent: Agent object
    :param start_tape: initial tape
    :param environment: Environment object
    :param max_loops: maximum number of loops, 50 by default

    :return: final tape after running the agent
    """
    final_tape = start_tape
    try:
        for event in main_loop(agent, start_tape, environment, max_loops=max_loops):
            if event.agent_event and event.agent_event.final_tape:
                final_tape = event.agent_event.final_tape
            elif event.env_tape:
                final_tape = event.env_tape
    except Exception as e:
        final_tape.metadata.error = f"Agent loop exception: {e}"
        logger.exception(colored(f"Agent loop exception: {e}, stopping", "red"))
    tape_id = final_tape.metadata.id
    final_tape.metadata = start_tape.metadata
    final_tape.metadata.id = tape_id
    final_tape.metadata.parent_id = start_tape.metadata.id
    return final_tape


async def async_execute_agent(
    agent: Agent[TapeType],
    start_tape: TapeType,
    environment: AsyncEnvironment,
    session: aiohttp.ClientSession,
    max_loops: int = 50,
) -> TapeType:
    final_tape = start_tape
    try:
        async for event in async_main_loop(agent, start_tape, environment, session, max_loops):
            if event.agent_event and event.agent_event.final_tape:
                final_tape = event.agent_event.final_tape
            elif event.env_tape:
                final_tape = event.env_tape
    except Exception as e:
        final_tape.metadata.error = f"Agent loop exception: {e}"
        logger.exception(colored(f"Agent loop exception: {e}, stopping", "red"))
    tape_id = final_tape.metadata.id
    final_tape.metadata = start_tape.metadata
    final_tape.metadata.id = tape_id
    final_tape.metadata.parent_id = start_tape.metadata.id
    return final_tape


class EnvironmentGroup:
    def __init__(self, cfg: DictConfig, n_envs: int = 1):
        self.cfg = cfg
        self.semaphore = asyncio.Semaphore(n_envs)
        self.envs: list[ToolCollectionEnvironment] = [instantiate(cfg) for _ in range(n_envs)]
        self.exit_stack = AsyncExitStack()
        logger.info(f"Created {len(self.envs)} environments")

    async def __aenter__(self):
        for env in self.envs:
            await env.ainitialize()
            logger.info(f"Environment tools: {env.tools_description()}")
            self.exit_stack.push_async_callback(env.aclose)
        logger.info(f"Initialized {len(self.envs)} environments")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.exit_stack.aclose()

    @asynccontextmanager
    async def get_env(self):
        logger.info("waiting for env")
        async with self.semaphore:
            env = self.envs.pop(0)
            logger.info(f"pop env, {len(self.envs)} environments left")
            try:
                yield env
            finally:
                await env.areset()
                self.envs.append(env)
                logger.info(f"push env back, {len(self.envs)} environments available")


async def async_execute_with_env(
    env_group: EnvironmentGroup,
    agent_cfg: DictConfig,
    start_tape: TapeType,
    session: aiohttp.ClientSession,
    max_loops: int = 50,
) -> TapeType:
    async with env_group.get_env() as env:
        agent = instantiate(agent_cfg, known_actions=env.actions(), tools_description=env.tools_description())
        final_tape = await async_execute_agent(agent, start_tape, env, session, max_loops=max_loops)

    return final_tape


def replay_tape(
    agent: Agent[TapeType],
    tape: TapeType,
    env: Environment[TapeType] | None = None,
    start_tape: TapeType | None = None,
    reuse_observations: bool = False,
    stop_on_mismatch: bool = True,
) -> bool:
    """
    Replay the tape with the agent and compare the steps with the old tape.
    Count mismatches and print diffs of them.

    :param agent: Agent object
    :param tape: Old tape object
    :param env: Environment object
    :param start_tape: initial tape, if None, the first observations of the tape are used
    :param reuse_observations: reuse observations from the tape instead of calling the environment
    :param stop_on_mismatch: stop the replay on the first mismatch

    :return: True if all steps match, False otherwise
    """
    if env is None and not reuse_observations:
        raise ValueError("Environment is required when not reusing observations")
    match: bool = True
    if start_tape is None:
        start_steps: list[Step] = []
        for step in tape.steps:
            if isinstance(step, Observation):
                start_steps.append(step)
            else:
                break
        start_tape = tape.model_copy(update=dict(steps=start_steps))

    event = None
    new_tape: TapeType = start_tape
    new_steps_count = len(start_tape)
    while new_steps_count < len(tape):
        for event in agent.run(new_tape):
            if event.step:
                step_dict = event.step.llm_dict()
                if new_steps_count >= len(tape.steps):
                    logger.error(f"Extra step {new_steps_count} from agent, kind: {step_dict.get('kind')}")
                    match = False
                    if stop_on_mismatch:
                        return False
                    break
                old_step_dict = tape.steps[new_steps_count].llm_dict()
                new_steps_count += 1
                kind = step_dict.get("kind")
                old_kind = old_step_dict.get("kind")
                if kind != old_kind:
                    logger.error(
                        f"Step {new_steps_count} kind mismatch: Old {old_kind}, New {kind}\nOld step: {old_step_dict}\nNew step: {step_dict}"
                    )
                    match = False
                    if stop_on_mismatch:
                        return False
                elif old_step_dict != step_dict:
                    logger.error(f"Step {new_steps_count} mismatch")
                    logger.error(diff_dicts(old_step_dict, step_dict))
                    match = False
                    if stop_on_mismatch:
                        return False
                else:
                    logger.debug(f"Step {new_steps_count} ok")
            if event.final_tape:
                break
        assert event and event.final_tape
        agent_tape = event.final_tape
        new_tape = agent_tape
        if isinstance(new_tape.steps[-1], StopStep):
            logger.debug("Agent emitted final step, stop")
            break

        if reuse_observations:
            observations: list[Step] = []
            for step in tape.steps[new_steps_count:]:
                if isinstance(step, Observation):
                    observations.append(step)
                else:
                    break
            if len(observations):
                logger.debug(f"Reusing {len(observations)} observations from tape")
            new_tape = agent_tape + observations
            new_steps_count += len(observations)
        else:
            assert env is not None
            new_tape = env.react(agent_tape)
            observations = new_tape.steps[len(agent_tape) :]
            for observation in observations:
                step_dict = observation.llm_dict()
                old_step_dict = tape.steps[new_steps_count].llm_dict()
                new_steps_count += 1
                if old_step_dict != step_dict:
                    logger.error(f"Observation {new_steps_count} mismatch")
                    logger.error(diff_dicts(old_step_dict, step_dict))
                    match = False
                    if stop_on_mismatch:
                        return False
                else:
                    logger.debug(f"Observation {new_steps_count} ok")

                if isinstance(observation, StopStep):
                    logger.debug(f"Environment emitted final step {observation}")
                    break
        if isinstance(new_tape.steps[-1], StopStep):
            logger.debug("Env emitted final step, stop")
            break
    if new_steps_count != len(tape.steps):
        logger.error(f"New tape has {new_steps_count} steps, old tape has {len(tape.steps)}")
        match = False
    return match


def replay_tapes(
    agent: Agent[TapeType],
    tapes: list[TapeType],
    env: Environment[TapeType] | None = None,
    start_tapes: list[TapeType] | None = None,
    reuse_observations: bool = False,
    stop_on_error: bool = False,
) -> int:
    """
    Validate the list of tapes with the agent and environment.
    Check that the agent produce exactly the same steps as the original ones.
    Returns the number of failed tapes.
    """
    ok = 0
    fails = 0
    for i, tape in enumerate(tapes):
        logger.debug(f"Tape {i}")
        try:
            matched = replay_tape(
                agent,
                tape,
                env,
                start_tape=start_tapes[i] if start_tapes else None,
                reuse_observations=reuse_observations,
            )
            if not matched:
                raise FatalError("Tape mismatch")
            ok += 1
        except FatalError as e:
            logger.error(colored(f"Fatal error: {e}, skip tape {i}/{len(tapes)} ({tape.metadata.id})", "red"))
            fails += 1
            if stop_on_error:
                raise e

        logger.debug(colored(f"Ok: {ok}, Fails: {fails}", "green"))
    return fails
