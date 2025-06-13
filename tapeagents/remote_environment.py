import asyncio
import atexit
import contextlib
import logging
import os
import random
import time
import traceback
import uuid
from multiprocessing import Pipe, Process, connection as mp_connection  # type: ignore
from typing import Annotated, Any, Union

import aiohttp
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field, TypeAdapter
from termcolor import colored

from tapeagents.core import Action, LLMOutputParsingFailureAction, Observation, TapeType, last_actions
from tapeagents.environment import AsyncEnvironment, Environment, UserStep
from tapeagents.tool_calling import ToolCallAction, ToolSpec
from tapeagents.utils import class_for_name, full_classname

logger = logging.getLogger(__name__)


class EnvironmentServer:
    """
    Manages multiple environment instances, each running in a separate process,
    and exposes them via an HTTP server using FastAPI. Clients can acquire an
    environment, operate on it using a session ID, and then release it.
    """

    def __init__(self, n_envs: int, host: str = "localhost", port: int = 8000, max_session_inactivity_secs: int = 600):
        if n_envs <= 0:
            raise ValueError("Number of instances must be positive.")
        self.n_envs = n_envs

        self.host = host
        self.port = port
        self.max_session_inactivity_secs = max_session_inactivity_secs

        self.env_pipes: dict[int, mp_connection.Connection] = {}
        self.env_processes: dict[int, Process] = {}
        self.sessions: dict[str, int] = {}  # session_id -> env_idx
        self.session_last_activity: dict[str, float] = {}  # session_id -> timestamp
        self.requests_in_progress: int = 0

    def start_envs(self, env_config: DictConfig):
        for i in range(self.n_envs):
            parent_conn, child_conn = Pipe()
            env_config_dict = OmegaConf.to_container(env_config, resolve=True)
            process = Process(
                target=EnvironmentServer._environment_worker,
                args=(env_config_dict, child_conn),
                daemon=True,  # Daemonize workers so they exit if main process crashes
            )
            process.start()

            self.env_pipes[i] = parent_conn
            self.env_processes[i] = process

    @staticmethod
    def _handle_step(environment: Environment, data: dict) -> dict:
        logger.info(f"Handling step: {data}")
        if data.get("kind") == "tool_call":
            action = ToolCallAction.model_validate(data)
        else:
            actions = [a for a in environment.actions() if not isinstance(a, ToolSpec)]
            type_adapter = TypeAdapter(Annotated[Union[tuple(actions)], Field(discriminator="kind")])
            action: Action = type_adapter.validate_python(data)

        observation = environment.step(action)
        return {"observation": observation.model_dump(), "classname": full_classname(type(observation))}

    @staticmethod
    def _handle_actions(environment: Environment, data: dict) -> dict:
        actions = environment.actions()
        return {
            "actions": [
                f"ToolSpec:{action.model_dump_json()}" if isinstance(action, ToolSpec) else full_classname(action)
                for action in actions
            ]
        }

    @staticmethod
    def _handle_reset(environment: Environment, data: dict) -> dict:
        environment.reset()
        return {"status": "ok"}

    @staticmethod
    def _handle_start_task(environment: Environment, data: dict) -> dict:
        start_result = environment.start_task(data)
        return start_result

    @staticmethod
    def _handle_shutdown(environment: Environment, data: dict) -> dict:
        environment.close()
        exit(0)

    @staticmethod
    def _environment_worker(env_config: dict, conn: mp_connection.Connection):
        logging.basicConfig(
            format="[%(asctime)s][%(name)s][%(levelname)s][%(process)d] - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[logging.StreamHandler()],
        )
        handlers = {
            "step": EnvironmentServer._handle_step,
            "actions": EnvironmentServer._handle_actions,
            "reset": EnvironmentServer._handle_reset,
            "start_task": EnvironmentServer._handle_start_task,
            "shutdown": EnvironmentServer._handle_shutdown,
        }
        environment: Environment = instantiate(OmegaConf.create(env_config))
        logger.info(f"Worker started for env: {environment.__class__.__name__} (PID: {os.getpid()})")
        try:
            logger.info(f"Worker {os.getpid()} initializing environment...")
            environment.initialize()
            logger.info(f"Worker {os.getpid()} environment initialized.")

            while True:
                command, data = conn.recv()
                logger.info(f"Worker {os.getpid()} received command: {command}")
                assert command in handlers, f"Worker {os.getpid()} unknown command: {command}"
                try:
                    result = handlers[command](environment, data)
                    conn.send(result)
                except Exception as e:
                    logger.exception(f"Worker {os.getpid()} error during {command}: {e}")
                    conn.send({"error": str(e), "status": "error", "details": traceback.format_exc()})
        except EOFError:  # Main process closed the pipe
            logger.info(f"Worker {os.getpid()} connection closed for {environment.__class__.__name__}")
        except KeyboardInterrupt:  # Graceful shutdown from worker side if possible
            logger.info(f"Worker {os.getpid()} received KeyboardInterrupt, shutting down.")
            environment.close()
        except Exception as e:
            logger.exception(f"Unhandled exception in worker {os.getpid()} for {environment.__class__.__name__}: {e}")
            # Try to inform parent if pipe is still open
            if not conn.closed:
                try:
                    conn.send(
                        {
                            "error": f"Unhandled worker exception: {str(e)}",
                            "status": "critical_error",
                            "details": traceback.format_exc(),
                        }
                    )
                except Exception:
                    pass  # Pipe might be broken
        finally:
            conn.close()
            logger.info(f"Worker {os.getpid()} for {environment.__class__.__name__} finished.")

    def _cleanup_inactive_sessions(self):
        """Remove sessions that have been inactive for longer than max_session_inactivity_secs."""
        current_time = time.time()
        inactive_sessions = []

        for session_id, last_activity in self.session_last_activity.items():
            if current_time - last_activity > self.max_session_inactivity_secs:
                inactive_sessions.append(session_id)

        for session_id in inactive_sessions:
            env_idx = self.sessions.get(session_id)
            if env_idx is not None:
                logger.info(f"Cleaning up inactive session {session_id} (env {env_idx})")
                # Reset the environment
                try:
                    parent_conn = self.env_pipes[env_idx]
                    parent_conn.send(("reset", None))
                    response = parent_conn.recv()
                    if isinstance(response, dict) and response.get("status") == "error":
                        logger.warning(f"Error resetting env {env_idx} during cleanup: {response.get('error')}")
                except Exception as e:
                    logger.warning(f"Failed to reset env {env_idx} during cleanup: {e}")

                # Remove from tracking
                del self.sessions[session_id]
                del self.session_last_activity[session_id]

    def _get_env_details(self, session_id: str) -> tuple[mp_connection.Connection, int]:
        if session_id not in self.sessions:
            raise HTTPException(status_code=400, detail=f"Invalid or expired session ID: {session_id}")

        # Update activity timestamp
        self.session_last_activity[session_id] = time.time()

        env_idx = self.sessions[session_id]
        assert env_idx in self.env_pipes, f"Environment index {env_idx} not found in pipes."
        assert env_idx in self.env_processes, f"Environment index {env_idx} not found in processes."

        if not self.env_processes[env_idx].is_alive():
            logger.error(f"Process for env_idx {env_idx} (session {session_id}) is not alive, killing session.")
            del self.sessions[session_id]
            raise HTTPException(status_code=503, detail="Environment process is not responding, session terminated.")

        parent_conn = self.env_pipes[env_idx]
        return parent_conn, env_idx

    def create_app(self):
        app = FastAPI(title="Environment Server", version="1.0.0")

        class ApiRequest(BaseModel):
            session_id: str

        class ActionRequest(ApiRequest):
            action_data: dict

        class TaskRequest(ApiRequest):
            task_data: dict = {}

        def _handle_worker_response(response: Any, operation_name: str):
            if isinstance(response, dict) and response.get("status") == "error":
                logger.error(
                    f"Worker error during {operation_name}: {response.get('error')}. Details: {response.get('details')}"
                )
                raise HTTPException(
                    status_code=500, detail=f"Worker error during {operation_name}: {response.get('error')}"
                )
            elif isinstance(response, dict) and response.get("status") == "critical_error":
                logger.error(
                    f"Critical worker error during {operation_name}: {response.get('error')}. Details: {response.get('details')}"
                )
                raise HTTPException(
                    status_code=503,
                    detail=f"Critical worker error during {operation_name}: {response.get('error')}. Details: {response.get('details')}",
                )
            return response

        async def _send_recv_async(conn, command, data):
            loop = asyncio.get_running_loop()

            def send_recv():
                conn.send((command, data))
                return conn.recv()

            try:
                result = await loop.run_in_executor(None, send_recv)
            except Exception as e:
                logger.error(f"Error during async send/recv for command {command}: {e}")
                result = {"status": "critical_error", "error": str(e), "details": traceback.format_exc()}
            return result

        @app.post("/acquire")
        async def acquire_environment():
            # Clean up inactive sessions before acquiring
            self._cleanup_inactive_sessions()

            for i in range(self.n_envs):
                if i not in self.sessions.values():
                    if not self.env_processes[i].is_alive():
                        logger.warning(f"Attempted to acquire env {i}, but its process is dead. Skipping.")
                        continue

                    session_id = str(uuid.uuid4())
                    self.sessions[session_id] = i
                    self.session_last_activity[session_id] = time.time()
                    logger.info(f"Environment {i} acquired with session ID: {session_id}")
                    logger.info(f"Remaining free environments: {self.n_envs - len(self.sessions)}")
                    return {"session_id": session_id}
            logger.warning("No free environments available for acquisition.")
            return {"error": "No free environments available. Please try again later."}

        @app.post("/release")
        async def release_environment(request: ApiRequest):
            session_id = request.session_id
            parent_conn, env_idx = self._get_env_details(session_id)

            try:
                response = await _send_recv_async(parent_conn, "reset", None)
                _handle_worker_response(response, f"release (reset for env {env_idx})")
            except (EOFError, BrokenPipeError) as pipe_err:
                logger.error(
                    f"Pipe error during release for session {session_id} (env {env_idx}): {pipe_err}. Env could be not reset properly."
                )
            finally:
                del self.sessions[session_id]
                if session_id in self.session_last_activity:
                    del self.session_last_activity[session_id]
                logger.info(f"Environment {env_idx} released from session {session_id} and reset.")
                logger.info(f"Remaining free environments: {self.n_envs - len(self.sessions)}")

            return {"status": "released", "env_idx": env_idx}

        @app.post("/step")
        async def step_endpoint(request: ActionRequest):
            parent_conn, env_idx = self._get_env_details(request.session_id)
            logger.info(
                f"Session {request.session_id} (Env {env_idx}): Run step {request.action_data['kind']}. Requests in progress: {self.requests_in_progress}"
            )
            self.requests_in_progress += 1
            response = await _send_recv_async(parent_conn, "step", request.action_data)
            self.requests_in_progress -= 1
            return _handle_worker_response(response, f"step for env {env_idx}")

        @app.post("/actions")
        async def actions_endpoint(request: ApiRequest):
            parent_conn, env_idx = self._get_env_details(request.session_id)
            logger.info(
                f"Session {request.session_id} (Env {env_idx}): Get actions. Requests in progress: {self.requests_in_progress}"
            )
            self.requests_in_progress += 1
            response = await _send_recv_async(parent_conn, "actions", None)
            self.requests_in_progress -= 1
            return _handle_worker_response(response, f"actions for env {env_idx}")

        @app.post("/reset")
        async def reset_endpoint(request: ApiRequest):
            logger.info(f"Resetting environment for session {request.session_id}")
            parent_conn, env_idx = self._get_env_details(request.session_id)
            logger.info(
                f"Session {request.session_id} (Env {env_idx}): Explicit reset. Requests in progress: {self.requests_in_progress}"
            )
            self.requests_in_progress += 1
            response = await _send_recv_async(parent_conn, "reset", None)
            self.requests_in_progress -= 1
            return _handle_worker_response(response, f"reset for env {env_idx}")

        @app.get("/health")
        async def health_check():
            """
            Health check endpoint to verify if the server is running.
            Returns a simple status message.
            """
            return {"status": "ok"}

        @app.post("/start_task")
        async def start_task_endpoint(request: TaskRequest):
            parent_conn, env_idx = self._get_env_details(request.session_id)
            logger.info(
                f"Session {request.session_id} (Env {env_idx}): Start task. Requests in progress: {self.requests_in_progress}"
            )
            self.requests_in_progress += 1
            response = await _send_recv_async(parent_conn, "start_task", request.task_data)
            self.requests_in_progress -= 1
            return _handle_worker_response(response, f"start_task for env {env_idx}")

        return app

    def shutdown(self):
        logger.info("Server shutting down. Cleaning up environment processes...")
        for env_idx, parent_conn in self.env_pipes.items():
            if self.env_processes[env_idx].is_alive():
                logger.info(f"Sending shutdown to worker for env {env_idx}")
                try:
                    parent_conn.send(("shutdown", None))
                except (BrokenPipeError, EOFError):
                    logger.warning(f"Pipe to worker {env_idx} already closed.")
            parent_conn.close()

        for env_idx, process in self.env_processes.items():
            logger.info(f"Waiting for worker process {env_idx} (PID: {process.pid}) to join...")
            process.join(timeout=5)
            if process.is_alive():
                logger.warning(f"Worker process {env_idx} (PID: {process.pid}) did not terminate, killing.")
                process.terminate()
                process.join()
        logger.info("All environment processes cleaned up.")

    def launch(self, env_config: DictConfig):
        app = self.create_app()
        self.start_envs(env_config)
        atexit.register(self.shutdown)

        logger.info(f"Starting Environment Server at http://{self.host}:{self.port} with {self.n_envs} environments.")
        uvicorn.run(app, host=self.host, port=self.port, timeout_keep_alive=3600, log_level="info")


class RemoteEnvironment(Environment):
    """
    Environment that proxies actions to a remote environment server.
    """

    def __init__(self, server_url: str):
        self.server_url = server_url
        self.session_id = None

    def initialize(self) -> None:
        response = requests.post(f"{self.server_url}/acquire")
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        response_data = response.json()
        self.session_id = response_data.get("session_id")
        logger.debug(f"Acquired environment with session ID: {self.session_id}")

    def start_task(self, task_data: dict) -> dict:
        response = requests.post(
            f"{self.server_url}/start_task", json={"task_data": task_data, "session_id": self.session_id}
        )
        if response.status_code != 200:
            logger.error(f"Failed to start task in environment: {response.text}")
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return response.json()

    def actions(self) -> tuple[type[Action], ...]:
        response = requests.post(f"{self.server_url}/actions", json={"session_id": self.session_id})
        if response.status_code != 200:
            logger.error(f"Failed to fetch actions from environment: {response.text}")
            raise HTTPException(status_code=response.status_code, detail=response.text)
        action_names = response.json().get("actions", [])
        actions = []
        for action in action_names:
            if action.startswith("ToolSpec:"):
                action = action[len("ToolSpec:") :]
                action_obj = ToolSpec.model_validate_json(action)
            else:
                action_obj = class_for_name(action)
            actions.append(action_obj)
        return tuple(actions)

    def react(self, tape: TapeType) -> TapeType:
        for action in last_actions(tape):
            observation = self.step(action)
            tape = tape.append(observation)
        return tape

    def step(self, action: Action) -> BaseModel:
        response = requests.post(
            f"{self.server_url}/step", json={"action_data": action.model_dump(), "session_id": self.session_id}
        )
        if response.status_code != 200:
            logger.error(f"Failed to step in environment: {response.text}")
            raise HTTPException(status_code=response.status_code, detail=response.text)
        response_dict = response.json()
        obs_dict = response_dict["observation"]
        cls: type[BaseModel] = class_for_name(response_dict["classname"])
        observation = cls.model_validate(obs_dict)
        return observation

    def reset(self) -> None:
        response = requests.post(f"{self.server_url}/reset", json={"session_id": self.session_id})
        if response.status_code != 200:
            logger.error(f"Failed to reset environment: {response.text}")
            raise HTTPException(status_code=response.status_code, detail=response.text)

    def close(self) -> None:
        if self.session_id:
            response = requests.post(f"{self.server_url}/release", json={"session_id": self.session_id})
            if response.status_code != 200:
                logger.error(f"Failed to release environment: {response.text}")
                raise HTTPException(status_code=response.status_code, detail=response.text)
        else:
            logger.warning("No session ID to release.")

    @contextlib.contextmanager
    def context(self):
        """
        Context manager to automatically acquire and release the environment.
        """
        self.initialize()
        try:
            yield self
        finally:
            self.close()
            logger.info("Environment session closed.")


class AsyncRemoteEnvironment(AsyncEnvironment):
    """
    Asynchronous environment that proxies actions to a remote environment server using aiohttp.
    """

    def __init__(self, server_url: str, max_parallel_requests: int = 32):
        self.server_url = server_url
        self.session_id: str | None = None
        self.session: aiohttp.ClientSession | None = None
        self.semaphore = asyncio.Semaphore(max_parallel_requests)

    async def ainitialize(self, session: aiohttp.ClientSession) -> None:
        self.session = session
        response_data = await self.api_call("acquire", suppress_errors=True)
        if "error" in response_data:
            raise ResourceWarning(f"Failed to acquire environment, server response: {response_data}")
        self.session_id = response_data.get("session_id")
        logger.debug(f"Acquired environment with session ID: {self.session_id}")
        await super().ainitialize()  # In case parent class has async initialization logic

    async def start_task(self, task_data: dict) -> dict:
        if not self.session or not self.session_id:
            raise RuntimeError("Environment not initialized. Call ainitialize first.")
        return await self.api_call("start_task", {"task_data": task_data})

    async def a_actions(self) -> tuple[type[Action], ...]:
        if not self.session or not self.session_id:
            raise RuntimeError("Environment not initialized. Call ainitialize first.")
        response_data = await self.api_call("actions")
        action_names = response_data.get("actions", [])
        actions = []
        for action in action_names:
            if action.startswith("ToolSpec:"):
                action = action[len("ToolSpec:") :]
                action_obj = ToolSpec.model_validate_json(action)
            else:
                action_obj = class_for_name(action)
            actions.append(action_obj)
        return tuple(actions)

    async def a_tools_description(self) -> str:
        desc_list = [a.description() for a in await self.a_actions()]
        return "\n".join(f"- {desc}" for desc in desc_list)

    async def astep(self, action: Action) -> Observation:
        t = time.perf_counter()
        if isinstance(action, LLMOutputParsingFailureAction):
            return UserStep(content="Try again")
        if not self.session or not self.session_id:
            raise RuntimeError("Environment not initialized. Call ainitialize first.")
        response_dict = await self.api_call("step", {"action_data": action.model_dump()})
        obs_dict = response_dict["observation"]
        obs_type: type[Observation] = class_for_name(response_dict["classname"])
        observation: Observation = obs_type.model_validate(obs_dict)
        observation.metadata.other["action_execution_time"] = time.perf_counter() - t
        observation.metadata.other["action_kind"] = action.kind
        return observation

    async def api_call(self, endpoint: str, data: dict | None = None, suppress_errors: bool = False) -> dict:
        if data is None:
            data = {}
        if self.session_id:
            data["session_id"] = self.session_id
        assert self.session, "AIOHTTP session must be initialized before making API calls."
        async with self.semaphore:
            async with self.session.post(f"{self.server_url}/{endpoint}", json=data) as response:
                if response.status != 200:
                    text = await response.text()
                    if not suppress_errors:
                        logger.error(f"Failed to call {endpoint} with data {data}: {text}")
                    raise HTTPException(status_code=response.status, detail=text)
                response_dict = await response.json()
        return response_dict

    async def areset(self) -> None:
        if not self.session or not self.session_id:
            raise RuntimeError("Environment not initialized. Call ainitialize first.")
        await self.api_call("reset")

    async def aclose(self) -> None:
        if self.session_id:
            try:
                await self.api_call("release")
                logger.debug(f"Async environment with session id {self.session_id} closed.")
            except aiohttp.ClientError as e:
                logger.error(colored(f"Failed to release environment: {e}", "red"))
            self.session_id = None
            self.session = None
        elif not self.session:
            logger.warning("No TCP session available to release environment.")
        else:
            logger.warning("No session ID to release.")

    def react(self, tape: TapeType) -> TapeType:
        raise NotImplementedError("Use areact for asynchronous environments.")

    async def areact(self, tape: TapeType) -> TapeType:
        for action in last_actions(tape):
            observation = await self.astep(action)
            tape = tape.append(observation)
        return tape

    @contextlib.asynccontextmanager
    async def acontext(
        self,
        session: aiohttp.ClientSession,
        wait_for_env: bool = True,
        initialization_timeout_sec: int = 3600,
    ):
        """
        Asynchronous context manager to automatically acquire and release the environment.
        The aiohttp.ClientSession is passed in and managed externally.
        """
        await self.wait_initialize(session, wait_for_env, initialization_timeout_sec)
        try:
            yield self
        except Exception as e:
            logger.exception(f"Exception caught in async context manager of the remote env: {e}")
            raise e
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt received, shutting down async environment.")
            raise
        finally:
            logger.debug("Closing environment session.")
            await self.aclose()

    async def wait_initialize(self, session, wait_for_env, initialization_timeout_sec):
        t = time.perf_counter()
        while True:
            try:
                await self.ainitialize(session)
                return
            except ResourceWarning as e:
                if not wait_for_env:
                    raise e
                if time.perf_counter() - t > initialization_timeout_sec:
                    logger.error(f"Failed to initialize environment after {initialization_timeout_sec} seconds: {e}")
                    raise e
                await asyncio.sleep(random.uniform(10, 100))
            except Exception as e:
                logger.error(f"Failed to initialize environment: {e}")
                raise e
