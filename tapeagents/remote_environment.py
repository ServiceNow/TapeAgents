import asyncio
import atexit
import contextlib
import logging
import os
import time
import traceback
import uuid
from multiprocessing import Pipe, Process, connection as mp_connection  # type: ignore
from typing import Annotated, Union

import aiohttp
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from hydra.utils import instantiate
from omegaconf import DictConfig
from pydantic import BaseModel, Field, TypeAdapter
from termcolor import colored

from tapeagents.core import Action, LLMOutputParsingFailureAction, TapeType, last_actions
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

    def __init__(self, n_envs: int, host: str = "localhost", port: int = 8000):
        if n_envs <= 0:
            raise ValueError("Number of instances must be positive.")
        self.n_envs = n_envs

        self.host = host
        self.port = port

        self.env_pipes: dict[int, mp_connection.Connection] = {}
        self.env_processes: dict[int, Process] = {}
        self.sessions: dict[str, int] = {}  # session_id -> env_idx

    def start_envs(self, env_config: DictConfig):
        for i in range(self.n_envs):
            parent_conn, child_conn = Pipe()

            process = Process(
                target=EnvironmentServer._environment_worker,
                args=(env_config, child_conn),
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
    def _environment_worker(env_config: DictConfig, conn: mp_connection.Connection):
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
        environment: Environment = instantiate(env_config)
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

    def _get_env_details(self, session_id: str) -> tuple[mp_connection.Connection, int]:
        if session_id not in self.sessions:
            raise HTTPException(status_code=400, detail=f"Invalid or expired session ID: {session_id}")
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

        def _handle_worker_response(response: dict, operation_name: str):
            if response.get("status") == "error":
                logger.error(
                    f"Worker error during {operation_name}: {response.get('error')}. Details: {response.get('details')}"
                )
                raise HTTPException(
                    status_code=500, detail=f"Worker error during {operation_name}: {response.get('error')}"
                )
            if response.get("status") == "critical_error":
                logger.error(
                    f"Critical worker error during {operation_name}: {response.get('error')}. Details: {response.get('details')}"
                )
                # This session is likely dead. The _get_env_details might catch this on next call if process died.
                raise HTTPException(
                    status_code=503,
                    detail=f"Critical worker error: {response.get('error')}. Environment may be unstable.",
                )
            return response

        @app.post("/acquire")
        async def acquire_environment():
            for i in range(self.n_envs):
                if i not in self.sessions.values():
                    if not self.env_processes[i].is_alive():
                        logger.warning(f"Attempted to acquire env {i}, but its process is dead. Skipping.")
                        continue

                    session_id = str(uuid.uuid4())
                    self.sessions[session_id] = i
                    logger.info(f"Environment {i} acquired with session ID: {session_id}")
                    logger.info(f"Remaining free environments: {self.n_envs - len(self.sessions)}")
                    return {"session_id": session_id}
            raise HTTPException(status_code=503, detail="No environments available at the moment.")

        @app.post("/release")
        async def release_environment(request: ApiRequest):
            session_id = request.session_id
            parent_conn, env_idx = self._get_env_details(session_id)

            try:
                parent_conn.send(("reset", None))
                response = parent_conn.recv()
                _handle_worker_response(response, f"release (reset for env {env_idx})")
            except (EOFError, BrokenPipeError) as pipe_err:
                logger.error(
                    f"Pipe error during release for session {session_id} (env {env_idx}): {pipe_err}. Marking as error."
                )

                raise HTTPException(status_code=503, detail=f"Communication error with env {env_idx} during release.")
            finally:
                del self.sessions[session_id]
                logger.info(f"Environment {env_idx} released from session {session_id} and reset.")
                logger.info(f"Remaining free environments: {self.n_envs - len(self.sessions)}")

            return {"status": "released", "env_idx": env_idx}

        @app.post("/step")
        async def step_endpoint(request: ActionRequest):
            parent_conn, env_idx = self._get_env_details(request.session_id)
            logger.info(f"Session {request.session_id} (Env {env_idx}): Run step {request.action_data['kind']}")
            parent_conn.send(("step", request.action_data))
            response = parent_conn.recv()
            return _handle_worker_response(response, f"step for env {env_idx}")

        @app.post("/actions")
        async def actions_endpoint(request: ApiRequest):
            parent_conn, env_idx = self._get_env_details(request.session_id)
            logger.info(f"Session {request.session_id} (Env {env_idx}): Get actions")
            parent_conn.send(("actions", None))
            response = parent_conn.recv()
            return _handle_worker_response(response, f"actions for env {env_idx}")

        @app.post("/reset")
        async def reset_endpoint(request: ApiRequest):
            logger.info(f"Resetting environment for session {request.session_id}")
            parent_conn, env_idx = self._get_env_details(request.session_id)
            logger.info(f"Session {request.session_id} (Env {env_idx}): Explicit reset")
            parent_conn.send(("reset", None))
            response = parent_conn.recv()
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
            logger.info(f"Session {request.session_id} (Env {env_idx}): Start task")
            parent_conn.send(("start_task", request.task_data))
            response = parent_conn.recv()
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
        uvicorn.run(app, host=self.host, port=self.port)


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
        logger.info(f"Acquired environment with session ID: {self.session_id}")

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

    def __init__(self, server_url: str):
        self.server_url = server_url
        self.session_id: str | None = None
        self.tcp_session: aiohttp.ClientSession | None = None

    async def ainitialize(self, session: aiohttp.ClientSession) -> None:
        self.tcp_session = session
        async with self.tcp_session.post(f"{self.server_url}/acquire") as response:
            if response.status != 200:
                raise HTTPException(status_code=response.status, detail=await response.text())
            response_data = await response.json()
        self.session_id = response_data.get("session_id")
        logger.info(f"Acquired environment with session ID: {self.session_id}")
        await super().ainitialize()  # In case parent class has async initialization logic

    async def start_task(self, task_data: dict) -> dict:
        if not self.tcp_session or not self.session_id:
            raise RuntimeError("Environment not initialized. Call ainitialize first.")
        async with self.tcp_session.post(
            f"{self.server_url}/start_task", json={"task_data": task_data, "session_id": self.session_id}
        ) as response:
            if response.status != 200:
                text = await response.text()
                logger.error(f"Failed to start task in environment: {text}")
                raise HTTPException(status_code=response.status, detail=text)
            return await response.json()

    async def a_actions(self) -> tuple[type[Action], ...]:
        if not self.tcp_session or not self.session_id:
            raise RuntimeError("Environment not initialized. Call ainitialize first.")
        async with self.tcp_session.post(
            f"{self.server_url}/actions", json={"session_id": self.session_id}
        ) as response:
            if response.status != 200:
                text = await response.text()
                logger.error(f"Failed to fetch actions from environment: {text}")
                raise HTTPException(status_code=response.status, detail=text)
            response_data = await response.json()
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

    async def astep(self, action: Action) -> BaseModel:
        if isinstance(action, LLMOutputParsingFailureAction):
            return UserStep(content="Try again")
        if not self.tcp_session or not self.session_id:
            raise RuntimeError("Environment not initialized. Call ainitialize first.")
        async with self.tcp_session.post(
            f"{self.server_url}/step", json={"action_data": action.model_dump(), "session_id": self.session_id}
        ) as response:
            if response.status != 200:
                text = await response.text()
                logger.error(f"Failed to step in environment: {text}")
                raise HTTPException(status_code=response.status, detail=text)
            response_dict = await response.json()
        obs_dict = response_dict["observation"]
        cls: type[BaseModel] = class_for_name(response_dict["classname"])
        observation = cls.model_validate(obs_dict)
        return observation

    async def areset(self) -> None:
        if not self.tcp_session or not self.session_id:
            raise RuntimeError("Environment not initialized. Call ainitialize first.")
        async with self.tcp_session.post(f"{self.server_url}/reset", json={"session_id": self.session_id}) as response:
            if response.status != 200:
                text = await response.text()
                logger.error(f"Failed to reset environment: {text}")
                raise HTTPException(status_code=response.status, detail=text)

    async def aclose(self) -> None:
        if self.session_id and self.tcp_session:
            try:
                async with self.tcp_session.post(
                    f"{self.server_url}/release", json={"session_id": self.session_id}
                ) as response:
                    if response.status != 200:
                        text = await response.text()
                        logger.error(colored(f"Failed to release environment: {text}", "red"))
                logger.info(f"Async environment with session id {self.session_id} closed.")
            except aiohttp.ClientError as e:
                logger.error(colored(f"Failed to release environment: {e}", "red"))
            self.session_id = None
        elif not self.tcp_session:
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
            logger.exception(f"Error caught in asyc context manager of the env: {e}")
            raise e
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt received, shutting down async environment.")
            raise
        finally:
            await self.aclose()

    async def wait_initialize(self, session, wait_for_env, initialization_timeout_sec):
        t = time.perf_counter()
        while True:
            try:
                await self.ainitialize(session)
                return
            except HTTPException as e:
                if e.status_code == 503 and wait_for_env:
                    if time.perf_counter() - t > initialization_timeout_sec:
                        logger.error(
                            f"Failed to initialize environment after {initialization_timeout_sec} seconds: {e.detail}"
                        )
                        raise e
                    logger.info("Waiting for environment to become available...")
                    await asyncio.sleep(5)
                else:
                    logger.error(f"Failed to initialize environment: {e.status_code}, {e.detail}")
                    raise e
