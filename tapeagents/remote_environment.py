import asyncio
import atexit
import contextlib
import logging
import os
import pickle
import socket
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from multiprocessing import Process
from typing import Annotated, Union

import aiohttp
import requests
import tenacity
import uvicorn
from fastapi import FastAPI, HTTPException
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field, TypeAdapter

from tapeagents.core import Action, LLMOutputParsingFailureAction, Observation, TapeType, last_actions
from tapeagents.environment import AsyncEnvironment, Environment, UserStep
from tapeagents.tool_calling import ToolCallAction, ToolSpec
from tapeagents.utils import class_for_name, full_classname

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(asctime)s][%(levelname)s][Server][%(process)d][%(name)s:%(lineno)d] - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)


@dataclass
class TaskWorker:
    """Information about a running task process."""

    worker_id: str
    process: Process
    socket_path: str
    start_time: float
    last_activity: float


class ResourceExhaustedException(Exception):
    """Raised when all environment slots are occupied."""

    pass


class ProcessPoolManager:
    """Manages a pool of environment processes, spawning them per task."""

    def __init__(self, max_workers: int, socket_dir: str = "/tmp/tapeagents_envs"):
        self.max_workers = max_workers
        self.socket_dir = socket_dir
        self.active_workers: dict[str, TaskWorker | None] = {}
        self.stopped_workers: int = 0
        self.lock = threading.Lock()

        # Create socket directory
        os.makedirs(self.socket_dir, exist_ok=True)

    def cleanup_dead_workers(self) -> None:
        """Remove dead workers from tracking."""
        dead_workers = []
        for worker_id, task_proc in self.active_workers.items():
            if task_proc is None:
                continue  # this worker is starting right now
            if not task_proc.process.is_alive():
                logger.info(f"Cleaning up dead worker {worker_id}")
                dead_workers.append(worker_id)
                # Clean up socket file
                try:
                    os.unlink(task_proc.socket_path)
                except FileNotFoundError:
                    pass

        for worker_id in dead_workers:
            del self.active_workers[worker_id]
        self.stopped_workers += len(dead_workers)

    def can_spawn_new_process(self) -> bool:
        """Check if we can spawn a new process."""
        self.cleanup_dead_workers()
        return len(self.active_workers) < self.max_workers

    def spawn_worker(self, worker_id: str, env_config: dict) -> str:
        """Spawn a new process for a worker and return socket path."""
        with self.lock:
            if not self.can_spawn_new_process():
                raise ResourceExhaustedException(
                    f"Cannot spawn process: {len(self.active_workers)}/{self.max_workers} slots occupied"
                )
            self.active_workers[worker_id] = None  # Mark as starting
        socket_path = os.path.join(self.socket_dir, f"worker_{worker_id}.sock")
        logger.info("Starting process")
        process = Process(
            target=ProcessPoolManager._task_worker, args=(env_config, socket_path, worker_id), daemon=True
        )
        process.start()
        logger.info(f"Process {process.pid} started for worker {worker_id}, wait for socket..")

        if not process.is_alive():
            logger.error(f"Worker {worker_id} failed to start!")
            raise RuntimeError(f"Failed to start worker {worker_id}")

        task_proc = TaskWorker(
            worker_id=worker_id,
            process=process,
            socket_path=socket_path,
            start_time=time.time(),
            last_activity=time.time(),
        )

        self.active_workers[worker_id] = task_proc
        logger.info(f"Spawned process {process.pid} for worker {worker_id}, socket: {socket_path}")
        return socket_path

    def get_socket_path(self, worker_id: str) -> str:
        """Get socket path for a worker, updating activity timestamp."""
        if worker_id not in self.active_workers:
            raise HTTPException(status_code=400, detail=f"Worker {worker_id} not found")

        task_proc = self.active_workers[worker_id]
        if task_proc is None:
            raise HTTPException(status_code=503, detail=f"Worker {worker_id} is starting, please wait")

        if not task_proc.process.is_alive():
            logger.error(f"Process for worker {worker_id} is dead, removing from tracking")
            del self.active_workers[worker_id]
            self.stopped_workers += 1
            raise HTTPException(status_code=503, detail=f"Worker {worker_id} process is not responding")

        # Update activity
        task_proc.last_activity = time.time()
        return task_proc.socket_path

    def terminate(self, worker_id: str) -> None:
        """Terminate a specific worker process."""
        if worker_id in self.active_workers:
            worker_info = self.active_workers[worker_id]
            if worker_info is None:
                return  # Worker is still starting, nothing to terminate
            logger.info(f"Terminating worker {worker_id} process {worker_info.process.pid}")

            if worker_info.process.is_alive():
                worker_info.process.terminate()
                worker_info.process.join(timeout=5)
                if worker_info.process.is_alive():
                    logger.warning(f"Force killing worker {worker_id} process {worker_info.process.pid}")
                    worker_info.process.kill()
                    worker_info.process.join()

            # Clean up socket
            try:
                os.unlink(worker_info.socket_path)
            except FileNotFoundError:
                pass

            del self.active_workers[worker_id]
            self.stopped_workers += 1

    def shutdown_all(self) -> None:
        """Shutdown all active task processes."""
        logger.info(f"Shutting down {len(self.active_workers)} active task processes")
        for worker_id in list(self.active_workers.keys()):
            self.terminate(worker_id)

    @staticmethod
    def _task_worker(env_config: dict, socket_path: str, worker_id: str):
        """Worker process for a single task using Unix domain socket communication."""
        logging.basicConfig(
            format=f"[%(asctime)s][%(levelname)s][Worker-{worker_id}][%(process)d][%(name)s:%(lineno)d] - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[logging.StreamHandler(), logging.FileHandler(f"/tmp/tapeagents_worker_{worker_id}.log")],
        )
        logger.info(f"Worker {worker_id} process starting")

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

        def _handle_actions(environment: Environment, data: dict) -> dict:
            actions = environment.actions()
            return {
                "actions": [
                    f"ToolSpec:{action.model_dump_json()}" if isinstance(action, ToolSpec) else full_classname(action)
                    for action in actions
                ]
            }

        def _handle_start_task(environment: Environment, data: dict) -> dict:
            start_result = environment.start_task(data)
            return {"start_result": start_result}

        def _handle_reset(environment: Environment, data: dict) -> dict:
            environment.reset()
            logger.info(f"Worker {worker_id} environment reset")
            return {"status": "ok", "should_exit": True}  # Signal worker to exit after reset

        # Initialize environment
        environment: Environment = instantiate(OmegaConf.create(env_config))
        logger.info(f"Worker started for {worker_id}, process {os.getpid()}")

        # Create Unix domain socket
        server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        logger.info(f"Worker {worker_id} created socket object")

        try:
            # Remove socket file if it exists
            try:
                os.unlink(socket_path)
            except FileNotFoundError:
                pass
            server_sock.bind(socket_path)
            server_sock.listen(1)
            logger.info(f"Worker {worker_id} now listening on socket {socket_path}")

            logger.info(f"Worker {worker_id} initializing environment...")
            environment.initialize()
            logger.info(f"Worker {worker_id} environment initialized successfully")

            while True:
                # Accept connection
                client_sock, addr = server_sock.accept()
                logger.debug(f"Worker {worker_id} accepted connection")

                try:
                    with client_sock:
                        # Receive data
                        data_length_bytes = client_sock.recv(4)
                        if len(data_length_bytes) != 4:
                            logger.warning(f"Worker {worker_id} received incomplete length header")
                            continue

                        data_length = int.from_bytes(data_length_bytes, byteorder="big")
                        data_bytes = b""
                        while len(data_bytes) < data_length:
                            chunk = client_sock.recv(data_length - len(data_bytes))
                            if not chunk:
                                break
                            data_bytes += chunk

                        if len(data_bytes) != data_length:
                            logger.warning(f"Worker {worker_id} received incomplete data")
                            continue

                        request = pickle.loads(data_bytes)  # Changed from json.loads
                        command = request.get("command")
                        data = request.get("data")

                        logger.info(f"Worker {worker_id} received command: {command}")

                        try:
                            should_exit = False
                            match command:
                                case "step":
                                    result = _handle_step(environment, data)
                                case "actions":
                                    result = _handle_actions(environment, data)
                                case "reset":
                                    result = _handle_reset(environment, data)
                                    should_exit = result.get("should_exit", False)
                                case "start_task":
                                    result = _handle_start_task(environment, data)
                                case "shutdown":
                                    environment.close()
                                    result = {"status": "ok"}
                                    should_exit = True
                                case _:
                                    raise ValueError(f"Unknown command: {command}")

                            # Send response
                            response_data = pickle.dumps(result)  # Changed from json.dumps
                            response_length = len(response_data)
                            client_sock.sendall(response_length.to_bytes(4, byteorder="big"))
                            client_sock.sendall(response_data)

                            if should_exit:
                                logger.info(f"Worker {worker_id} exiting after {command}")
                                break

                        except Exception as e:
                            logger.exception(f"Worker {worker_id} error during {command}: {e}")
                            error_result = {"error": str(e), "status": "error", "details": traceback.format_exc()}
                            response_data = pickle.dumps(error_result)  # Changed from json.dumps
                            response_length = len(response_data)
                            client_sock.sendall(response_length.to_bytes(4, byteorder="big"))
                            client_sock.sendall(response_data)

                except Exception as e:
                    logger.exception(f"Worker {worker_id} connection error: {e}")

        except Exception as e:
            logger.exception(f"Worker {worker_id} socket error: {e}")
        finally:
            try:
                environment.close()
            except Exception as e:
                logger.exception(f"Worker {worker_id} error closing environment: {e}")

            server_sock.close()
            try:
                os.unlink(socket_path)
            except FileNotFoundError:
                pass
            logger.info(f"Worker {worker_id} process {os.getpid()} exiting")


async def send_socket_request(socket_path: str, command: str, data: dict | None = None, timeout: int = 60) -> dict:
    """Send a request to a Unix domain socket and return the response using asyncio."""
    try:
        # Connect using asyncio - no need for wait_for wrapper
        reader, writer = await asyncio.open_unix_connection(socket_path)

        try:
            # Prepare request
            request = {"command": command, "data": data}
            request_data = pickle.dumps(request)
            request_length = len(request_data)

            # Send request
            writer.write(request_length.to_bytes(4, byteorder="big"))
            writer.write(request_data)
            await writer.drain()

            # Receive response length - readexactly handles connection issues
            response_length_bytes = await reader.readexactly(4)
            response_length = int.from_bytes(response_length_bytes, byteorder="big")

            # Receive response data
            response_data = await reader.readexactly(response_length)

            response = pickle.loads(response_data)
            return response

        finally:
            writer.close()
            await writer.wait_closed()

    except (OSError, ConnectionError, asyncio.IncompleteReadError) as e:
        raise ConnectionError(f"Failed to communicate with socket {socket_path}: {e}")
    except Exception as e:
        raise ConnectionError(f"Socket communication error: {e}")


class EnvironmentServer:
    """
    Manages multiple environment instances using process-per-task model,
    and exposes them via an HTTP server using FastAPI. Each task gets its own
    environment process that dies when the task completes.
    """

    def __init__(
        self,
        n_envs: int,
        host: str = "localhost",
        port: int = 8000,
        env_call_timeout: int = 60,
    ):
        if n_envs <= 0:
            raise ValueError("Number of instances must be positive.")

        self.host = host
        self.port = port
        self.env_call_timeout = env_call_timeout

        # Process pool manager handles task processes
        self.pool_manager = ProcessPoolManager(max_workers=n_envs)
        self.env_config: dict | None = None

    def create_app(self):
        app = FastAPI(title="Environment Server", version="2.0.0")

        class TaskRequest(BaseModel):
            task_data: dict = {}

        class ActionRequest(BaseModel):
            worker_id: str
            action_data: dict

        class WorkerRequest(BaseModel):
            worker_id: str

        async def call_task_worker(worker_id: str, command: str, data: dict | None = None) -> dict:
            """Send a command to a task process via Unix domain socket."""
            socket_path = self.pool_manager.get_socket_path(worker_id)

            try:
                response = await send_socket_request(socket_path, command, data, self.env_call_timeout)

                if response.get("status") == "error":
                    msg = f"Worker {worker_id} error: {response.get('error')}"
                    raise HTTPException(status_code=500, detail=msg)

                return response

            except ConnectionError as e:
                logger.error(f"Connection error to worker {worker_id} when running '{command}': {e}")
                # Clean up dead task
                self.pool_manager.terminate(worker_id)
                raise HTTPException(
                    status_code=503,
                    detail=f"Worker {worker_id} process connection error when running '{command}': {e}, worker terminated",
                )
            except Exception as e:
                logger.exception(f"Error calling worker {worker_id}: {e}")
                raise HTTPException(
                    status_code=503, detail=f"Worker {worker_id} communication error when running '{command}': {e}"
                )

        @app.post("/start_task")
        async def start_task_endpoint(request: TaskRequest):
            """Start a new task, spawning a dedicated environment process."""
            if self.env_config is None:
                raise HTTPException(status_code=500, detail="Environment not configured")

            worker_id = str(uuid.uuid4())

            try:
                # Spawn new process for this task
                socket_path = os.path.join(self.pool_manager.socket_dir, f"worker_{worker_id}.sock")
                logger.info(f"Spawning worker {worker_id} with socket {socket_path}")
                await asyncio.get_event_loop().run_in_executor(
                    None, self.pool_manager.spawn_worker, worker_id, self.env_config
                )
                logger.info(f"Created worker {worker_id} with socket {socket_path}")

                # Wait for socket to be ready
                socket_ready = False
                max_wait_time = 60  # seconds
                check_interval = 0.1  # seconds
                elapsed_time = 0

                t = time.perf_counter()
                while elapsed_time < max_wait_time and not socket_ready:
                    if os.path.exists(socket_path):
                        try:
                            test_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                            test_sock.settimeout(0.1)
                            test_sock.connect(socket_path)
                            test_sock.close()
                            socket_ready = True
                            logger.info(
                                f"Socket {socket_path} is ready for worker {worker_id} after {elapsed_time:.2f} seconds"
                            )
                            break
                        except (socket.error, ConnectionRefusedError, FileNotFoundError):
                            # Socket exists but not ready yet
                            pass
                    await asyncio.sleep(check_interval)
                    elapsed_time = time.perf_counter() - t

                if not socket_ready:
                    self.pool_manager.terminate(worker_id)
                    raise HTTPException(
                        status_code=500,
                        detail=f"Socket {socket_path} not ready within {max_wait_time} seconds, worker {worker_id} terminated",
                    )

                # Start the task
                response = await call_task_worker(worker_id, "start_task", request.task_data)

                return {"worker_id": worker_id, "start_result": response.get("start_result")}

            except ResourceExhaustedException as e:
                logger.warning(f"Resource exhaustion: {e}")
                raise HTTPException(status_code=503, detail=str(e))
            except Exception as e:
                logger.exception(f"Failed to start task: {e}")
                # Clean up on failure
                self.pool_manager.terminate(worker_id)
                raise HTTPException(status_code=500, detail=f"Failed to start task: {str(e)}")

        @app.post("/step")
        async def step_endpoint(request: ActionRequest):
            """Execute an action in the specified task environment."""
            return await call_task_worker(request.worker_id, "step", request.action_data)

        @app.post("/actions")
        async def actions_endpoint(request: WorkerRequest):
            """Get available actions for the specified task environment."""
            return await call_task_worker(request.worker_id, "actions")

        @app.post("/reset")
        async def reset_endpoint(request: WorkerRequest):
            """Reset the task environment and terminate its process."""
            try:
                # Send reset command to environment, this will cause worker to exit
                await call_task_worker(request.worker_id, "reset")

                # Wait a moment for graceful shutdown
                await asyncio.sleep(0.1)

                # Ensure process is terminated
                self.pool_manager.terminate(request.worker_id)

                return {"status": "ok", "message": f"Task {request.worker_id} reset and terminated"}

            except HTTPException as e:
                # If process is already dead, that's fine for reset
                if e.status_code == 503:
                    logger.info(f"Task {request.worker_id} was already dead during reset")
                    return {"status": "ok", "message": f"Task {request.worker_id} was already terminated"}
                raise
            except Exception as e:
                logger.exception(f"Error resetting task {request.worker_id}: {e}")
                # Force cleanup
                self.pool_manager.terminate(request.worker_id)
                raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

        @app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "ok",
                "active_workers": len(self.pool_manager.active_workers),
                "max_workers": self.pool_manager.max_workers,
                "stopped_workers": self.pool_manager.stopped_workers,
            }

        @app.get("/workers")
        async def list_workers():
            """List all active workers."""
            self.pool_manager.cleanup_dead_workers()
            workers = []
            for worker_id, task_proc in self.pool_manager.active_workers.items():
                workers.append(
                    {
                        "worker_id": worker_id,
                        "pid": task_proc.process.pid,
                        "start_time": task_proc.start_time,
                        "last_activity": task_proc.last_activity,
                        "age_seconds": time.time() - task_proc.start_time,
                    }
                )
            return {"workers": workers}

        @app.delete("/workers/{worker_id}")
        async def stop_worker(worker_id: str):
            """Manually terminate a specific worker."""
            try:
                self.pool_manager.terminate(worker_id)
                return {"status": "ok", "message": f"Worker {worker_id} terminated"}
            except Exception as e:
                logger.exception(f"Error terminating worker {worker_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to terminate worker: {str(e)}")

        return app

    def shutdown(self):
        """Shutdown all active worker processes."""
        logger.info("Server shutting down. Cleaning up all worker processes...")
        self.pool_manager.shutdown_all()
        logger.info("All worker processes cleaned up.")

    def launch(self, env_config: DictConfig):
        """Launch the environment server."""
        config_container = OmegaConf.to_container(env_config, resolve=True)
        if not isinstance(config_container, dict):
            raise ValueError("Environment config must be a dictionary")
        self.env_config = config_container
        app = self.create_app()
        atexit.register(self.shutdown)

        logger.info(
            f"Starting Environment Server at http://{self.host}:{self.port} with max {self.pool_manager.max_workers} processes."
        )
        uvicorn.run(app, host=self.host, port=self.port, timeout_keep_alive=3600, log_level="info")


class RemoteEnvironment(Environment):
    """
    Environment that proxies actions to a remote environment server using the new task-based API.
    """

    def __init__(self, server_url: str):
        self.server_url = server_url
        self.worker_id: str | None = None

    def initialize(self) -> None:
        """Initialize is a no-op since task creation happens in start_task."""
        pass

    def start_task(self, task_data: dict) -> dict:
        """Start a new task on the server."""
        response = requests.post(f"{self.server_url}/start_task", json={"task_data": task_data})
        if response.status_code != 200:
            logger.error(f"Failed to start task in environment: {response.text}")
            raise HTTPException(status_code=response.status_code, detail=response.text)

        result = response.json()
        self.worker_id = result.get("worker_id")
        logger.debug(f"Started task with worker ID: {self.worker_id}")
        return result.get("start_result", {})

    def actions(self) -> tuple[type[Action], ...]:
        if not self.worker_id:
            raise RuntimeError("No active task. Call start_task first.")

        response = requests.post(f"{self.server_url}/actions", json={"worker_id": self.worker_id})
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
        if not self.worker_id:
            raise RuntimeError("No active worker. Call start_task first.")

        response = requests.post(
            f"{self.server_url}/step", json={"worker_id": self.worker_id, "action_data": action.model_dump()}
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
        if not self.worker_id:
            logger.warning("No active task to reset.")
            return

        response = requests.post(f"{self.server_url}/reset", json={"worker_id": self.worker_id})
        if response.status_code != 200:
            logger.error(f"Failed to reset environment: {response.text}")
            raise HTTPException(status_code=response.status_code, detail=response.text)

        # Reset clears the worker
        self.worker_id = None

    def close(self) -> None:
        if self.worker_id:
            try:
                # Reset will terminate the task process
                self.reset()
            except Exception as e:
                logger.error(f"Error during close: {e}")
        else:
            logger.debug("No active task to close.")

    @contextlib.contextmanager
    def context(self):
        """
        Context manager to automatically start and clean up the task.
        Note: You still need to call start_task manually in the new API.
        """
        try:
            yield self
        finally:
            self.close()
            logger.info("Environment task closed.")


class AsyncRemoteEnvironment(AsyncEnvironment):
    """
    Asynchronous environment that proxies actions to a remote environment server using the new task-based API.
    """

    def __init__(self, server_url: str, max_parallel_requests: int = 32, start_timeout_sec: int = 3600):
        self.server_url = server_url
        self.worker_id: str | None = None
        self.session: aiohttp.ClientSession | None = None
        self.semaphore = asyncio.Semaphore(max_parallel_requests)
        self.start_timeout_sec = start_timeout_sec

    async def ainitialize(self, session: aiohttp.ClientSession) -> None:
        """Initialize with aiohttp session."""
        self.session = session
        await super().ainitialize()  # In case parent class has async initialization logic

    async def start_task(self, task_data: dict) -> dict:
        t = time.perf_counter()
        while True:
            try:
                result = await self._start_task(task_data)
                break
            except Exception as e:
                if time.perf_counter() - t > self.start_timeout_sec:
                    logger.error(f"Failed to start task after {self.start_timeout_sec} seconds: {e}")
                    raise HTTPException(status_code=500, detail=f"Failed to start task: {str(e)}")
                logger.warning(f"Failed to start task, retry after 5 seconds: {e}")
                await asyncio.sleep(5)
        start_time = time.perf_counter() - t
        logger.info(f"Task started after {start_time:.2f} seconds")
        return result

    async def _start_task(self, task_data: dict) -> dict:
        """Start a new task on the server."""
        if not self.session:
            raise RuntimeError("Environment not initialized. Call ainitialize first.")
        response_dict = await self.api_call("start_task", {"task_data": task_data}, suppress_errors=True)
        self.worker_id = response_dict.get("worker_id")
        logger.debug(f"Started async task with ID: {self.worker_id}")
        return response_dict.get("start_result", {})

    async def a_actions(self) -> tuple[type[Action], ...]:
        if not self.session or not self.worker_id:
            raise RuntimeError("Environment not initialized or no active task.")
        response_data = await self.api_call("actions", {"worker_id": self.worker_id})
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
        if not self.session or not self.worker_id:
            raise RuntimeError("Environment not initialized or no active task.")
        response_dict = await self.api_call("step", {"worker_id": self.worker_id, "action_data": action.model_dump()})
        obs_dict = response_dict["observation"]
        obs_type: type[Observation] = class_for_name(response_dict["classname"])
        observation: Observation = obs_type.model_validate(obs_dict)
        observation.metadata.other["action_execution_time"] = time.perf_counter() - t
        observation.metadata.other["action_kind"] = action.kind
        return observation

    @tenacity.retry(
        retry=tenacity.retry_if_exception_type(HTTPException),
        stop=tenacity.stop_after_delay(3600),  # Retry for up to 1 hour
        wait=tenacity.wait_random_exponential(multiplier=1, max=60),
        # wait randomly up to 2^x * 1 seconds between each retry until the range reaches 60 seconds
    )
    async def api_call(self, endpoint: str, data: dict | None = None, suppress_errors: bool = False) -> dict:
        if data is None:
            data = {}
        assert self.session, "AIOHTTP session must be initialized before making API calls."
        async with self.semaphore:
            async with self.session.post(f"{self.server_url}/{endpoint}", json=data) as response:
                if response.status != 200:
                    text = await response.text()
                    if not suppress_errors:
                        logger.error(f"Failed to call remote env /{endpoint}: {text}")
                    raise HTTPException(status_code=response.status, detail=text)
                response_dict = await response.json()
        return response_dict

    async def areset(self) -> None:
        if not self.session or not self.worker_id:
            logger.warning("Environment not initialized or no active task to reset.")
            return
        await self.api_call("reset", {"worker_id": self.worker_id})
        # Reset clears the task
        self.worker_id = None

    async def aclose(self) -> None:
        if self.worker_id:
            try:
                await self.areset()
                logger.debug(f"Async environment with task id {self.worker_id} closed.")
            except Exception as e:
                logger.error(f"Failed to reset task correctly: {e}")
            self.worker_id = None
        elif not self.session:
            logger.warning("No session available.")
        else:
            logger.debug("No active worker to close.")

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
        Asynchronous context manager to automatically manage the environment.
        The aiohttp.ClientSession is passed in and managed externally.
        Note: You still need to call start_task manually in the new API.
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
            logger.debug("Closing environment task.")
            await self.aclose()

    async def wait_initialize(self, session, wait_for_env, initialization_timeout_sec):
        """Wait for environment to be available."""
        await self.ainitialize(session)
