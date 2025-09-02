import asyncio
import threading
import time
from typing import Any, Dict

import aiohttp
import requests
from omegaconf import OmegaConf
from pydantic import BaseModel

from tapeagents.core import Action, Observation, TapeType
from tapeagents.environment import Environment
from tapeagents.remote_environment import (
    AsyncRemoteEnvironment,
    EnvironmentServer,
    RemoteEnvironment,
)


# Mock environment classes for testing
class MockAction(Action):
    kind: str = "mock_action"
    message: str = "test message"


class MockObservation(Observation):
    content: str


class MockEnvironment(Environment):
    """A concrete mock environment for testing."""

    def __init__(self):
        self.step_count = 0

    def initialize(self) -> None:
        """Initialize the mock environment."""
        pass

    def actions(self) -> tuple[type[Action], ...]:
        """Return available actions."""
        return (MockAction,)

    def step(self, action: Action) -> BaseModel:
        """Execute an action and return an observation."""
        self.step_count += 1
        return MockObservation(content=f"Step {self.step_count}: Received {action.kind}")

    def react(self, tape: TapeType) -> TapeType:
        """Process the tape and return updated tape."""
        # Simple implementation for testing
        return tape

    def close(self) -> None:
        """Close the environment."""
        pass


def create_mock_env_config() -> Dict[str, Any]:
    """Create a mock environment configuration for testing."""
    return {
        "_target_": "tests.test_remote_env.MockEnvironment",  # Use our concrete mock environment
        # Add any other environment-specific parameters here
    }


def test_server_startup():
    server = EnvironmentServer(n_envs=2, port=8001)
    env_config = OmegaConf.create(create_mock_env_config())

    # Start server in a thread
    def run_server():
        try:
            server.launch(env_config)
        except KeyboardInterrupt:
            pass

    try:
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

        # Wait for server to start
        time.sleep(2)

        # Test health endpoint
        response = requests.get("http://localhost:8001/health")
        assert response.status_code == 200

        # Test task listing
        response = requests.get("http://localhost:8001/workers")
        assert response.status_code == 200
    finally:
        server.shutdown()


def test_sync_client():
    server = EnvironmentServer(n_envs=2, port=8002)
    env_config = OmegaConf.create(create_mock_env_config())

    # Start server in a thread
    def run_server():
        try:
            server.launch(env_config)
        except KeyboardInterrupt:
            pass

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Wait for server to start
    time.sleep(2)

    try:
        # Test client
        client = RemoteEnvironment("http://localhost:8002")

        with client.context():
            client.start_task({"test": "data"})

            # The worker should now be active
            response = requests.get("http://localhost:8002/workers")
            workers = response.json()["workers"]
            assert len(workers) == 1

            # Reset should terminate the worker
            client.reset()

            # worker should be gone
            response = requests.get("http://localhost:8002/workers")
            workers = response.json()["workers"]
            assert len(workers) == 0
    finally:
        server.shutdown()


def test_async_client():
    asyncio.run(_test_async_client())


async def _test_async_client():
    server = EnvironmentServer(n_envs=2, port=8003)
    env_config = OmegaConf.create(create_mock_env_config())

    # Start server in a thread
    def run_server():
        try:
            server.launch(env_config)
        except KeyboardInterrupt:
            pass

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Wait for server to start
    await asyncio.sleep(2)

    try:
        async with aiohttp.ClientSession() as session:
            client = AsyncRemoteEnvironment("http://localhost:8003")

            async with client.acontext(session):
                await client.start_task({"async_test": "data"})

                # The worker should now be active
                async with session.get("http://localhost:8003/workers") as response:
                    workers_data = await response.json()
                    workers = workers_data["workers"]
                    assert len(workers) == 1

                # Reset should terminate the worker
                await client.areset()

                # Worker should be gone
                async with session.get("http://localhost:8003/workers") as response:
                    workers_data = await response.json()
                    workers = workers_data["workers"]
                    assert len(workers) == 0
    finally:
        server.shutdown()


def test_process_per_task():
    server = EnvironmentServer(n_envs=3, port=8004)
    env_config = OmegaConf.create(create_mock_env_config())

    # Start server in a thread
    def run_server():
        try:
            server.launch(env_config)
        except KeyboardInterrupt:
            pass

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Wait for server to start
    time.sleep(2)

    try:
        # Start multiple tasks
        worker_ids = []
        for i in range(3):
            response = requests.post("http://localhost:8004/start_task", json={"task_data": {"task_num": i}})
            assert response.status_code == 200
            worker_id = response.json()["worker_id"]
            worker_ids.append(worker_id)

        # Check that we have 3 active tasks
        response = requests.get("http://localhost:8004/workers")
        workers = response.json()["workers"]
        assert len(workers) == 3

        # Each task should have a different PID
        pids = {worker["pid"] for worker in workers}
        assert len(pids) == 3

        # Reset one task
        response = requests.post("http://localhost:8004/reset", json={"worker_id": worker_ids[0]})
        assert response.status_code == 200

        # Should have 2 tasks left
        response = requests.get("http://localhost:8004/workers")
        workers = response.json()["workers"]
        assert len(workers) == 2

        # Clean up remaining tasks
        for worker_id in worker_ids[1:]:
            requests.post("http://localhost:8004/reset", json={"worker_id": worker_id})

    finally:
        server.shutdown()


if __name__ == "__main__":
    test_server_startup()
    test_sync_client()
    test_async_client()
    test_process_per_task()
