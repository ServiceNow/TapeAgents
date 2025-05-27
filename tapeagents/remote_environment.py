import logging
from abc import ABC
from typing import Annotated, Union

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, TypeAdapter

from tapeagents.agent import TapeType
from tapeagents.core import Action, Tape
from tapeagents.environment import Environment
from tapeagents.utils import class_for_name

logger = logging.getLogger(__name__)


class EnvironmentServer(ABC):
    """
    Class that turns provided environment into a HTTP server using FastAPI.
    This allows remote execution of actions and retrieval of observations.
    The server will run on the specified port and can be accessed via HTTP requests.
    The environment should also implement the `reset` method to reset its state.
    The server will provide endpoints for executing actions and resetting the environment.
    """

    def __init__(
        self, environment: Environment, host: str = "localhost", port: int = 8000, action_types: tuple[str, ...] = ()
    ):
        self.environment = environment
        self.host = host
        self.port = port
        action_types = tuple(class_for_name(step) for step in action_types)
        self._action_type = Annotated[Union[action_types], Field(discriminator="kind")]

    def create_app(self):
        app = FastAPI(title="Environment Server", version="1.0.0")

        class ActionRequest(BaseModel):
            action_data: dict

        class TaskRequest(BaseModel):
            task_data: dict = {}

        @app.post("/step")
        async def step_endpoint(request: ActionRequest):
            try:
                # Convert dict back to Action object
                action = TypeAdapter(self._action_type).validate_python(request.action_data)
                logger.info(f"Run step {type(action).__name__} in environment {self.environment.__class__.__name__}")
                observation = self.environment.step(action)
                return {"observation": observation.model_dump()}
            except Exception as e:
                logger.exception(f"Failed to execute step: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/reset")
        async def reset_endpoint():
            try:
                self.environment.reset()
                return {"status": "ok"}
            except Exception as e:
                logger.exception(f"Failed to reset environment: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/health")
        async def health_check():
            """
            Health check endpoint to verify if the server is running.
            Returns a simple status message.
            """
            return {"status": "ok"}

        @app.post("/start_task")
        async def start_task_endpoint(request: TaskRequest):
            try:
                start_result: dict = self.environment.start_task(request.task_data)
                return start_result
            except Exception as e:
                logger.exception(f"Failed to start task: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        return app

    def run(self):
        app = self.create_app()
        logger.info(f"Starting Environment Server at http://{self.host}:{self.port}")
        uvicorn.run(app, host=self.host, port=self.port)


class RemoteEnvironment(Environment):
    """
    Environment that proxies actions to a remote environment server.
    """

    def __init__(self, server_url: str, observation_types: tuple[str, ...]):
        self.server_url = server_url
        obs_types = tuple(class_for_name(step) for step in observation_types)
        self._observation_type = Annotated[Union[obs_types], Field(discriminator="kind")]

    def start_task(self, task_data: dict) -> dict:
        response = requests.post(f"{self.server_url}/start_task", json={"task_data": task_data})
        response.raise_for_status()
        return response.json()

    def react(self, tape: TapeType) -> TapeType:
        for action in self.last_actions(tape):
            observation = self.step(action)
            tape = tape.append(observation)
        return tape

    def last_actions(self, tape: Tape) -> list[Action]:
        return [step for step in tape.steps[-tape.metadata.n_added_steps :] if isinstance(step, Action)]

    def step(self, action: Action) -> BaseModel:
        response = requests.post(f"{self.server_url}/step", json={"action_data": action.model_dump()})
        response.raise_for_status()
        obs_dict = response.json()["observation"]
        observation = TypeAdapter(self._observation_type).validate_python(obs_dict)
        return observation

    def reset(self) -> None:
        response = requests.post(f"{self.server_url}/reset")
        response.raise_for_status()
