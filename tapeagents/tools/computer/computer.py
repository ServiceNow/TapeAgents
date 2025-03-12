import atexit
import base64
import logging
import os
import socket
import time

import podman
import requests
from PIL import Image

from tapeagents.core import Action
from tapeagents.tools.base import StatefulTool
from tapeagents.tools.computer.steps import (
    ComputerObservation,
    GetCursorPositionAction,
    KeyPressAction,
    MouseClickAction,
    MouseClickAtAction,
    MouseHoverAction,
    MouseMoveAction,
    OpenUrlAction,
    PageDownAction,
    PageUpAction,
    RunTerminalCommand,
    TypeTextAction,
)
from tapeagents.tools.grounding import GroundingModel

logger = logging.getLogger(__name__)


class Computer(StatefulTool):
    """
    Computer tool for executing actions in the container with linux desktop.
    Can execute linux terminal commands, open URLs in the browser.
    Can move mouse, click, scroll, type text, enter hotkeys.
    Can also take screenshots and get cursor position.
    """

    exp_path: str | None = None
    actions: tuple[type[Action], ...] = ()
    observations: tuple[type[ComputerObservation], ...] = (ComputerObservation,)
    computer_url: str = "http://localhost"
    api_port: int = 8000
    use_grounding: bool = True
    grounding_api_url: str
    container_image: str = "computer"
    container_name: str = "computer"

    def model_post_init(self, __context):
        self._grounding = GroundingModel(url=self.grounding_api_url)
        self._screenshot_dir = f"{self.exp_path}/attachments/remote_screenshots/"
        os.makedirs(self._screenshot_dir, exist_ok=True)
        self._action_map = {
            TypeTextAction: self.remote_execute_action,
            OpenUrlAction: self.remote_execute_action,
            KeyPressAction: self.remote_execute_action,
            PageUpAction: self.page_up,
            PageDownAction: self.page_down,
            RunTerminalCommand: self.remote_execute_action,
        }
        if self.use_grounding:
            self._action_map[MouseClickAtAction] = self.mouse_click
            self._action_map[MouseHoverAction] = self.mouse_hover
        else:
            self._action_map[MouseClickAction] = self.remote_execute_action
            self._action_map[MouseMoveAction] = self.remote_execute_action
            self._action_map[GetCursorPositionAction] = self.remote_execute_action
        self.actions = tuple(self._action_map.keys())
        if os.environ.get("REUSE_COMPUTER_CONTAINER"):
            os.environ["COMPUTER_CONTAINER_NAME"] = self.container_name
        else:
            self.api_port = launch_container(self.container_image, self.container_name, stop_at_exit=True)

    def execute_action(self, action: Action) -> ComputerObservation:
        action_type = type(action)
        if action_type in self._action_map:
            return self._action_map[action_type](action)
        raise ValueError(f"Unknown action type: {action_type}")

    def mouse_hover(self, action: MouseHoverAction) -> ComputerObservation:
        x, y = self._grounding.get_coords(self.get_screen(), action.element_description)
        return self.remote_execute_action(MouseMoveAction(x=int(x), y=int(y)))

    def mouse_click(self, action: MouseClickAtAction) -> ComputerObservation:
        x, y = self._grounding.get_coords(self.get_screen(), action.element_description)
        return self.remote_execute_action(MouseClickAction(x=int(x), y=int(y), button=action.button))

    def page_up(self, action: PageUpAction) -> ComputerObservation:
        return self.remote_execute_action(KeyPressAction(text="pageup"))

    def page_down(self, action: PageDownAction) -> ComputerObservation:
        return self.remote_execute_action(KeyPressAction(text="pagedown"))

    def remote_execute_action(self, action: Action) -> ComputerObservation:
        payload = {"kind": action.kind, "params": action.model_dump()}
        try:
            response = requests.post(f"{self.computer_url}:{self.api_port}/execute", json=payload)
            response.raise_for_status()
            obs_dict = response.json()
            logger.info(f"Received observation: {obs_dict.keys()}")
            return self.save_screenshot(ComputerObservation(**obs_dict))
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return ComputerObservation(image_path="", error=f"API request failed: {str(e)}")

    def save_screenshot(self, obs: ComputerObservation) -> ComputerObservation:
        if obs.base64_image:
            fname = f"{self._screenshot_dir}/screen_{int(time.time())}.png"
            with open(fname, "wb") as f:
                f.write(base64.b64decode(obs.base64_image))
            obs.image_path = fname
            obs.base64_image = None
        return obs

    def get_screen(self) -> Image:
        obs = self.remote_execute_action(GetCursorPositionAction())
        if obs.error:
            raise ValueError(f"Failed to get screen: {obs.error}")
        return Image.open(obs.image_path)

    def reset(self):
        self.remote_execute_action(RunTerminalCommand(command='xdotool search "" windowkill %@'))

    def close(self):
        return super().close()


def find_free_port(start_port: int) -> int:
    """Find the first available TCP port starting from start_port."""
    port = start_port
    max_port = start_port + 1000
    while port < max_port:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port
        except OSError:
            logger.warning(f"Port {port} is already in use, trying next")
            port += 1
    raise RuntimeError(f"No free ports in range {start_port}-{max_port}")


def launch_container(
    container_image="computer",
    container_name="computer",
    stop_at_exit=False,
    reuse_container=False,
) -> int:
    podman_client = podman.from_env()
    logger.info(f"Reuse container: {reuse_container}")
    if reuse_container and podman_client.containers.exists(container_name):
        logger.info(f"Container '{container_name}' already exists, reusing it.")
        container = podman_client.containers.get(container_name)
        container.start()
        api_port = 8000
    else:
        unique_container_name = container_name
        i = 0
        while podman_client.containers.exists(unique_container_name):
            i += 1
            if i > 100:
                raise RuntimeError(f"Too many containers with the same name '{container_name}'")
            unique_container_name = f"{container_name}_{i}"
        logger.info(f"Start container '{unique_container_name}' from image '{container_image}' ")
        os.environ["COMPUTER_CONTAINER_NAME"] = unique_container_name
        home_dir = os.path.join(os.getcwd(), ".computer")
        zip_home = os.path.join(os.path.dirname(__file__), "home.zip")
        if os.path.exists(home_dir):
            os.system(f"rm -rf {home_dir}")
            logger.info(f"Removed existing home directory: {home_dir}")
        else:
            os.makedirs(home_dir)
        os.system(f"unzip -qq -o {zip_home} -d {home_dir}")
        logger.info(f"Recreated home directory from zip: {zip_home}")
        if reuse_container:
            api_port = 8000
            vnc_port = 3000
        else:
            api_port = find_free_port(8000)
            vnc_port = find_free_port(3000)
        logger.info(f"Use ports: api {api_port}, vnc {vnc_port}")
        container = podman_client.containers.create(
            container_image,
            name=unique_container_name,
            auto_remove=True,
            ports={"8000": api_port, "3000": vnc_port},
            mounts=[
                {
                    "type": "bind",
                    "source": os.path.join(home_dir, "home"),
                    "target": "/config",
                }
            ],
        )
        container.start()
        logger.info("Starting..")
    while container.status != "running":
        time.sleep(0.2)
        container.reload()
    logger.info("Container ready")

    if stop_at_exit:

        def cleanup() -> None:
            try:
                logger.info(f"Stopping container {container.name}")
                container.stop()
                container.remove(force=True)
            except podman.errors.NotFound:
                pass
            atexit.unregister(cleanup)

        atexit.register(cleanup)

    return api_port
