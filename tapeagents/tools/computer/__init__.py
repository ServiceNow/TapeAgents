"""
Computer tool, based on claude computer use demo
"""

import atexit
import logging
import os
import time

import podman

logger = logging.getLogger("computer")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(funcName)s - %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def launch_container(container_image="computer", container_name="computer", stop_at_exit=False):
    podman_client = podman.from_env()
    try:
        container = podman_client.containers.get(container_name)
    except podman.errors.NotFound:
        logger.info(f"Creating container from image '{container_image}'")
        home_dir = os.path.join(os.getcwd(), ".computer")
        zip_home = os.path.join(os.path.dirname(__file__), "home.zip")
        if os.path.exists(home_dir):
            os.system(f"rm -rf {home_dir}")
            logger.info(f"Removed existing home directory: {home_dir}")
        else:
            os.makedirs(home_dir)
        os.system(f"unzip -qq -o {zip_home} -d {home_dir}")
        logger.info(f"Recreated home directory from zip: {zip_home}")
        container = podman_client.containers.create(
            container_image,
            name=container_name,
            auto_remove=True,
            ports={
                "3000": 3000,
                "8000": 8000,
            },
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
            except podman.errors.NotFound:
                pass
            atexit.unregister(cleanup)

        atexit.register(cleanup)
