"""
I/O routines for Tapes.
"""

import json
import logging
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Type

import yaml
from pydantic import TypeAdapter

from tapeagents.agent import Agent
from tapeagents.config import ATTACHMENT_DEFAULT_DIR
from tapeagents.core import Tape, TapeType
from tapeagents.steps import UnknownStep, VideoObservation

logger = logging.getLogger(__name__)


class TapeSaver:
    """A class for saving Tape objects using YAML format.

    This class provides functionality to save Tape objects using a YAML dumper.
    It handles the serialization of Tape objects into YAML format.

    Example:
        ```python
        dumper = yaml.SafeDumper(output_file)
        saver = TapeSaver(dumper)
        saver.save(tape)
        ```
    """

    def __init__(self, yaml_dumper: yaml.SafeDumper):
        """Initialize TapeIOYML with a YAML dumper.

        Args:
            yaml_dumper (yaml.SafeDumper): The YAML dumper instance to use for serialization.
        """
        self._dumper = yaml_dumper

    def save(self, tape: Tape):
        """
        Saves the tape data using the configured dumper.

        Args:
            tape (Tape): The tape object containing the data to be saved.
        """
        self._dumper.represent(tape.model_dump(by_alias=True))


@contextmanager
def stream_yaml_tapes(filename: Path | str, mode: str = "w") -> Generator[TapeSaver, None, None]:
    """Stream YAML tapes to a file.

    This function creates a context manager that allows streaming YAML documents to a file.
    It handles file creation, directory creation if necessary, and proper resource cleanup.

    Args:
        filename (Union[Path, str]): Path to the output YAML file. Can be either a string or Path object.
        mode (str, optional): File opening mode. Defaults to "w" (write mode).

    Yields:
        Generator[TapeSaver, None, None]: A TapeSaver instance that can be used to write YAML documents.

    Raises:
        OSError: If there are issues with file/directory creation or permissions.
        yaml.YAMLError: If there are YAML serialization errors.
    """
    if isinstance(filename, str):
        filename = Path(filename)
    logger.info(f"Writing to {filename} in mode {mode}")

    # Create directory path if it does not exist
    filename.parent.mkdir(parents=True, exist_ok=True)

    # Open file for writing and create dumper instance
    _file = open(filename, mode)
    _dumper = yaml.SafeDumper(
        stream=_file,
        default_flow_style=False,
        explicit_start=True,
        sort_keys=False,
    )
    _dumper.open()

    # Yield the dumper to the caller
    yield TapeSaver(_dumper)

    # Close the dumper and file
    _dumper.close()
    _file.close()


def save_json_tape(tape: Tape, tapes_dir: str, name: str = ""):
    """Save a Tape object to a JSON file.

    Args:
        tape (Tape): The Tape object to be saved
        tapes_dir (str): Directory path where the JSON file will be saved
        name (str, optional): Name of the output JSON file. If empty, tapes_dir is used as the full path.
            If provided without .json extension, it will be added automatically. Defaults to "".

    Example:
        ```python
        tape = Tape(...)
        save_json_tape(tape, "/path/to/dir", "my_tape")
        # Saves to /path/to/dir/my_tape.json
        ```

    """
    if name:
        os.makedirs(tapes_dir, exist_ok=True)
    fname = name if name.endswith(".json") else f"{name}.json"
    fpath = os.path.join(tapes_dir, fname) if name else tapes_dir
    with open(fpath, "w") as f:
        f.write(tape.model_dump_json(indent=4))


def save_tape_images(tape: Tape, images_dir: str):
    os.makedirs(images_dir, exist_ok=True)
    for i, step in enumerate(tape):
        if hasattr(step, "image_path"):
            image_path = os.path.join(images_dir, f"{step.metadata.id}.png")
            shutil.copy(step.image_path, image_path)


def load_tape_dicts(path: Path | str, file_extension: str = ".yaml") -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    if file_extension not in (".yaml", ".json"):
        raise ValueError(f"Unsupported file extension: {file_extension}")
    if os.path.isdir(path):
        logger.info(f"Loading tapes from dir {path}")
        paths = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(file_extension)])
    else:
        paths = [path]
        file_extension = os.path.splitext(path)[-1]
    tapes = []
    for path in paths:
        with open(path) as f:
            if file_extension == ".yaml":
                data = list(yaml.safe_load_all(f))
            else:
                data = json.load(f)
        if not isinstance(data, list):
            data = [data]
        tapes.extend(data)
    return tapes


def load_tapes(
    tape_class: Type[TapeType],
    path: Path | str,
    file_extension: str = ".yaml",
    attachment_dir: str = None,
) -> list[TapeType]:
    """Load tapes from dir with YAML or JSON files.

    This function loads tapes from a file or directory and converts them into tape objects
    using the specified tape class or type adapter.

    Args:
        tape_class (Type[TapeType]): The class or type adapter used to validate and create tape objects.
        path (Union[Path, str]): Path to a file or directory containing tape configurations.
        file_extension (str, optional): File extension to filter by when loading from directory.
            Must be either '.yaml' or '.json'. Defaults to '.yaml'.
        attachment_dir (str, optional): The directory to use for attachments. If None, a default directory will be used.

    Returns:
        list[TapeType]: A list of validated tape objects.

    Raises:
        FileNotFoundError: If the specified path does not exist.
        ValueError: If an unsupported file extension is provided.

    Example:
        ```python
        tapes = load_tapes(TapeClass, "configs/tapes.yaml")
        tapes = load_tapes(tape_adapter, "configs/tapes", ".json")
        ```
    """
    tapes = []
    data = load_tape_dicts(path, file_extension)
    attachment_dir_resolved = get_attachment_dir(path, attachment_dir)
    for tape_dict in data:
        tape = tape_class.model_validate(tape_dict)
        if attachment_dir_resolved:
            # Update attachment_dir for steps that needs it
            for step in tape:
                if isinstance(step, (VideoObservation)):
                    step.attachment_dir = attachment_dir_resolved
        tapes.append(tape)
    return tapes


def load_legacy_tapes(tape_class: Type[TapeType], path: Path | str, step_class: Type | TypeAdapter) -> list[TapeType]:
    tapes = []
    data = load_tape_dicts(path, ".json")
    for tape_dict in data:
        try:
            tape = tape_class.model_validate(tape_dict)
        except Exception:
            step_dicts = tape_dict["steps"]
            tape_dict["steps"] = []
            tape = tape_class.model_validate(tape_dict)
            step_loader = step_class.model_validate if isinstance(step_class, Type) else step_class.validate_python
            steps = []
            for step_dict in step_dicts:
                try:
                    steps.append(step_loader(step_dict))
                except Exception as e:
                    logger.warning(f"Failed to load step: {e}")
                    steps.append(UnknownStep(content=json.dumps(step_dict, indent=2, ensure_ascii=False)))
            tape.steps = steps
        tapes.append(tape)
    return tapes


def save_agent(agent: Agent, filename: str) -> str:
    with open(filename, "w") as f:
        yaml.dump(agent.model_dump(), f)
    return filename


def get_attachment_dir(tape_path: Path | str, attachment_dir: Path | str) -> str | None:
    """
    Determines the directory to use for tape attachments.

    Args:
        tape_path (Path | str]): Path to a file or directory containing tape configurations.
        attachment_dir (Path | str): The directory to use for attachments. If None, a default directory will be used.

    Returns:
        str | None: The path to the attachment directory if it exists, otherwise None.

    Raises:
        FileNotFoundError: If the provided attachment_dir does not exist.

    Examples:
        >>> get_attachment_dir("data/tapes.yaml", "data/images")
        'data/images'

        >>> get_attachment_dir("data/tapes.yaml", None)
        'data/attachments'

        >>> get_attachment_dir("data/tapes/", None)
        'data/attachments'
    """
    if attachment_dir:
        # Use attachment_dir if provided and exists
        if not Path(attachment_dir).is_dir():
            raise FileNotFoundError(f"Tape attachment directory not found: {attachment_dir}")
        return str(attachment_dir)
    else:
        # Use ATTACHMENT_DEFAULT_DIR if exists
        path_obj = Path(tape_path)
        default_path = path_obj.parent.parent if path_obj.is_file() else path_obj.parent
        default_path /= ATTACHMENT_DEFAULT_DIR
        if Path(default_path).is_dir():
            logger.info(f"Use tape attachment director: {default_path}")
            return str(default_path)
    return None
