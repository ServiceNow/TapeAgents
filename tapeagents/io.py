"""
I/O utilities for Tape objects.
"""

import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Type

import yaml
from pydantic import TypeAdapter

from .core import Tape

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
    fname = name if name.endswith(".json") else f"{name}.json"
    fpath = os.path.join(tapes_dir, fname) if name else tapes_dir
    with open(fpath, "w") as f:
        f.write(tape.model_dump_json(indent=4))


def load_tapes(tape_class: Type | TypeAdapter, path: Path | str, file_extension: str = ".yaml") -> list[Tape]:
    """Load tapes from dir with YAML or JSON files.

    This function loads tapes from a file or directory and converts them into tape objects
    using the specified tape class or type adapter.

    Args:
        tape_class (Union[Type, TypeAdapter]): The class or type adapter used to validate and create tape objects.
        path (Union[Path, str]): Path to a file or directory containing tape configurations.
        file_extension (str, optional): File extension to filter by when loading from directory.
            Must be either '.yaml' or '.json'. Defaults to '.yaml'.

    Returns:
        list[Tape]: A list of validated tape objects.

    Raises:
        FileNotFoundError: If the specified path does not exist.
        ValueError: If an unsupported file extension is provided.

    Example:
        ```python
        tapes = load_tapes(TapeClass, "configs/tapes.yaml")
        tapes = load_tapes(tape_adapter, "configs/tapes", ".json")
        ```
    """
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
    loader = tape_class.model_validate if isinstance(tape_class, Type) else tape_class.validate_python
    tapes = []
    for path in paths:
        with open(path) as f:
            if file_extension == ".yaml":
                data = list(yaml.safe_load_all(f))
            else:
                data = json.load(f)
        if not isinstance(data, list):
            data = [data]
        for tape in data:
            tapes.append(loader(tape))
    return tapes
