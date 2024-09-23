import os
import logging

from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Type

from pydantic import TypeAdapter
import yaml

from .core import Tape

logger = logging.getLogger(__name__)


class TapeSaver:
    def __init__(self, yaml_dumper: yaml.SafeDumper):
        self._dumper = yaml_dumper

    def save(self, tape: Tape):
        self._dumper.represent(tape.model_dump(by_alias=True))


@contextmanager
def save_tapes(filename: Path, mode: str = "w") -> Generator[TapeSaver, None, None]:
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


def load_tapes(tape_class: Type | TypeAdapter, path: Path | str) -> list[Tape]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    loader = tape_class.model_validate if isinstance(tape_class, Type) else tape_class.validate_python
    with open(path, "r") as file:
        return [loader(data) for data in yaml.safe_load_all(file)]
