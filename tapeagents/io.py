import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Type

import yaml

from .core import Tape, TapeType

logger = logging.getLogger(__name__)


class TapeSaver:
    def __init__(self, yaml_dumper: yaml.SafeDumper):
        self._dumper = yaml_dumper

    def save(self, tape: Tape):
        self._dumper.represent(tape.model_dump(by_alias=True))


@contextmanager
def stream_yaml_tapes(filename: Path | str, mode: str = "w") -> Generator[TapeSaver, None, None]:
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
    fname = name if name.endswith(".json") else f"{name}.json"
    fpath = os.path.join(tapes_dir, fname) if name else tapes_dir
    with open(fpath, "w") as f:
        f.write(tape.model_dump_json(indent=4))


def load_tapes(tape_class: Type[TapeType], path: Path | str, file_extension: str = ".yaml") -> list[TapeType]:
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
