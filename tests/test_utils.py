

import json
import os

from tapeagents.core import TrainingText


def load_tape_dict(run_dir: str, fname: str = "tape.json") -> dict:
    tape_fpath = os.path.join(run_dir, fname)
    with open(tape_fpath, "r") as f:
        tape_dict = json.load(f)
    return tape_dict


def load_traces(run_dir: str, fname: str = "traces.json") -> list[TrainingText]:
    traces_fpath = os.path.join(run_dir, fname)
    with open(traces_fpath, "r") as f:
        traces = [TrainingText.model_validate(t) for t in json.load(f)]
    return traces
