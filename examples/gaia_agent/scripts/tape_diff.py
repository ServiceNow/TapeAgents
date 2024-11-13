import sys

from tapeagents.rendering import BasicRenderer
from tapeagents.tape_diff import TapeDiffGUI

from ..eval import load_results, tape_correct


def loader_fn(fname: str) -> tuple[list, dict, str]:  # returns tapes, prompts and title
    results = load_results(fname)
    tapes = results.tapes
    prompts = {p["prompt"]["id"]: p["prompt"] for p in results.prompts}
    acc, n_solved = results.accuracy
    header = f"Accuracy: {acc:.2f}%, {n_solved} out of {len(tapes)}"
    return tapes, prompts, header


def tape_name_fn(n: int, tape: dict, tape2: dict | None) -> str:
    correct = tape_correct(tape)
    task = tape["metadata"]["task"]
    mark = "+" if correct else "-"
    if tape2:
        mark += "+" if tape_correct(tape2) else "-"
    if task["file_name"]:
        mark += "[f]"
    return f"{mark}L{task['Level']}:{n+1}: {task['Question'][:32]}"


if __name__ == "__main__":
    assert len(sys.argv) == 3, "Usage: python -m scripts.tape_diff <fname1> <fname2>"
    gui = TapeDiffGUI(BasicRenderer(), loader_fn, tape_name_fn, sys.argv[1], sys.argv[2])
    gui.launch()
