import os
import sys
from pathlib import Path

from examples.rl_gsm8k.cot_math_agent import MathTape
from tapeagents.renderers.camera_ready_renderer import CameraReadyRenderer
from tapeagents.tape_browser import TapeBrowser

# comment this code out if loading the prompt and completions takes too long for you
tape_dir = Path(sys.argv[1])
exp_dir = tape_dir
# try to find a parent directory for tape_dir path that contains llm_calls.sqlite
while not os.path.exists(exp_dir / "llm_calls.sqlite") and exp_dir != Path("."):
    exp_dir = exp_dir.parent
os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(exp_dir, "llm_calls.sqlite")


browser = TapeBrowser(MathTape, sys.argv[1], CameraReadyRenderer(), file_extension=".json")
browser.launch(port=7680 if len(sys.argv) < 3 else int(sys.argv[2]))
