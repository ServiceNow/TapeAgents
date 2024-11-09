import os
from pathlib import Path
import sys

from tapeagents.renderers.camera_ready_renderer import CameraReadyRenderer
from tapeagents.tape_browser import TapeBrowser
from tapeagents.team import TeamTape

# comment this code out if loading the prompt and completions takes too long for you
tape_dir = Path(sys.argv[1])
exp_dir = tape_dir
# try to find a parent directory for tape_dir path that contains tapedata.sqlite
while not os.path.exists(exp_dir / "tapedata.sqlite") and exp_dir != Path("."):
    exp_dir = exp_dir.parent
os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(exp_dir, "tapedata.sqlite")


browser = TapeBrowser(TeamTape, sys.argv[1], CameraReadyRenderer(), file_extension=".json")
browser.launch(port=7680 if len(sys.argv) < 3 else int(sys.argv[2]))