import os
import sys
from pathlib import Path

from tapeagents.renderers.camera_ready_renderer import CameraReadyRenderer
from tapeagents.tape_browser import TapeBrowser
from tapeagents.team import TeamTape

tape_dir = Path(sys.argv[1])
os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(tape_dir, "tapedata.sqlite")


browser = TapeBrowser(TeamTape, sys.argv[1], CameraReadyRenderer(), file_extension=".json")
browser.launch(port=7680 if len(sys.argv) < 3 else int(sys.argv[2]))
