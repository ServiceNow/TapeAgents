import logging
import os
import sys
from pathlib import Path

from tapeagents.core import Tape
from tapeagents.io import load_tapes
from tapeagents.renderers.camera_ready_renderer import CameraReadyRenderer
from tapeagents.tape_browser import TapeBrowser

from ..tape import FormFillerTape
from ..user_simulator_agent import UserSimulatorTape

# Syntax: python -m examples.form_filler.scripts.tape_browser <path_to_tapes_folder> [<path_to_sqlite_db>] [<port>]

tape_dir = Path(sys.argv[1])

if len(sys.argv) >= 3:
    os.environ["TAPEAGENTS_SQLITE_DB"] = sys.argv[2]

logger = logging.getLogger(__name__)


# Configure to show info messages in the console
logging.basicConfig(level=logging.INFO)


class RecursiveTapeBrowser(TapeBrowser):
    def load_tapes(self, fname: str) -> list[Tape]:
        tapes = []
        try:
            tapes = load_tapes(FormFillerTape, fname)
            logger.info(f"{len(tapes)} FormFillerTape tapes loaded from {fname}")
        except Exception as e:
            try:
                tapes = load_tapes(UserSimulatorTape, fname)
                logger.info(f"{len(tapes)} UserSimulatorTape loaded from {fname}")
            except Exception as e2:
                logger.error(f"Could not load tapes from {fname}.")
                logger.error(f"Tried loading as FormFillerTape, got this error: {e}")
                logger.error(f"Tried loading as UserSimulatorTape, got this error: {e2}")
        return tapes

    def get_tape_files(self) -> list[str]:
        files = sorted(
            [
                os.path.join(root, f)
                for root, _, filenames in os.walk(self.tapes_folder)
                for f in filenames
                if f.endswith(self.file_extension) and ".hydra" not in root
            ]
        )
        assert files, f"No files found in {self.tapes_folder}"
        logger.info(f"{len(files)} files found in {self.tapes_folder}")
        indexed = 0
        nonempty_files = []
        for i, file in enumerate(files):
            if file in self.files:
                continue  # already indexed
            tapes = []
            try:
                tapes = self.load_tapes(file)
            except Exception as e:
                logger.error(f"Could not load tapes from {file}: {e}")
                continue
            if not len(tapes):
                logger.warning(f"File {file} does not contain any known tapes, skip")
                continue
            for j, tape in enumerate(tapes):
                tape_id = tape.metadata.id
                parent_id = tape.metadata.parent_id
                if tape_id:
                    if tape_id in self.tape_index and self.tape_index[tape_id] != (i, j):
                        raise ValueError(
                            f"Duplicate tape id {tape_id}. Both in {self.tape_index[tape_id]} and {(i, j)}"
                        )
                    indexed += 1
                    self.tape_index[tape_id] = (i, j)
                    if parent_id:
                        if parent_id not in self.tape_children:
                            self.tape_children[parent_id] = []
                        self.tape_children[parent_id].append(tape_id)
            nonempty_files.append(file)
        logger.info(f"Indexed {indexed} new tapes, index size: {len(self.tape_index)}")
        logger.info(f"{len(self.tape_children)} tapes with children found")
        return nonempty_files


browser = RecursiveTapeBrowser(Tape, sys.argv[1], CameraReadyRenderer(), file_extension=".yaml")
browser.launch(port=7680 if len(sys.argv) < 4 else int(sys.argv[3]))
