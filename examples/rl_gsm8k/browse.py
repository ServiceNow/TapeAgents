import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

from tapeagents.rendering import PrettyRenderer
from tapeagents.tape_browser import TapeBrowser
from examples.gsm8k_tuning.math_agent import MathTape


TapeBrowser(MathTape, "outputs/rl_debug_v6/tapes/train", PrettyRenderer(), file_extension=".json").launch()