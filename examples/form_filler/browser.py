import sys

from examples.form_filler.tape import FormFillerTape
from tapeagents.rendering import PrettyRenderer
from tapeagents.tape_browser import TapeBrowser


def main(dirname: str):
    renderer = PrettyRenderer()
    browser = TapeBrowser(FormFillerTape, dirname, renderer)
    browser.launch()


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python -m tapeagents.examples.ghreat.browser <dirname>"
    main(sys.argv[1])
