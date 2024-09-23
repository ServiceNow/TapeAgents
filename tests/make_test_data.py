import shutil
import sys
from pathlib import Path

import testbook

from examples import delegate_stack
from tapeagents.utils import run_in_tmp_dir_to_make_test_data

if __name__ == "__main__":
    match sys.argv[1:]:
        case ["delegate_stack"]:
            with run_in_tmp_dir_to_make_test_data("delegate_stack"):
                delegate_stack.main()
        case ["intro_notebook"]:
            intro_notebook_path = Path("intro.ipynb").resolve()
            assets_path = Path("assets").resolve()
            with run_in_tmp_dir_to_make_test_data("intro_notebook", keep_llm_cache=True):
                shutil.copytree(assets_path, Path("assets"))
                with testbook.testbook(intro_notebook_path) as tb:
                    tb.inject(
                        """
                        from tapeagents import llms
                        llms._force_cache = True
                        """,
                        before=0,
                    )
                    tb.execute()
        case _:
            print("Usage: python -m examples.make_test_data [delegate_stack]")
            sys.exit(1)
