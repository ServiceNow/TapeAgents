import sys
import testbook
from pathlib import Path

from tapeagents.examples import delegate_stack
from tapeagents.utils import run_in_tmp_dir_to_make_test_data

if __name__ == "__main__":    
    match sys.argv[1:]:
        case ["delegate_stack"]:
            with run_in_tmp_dir_to_make_test_data("delegate_stack"):
                delegate_stack.main()
        case ["intro_notebook"]:
            intro_notebook_path = Path("tapeagents/examples/intro.ipynb").resolve()
            with run_in_tmp_dir_to_make_test_data("intro_notebook", keep_llm_cache=True):
                with testbook.testbook(intro_notebook_path) as tb:
                    tb.inject(
                        """
                        from tapeagents import llms
                        llms._force_cache = True
                        """,
                        before=0
                    )
                    tb.execute()
        case _:
            print(f"Usage: python -m tapeagents.examples.make_test_data [delegate_stack]")
            sys.exit(1)