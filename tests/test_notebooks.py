import pathlib
from pathlib import Path
import testbook

res_dir = f"{pathlib.Path(__file__).parent.resolve()}/res"

def test_intro_notebook():
    intro_notebook_path = Path("intro.ipynb").resolve()
    with testbook.testbook(intro_notebook_path) as tb:
        tb.inject(
            f"""
            from tapeagents import llms
            llms._force_cache = True
            import os
            os.chdir("{res_dir}/intro_notebook")
            """,
            before=0
        )
        tb.execute()