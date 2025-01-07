import contextlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

import testbook
import yaml
from omegaconf import DictConfig

import tapeagents.observe
from examples import delegate_stack
from examples.data_science import data_science
from examples.optimize import optimize
from examples.tape_improver import tape_improver


@contextlib.contextmanager
def run_test_in_tmp_dir(test_name: str):
    """Copy test resources to a temporary directory and run the test there"""
    cur_dir = os.getcwd()
    tmpdir = tempfile.mkdtemp()
    test_data_dir = Path(f"{test_name}").resolve()
    os.chdir(tmpdir)
    shutil.copytree(test_data_dir, tmpdir, dirs_exist_ok=True)
    # force creation of SQLite tables
    tapeagents.observe._checked_sqlite = False
    yield tmpdir
    os.chdir(cur_dir)


@contextlib.contextmanager
def run_in_tmp_dir_to_make_test_data(test_group, test_name: str, keep_llm_cache=False):
    cur_dir = os.getcwd()
    tmpdir = tempfile.mkdtemp()
    os.chdir(tmpdir)
    try:
        yield
    # find all non-directory files that got created
    finally:
        created_files = []
        for root, _, files in os.walk(tmpdir):
            for file in files:
                # For most of the code we test in TapeAgents we create ReplayLLM
                # that looks up prompts and outputs in the SQLite database. For this
                # reason, by default we don't save the LLM cache files. If you want
                # make test data for a Jupyter notebook, you can use the keep_llm_cache
                # to save the LLM cache files.
                if file.startswith("llm_cache") and not keep_llm_cache:
                    continue
                created_files.append(os.path.relpath(os.path.join(root, file), tmpdir))
        cp_source = " ".join(f"$TMP/{f}" for f in created_files)
        test_data_dir = f"tests/{test_group}/res/{test_name}"
        print("Saved test data to ", tmpdir)
        print("To update test data, run these commands:")
        print(f"mkdir {test_data_dir}")
        print(f"TMP={tmpdir}; cp {cp_source} {test_data_dir}")
    os.chdir(cur_dir)


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["delegate_stack"]:
            with run_in_tmp_dir_to_make_test_data("examples", "delegate_stack"):
                delegate_stack.main()
        case ["intro_notebook"]:
            intro_notebook_path = Path("intro.ipynb").resolve()
            assets_path = Path("assets").resolve()
            with run_in_tmp_dir_to_make_test_data("examples", "intro_notebook", keep_llm_cache=True):
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
        case ["tape_improver"]:
            with run_in_tmp_dir_to_make_test_data("examples", "tape_improver"):
                tape_improver.main("run improver")
        case ["data_science"]:
            with run_in_tmp_dir_to_make_test_data("examples", "data_science"):
                data_science.main(studio=False)
        case ["optimize"]:
            models = [
                # "gpt-3.5-turbo",
                "gpt-4o-mini",
                # "llama-3.3-70b-instruct"
            ]
            for model in models:
                with open(f"tests/examples/res/optimize/{model}/config.yaml") as f:
                    cfg = DictConfig(yaml.safe_load(f))
                with run_in_tmp_dir_to_make_test_data("examples", f"optimize/{model}", keep_llm_cache=True):
                    optimize.run(cfg)
        case _:
            raise Exception(
                "Usage: python -m tests.make_test_data [delegate_stack | intro_notebook | tape_improver | data_science | optimize]"
            )
