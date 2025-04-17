import gzip
import logging
import os
import shutil
from pathlib import Path

import yaml
from omegaconf import DictConfig

from .evaluate import main as evaluate

logging.basicConfig(level=logging.INFO)

test_path = Path("tests/examples/res/gaia_agent/")

run_dir = Path("/tmp/gaiatest")
if run_dir.exists():
    print(f"Tmp dir exists {run_dir}, removing")
    shutil.rmtree(run_dir)

run_dir.mkdir(parents=True)
cache_dir = run_dir / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["TAPEAGENTS_CACHE_DIR"] = str(cache_dir)
os.environ["TAPEAGENTS_SQLITE_DB"] = str(run_dir / "tapedata.sqlite")
(run_dir / ".hydra").mkdir(parents=True, exist_ok=True)
shutil.copyfile(test_path / "config.yaml", run_dir / ".hydra" / "config.yaml")
print("Tmp dir ready:", os.listdir(run_dir))

with open(test_path / "config.yaml") as f:
    cfg = DictConfig(yaml.safe_load(f))
cfg.exp_path = str(run_dir)
cfg.environment.tools[2].mock = False
evaluate(cfg)

print("Done, now copy the tapes, cache and db")
shutil.copytree(run_dir / "tapes", test_path / "tapes", dirs_exist_ok=True)
shutil.copytree(run_dir / "cache", test_path / "cache", dirs_exist_ok=True)
with open(run_dir / "tapedata.sqlite", "rb") as f_in:
    with gzip.open(test_path / "tapedata.sqlite.gz", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
print("Test data updated, now run the tests with `make test`")
