from hydra import compose, initialize

from tapeagents.tools.container_executor import init_code_sandbox

with initialize(version_base=None, config_path="../../../conf", job_name="computer_demo"):
    cfg = compose(config_name="gaia_demo")
init_code_sandbox(cfg.exp_path, no_deps=True)
while True:
    pass
