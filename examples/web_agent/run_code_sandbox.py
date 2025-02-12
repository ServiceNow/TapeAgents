import omegaconf

from tapeagents.tools.container_executor import init_code_sandbox

cfg = omegaconf.OmegaConf.load("conf/web_agent.yaml")
init_code_sandbox(cfg.exp_path, no_deps=True)
while True:
    pass
