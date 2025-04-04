import logging
import os

from hydra import compose, initialize
from omegaconf import DictConfig

from tapeagents.core import StopStep, Tape
from tapeagents.dialog_tape import UserStep
from tapeagents.orchestrator import get_agent_and_env_from_config, main_loop

logging.basicConfig(level=logging.INFO)


def main(cfg: DictConfig):
    agent, env = get_agent_and_env_from_config(cfg)
    question = """If Eliud Kipchoge could maintain his record-making marathon pace indefinitely,
    how many thousand hours would it take him to run the distance between the Earth and the Moon its closest approach? 
    Please use the minimum perigee value on the Wikipedia page for the Moon when carrying out your calculation. 
    Round your result to the nearest 1000 hours and do not use any comma separators if necessary.
    """
    tape = Tape(steps=[UserStep(content=question)])
    for event in main_loop(agent, tape, env, max_loops=50):
        if event.observation:
            pass
            # input("Press any key to continue to the next turn...")
        if (
            event.agent_event
            and event.agent_event.final_tape
            and isinstance(event.agent_event.final_tape[-1], StopStep)
        ):
            print("Run ended")
            break
    env.close()


if __name__ == "__main__":
    with initialize(config_path="../../conf"):
        cfg = compose(config_name="gaia_mcp.yaml")
    assert os.environ.get("OPENAI_API_KEY") is not None, "set OPENAI_API_KEY env var"
    assert os.environ.get("SERPER_API_KEY") is not None, "set SERPER_API_KEY env var"
    main(cfg)
