import hydra
from omegaconf import DictConfig

from tapeagents.core import Tape
from tapeagents.demo import Demo
from tapeagents.rendering import GuidedAgentRender

from ..agent import GaiaAgent
from ..environment import GaiaEnvironment
from ..steps import GaiaQuestion
from ..tape import GaiaTape


class GaiaDemo(Demo):
    def add_user_step(self, user_input: str, tape: Tape) -> Tape:
        return tape.append(GaiaQuestion(content=user_input))


@hydra.main(
    version_base=None,
    config_path="../../../conf/tapeagent",
    config_name="gaia_openai",
)
def main(cfg: DictConfig) -> None:
    llm = hydra.utils.instantiate(cfg.llm)
    env = GaiaEnvironment(vision_lm=llm)
    agent = GaiaAgent.create(llm, **cfg.agent)
    demo = GaiaDemo(agent, GaiaTape(), env, GuidedAgentRender())
    demo.launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
