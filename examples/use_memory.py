import hydra
from omegaconf import DictConfig

from tapeagents.dialog_tape import DialogTape, UserStep
from tapeagents.orchestrator import get_agent_and_env_from_config, main_loop


def do_loop(agent, start_tape, env):
    for event in main_loop(agent, start_tape, env):
        if ae := event.agent_event:
            if ae.step:
                print(ae.step.model_dump_json(indent=2))
            if ae.final_tape:
                final_tape = ae.final_tape
        if event.observation:
            print(event.observation.model_dump_json(indent=2))
    return final_tape


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="use_memory",
)
def main(cfg: DictConfig):
    agent, env = get_agent_and_env_from_config(cfg)

    print("Run the agent!")
    do_loop(
        agent, DialogTape() + [UserStep(content="Perform the calculation 2 + 2. Perform the calculation 3 + 3.")], env
    )
    do_loop(
        agent, DialogTape() + [UserStep(content="Perform the calculation 2 + 2. Perform the calculation 5 + 2.")], env
    )


if __name__ == "__main__":
    main()
