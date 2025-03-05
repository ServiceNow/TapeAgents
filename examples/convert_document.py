import argparse

from hydra import compose, initialize
from omegaconf import DictConfig

from tapeagents.dialog_tape import DialogTape, UserStep
from tapeagents.orchestrator import get_agent_and_env_from_config, main_loop


def main(cfg: DictConfig, path: str) -> None:
    agent, env = get_agent_and_env_from_config(cfg)

    print("Run the agent!")
    for event in main_loop(
        agent,
        DialogTape() + [UserStep(content=f"Read and convert the document at `{path}` and return its results to me")],
        env,
    ):
        if ae := event.agent_event:
            if ae.step:
                print(ae.step.model_dump_json(indent=2))
        if event.observation:
            print(event.observation.model_dump_json(indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", "-i", type=str, required=True, help="Document to convert")
    args = parser.parse_args()
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="convert_document")
    main(cfg, path=args.input_path)
