import hydra

from tapeagents.finetune.finetune import run_finetuning_loop


@hydra.main(version_base=None, config_path="../../conf", config_name="finetune")
def finetune_with_config(cfg):
    run_finetuning_loop(cfg)


if __name__ == "__main__":
    finetune_with_config()
