from tapeagents.finetune.data import load_samples
from tapeagents.finetune.finetune import load_config, run_finetuning_loop

train_samples_file = "gsm8k/tuning/llama31_70b_train_t02/training_samples_3k.jsonl"
training_samples = load_samples(train_samples_file)
cfg = load_config("llama31_8b", output_dir="gsm8k/tuning/llama31_70b_train_t02/tune_llama31_8b_1")
run_finetuning_loop(cfg=cfg, training_samples=training_samples)
