import contextlib
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from transformers import (
    get_scheduler,
    set_seed,
)

from tapeagents.core import TrainingText

from .checkpoints import (
    load_model,
    load_tokenizer,
    load_training_state,
    remove_results,
    save_model_and_tokenizer,
    save_training_state,
)
from .context import accelerator, logger
from .data import create_dataloader, prepare_dataloaders
from .eval import evaluate_and_get_metrics
from .logging_ import log_metrics, log_time, setup_logging
from .optim import get_optimizer
from .rl import RLConfig, make_rl_data_callback, rl_step
from .rl.utils import get_avg_rl_stats
from .types import DataArgs, DataPartArgs, ModelClass, TrainingMetrics


def load_config(config_name: str, config_dir: str = "../../conf/finetune", output_dir: str = "./output") -> DictConfig:
    with initialize(version_base=None, config_path=config_dir, job_name=config_name):
        config = DictConfig(dict(finetune=compose(config_name=config_name), output_dir=output_dir))
    return config


def run_finetuning_loop(
    cfg: DictConfig,
    data_path: str | None = None,
    training_samples: list[TrainingText] | None = None,
    interrupt_train_steps: int = -1,
    wandb_run=None,
):
    dt = time.perf_counter()
    num_processes = accelerator.state.num_processes  # type: ignore
    args = cfg.finetune if "finetune" in cfg else cfg
    eval_fn = instantiate(args.eval_callback)
    if training_samples:
        logger.info(f"Using {len(training_samples)} training samples")
        data_args = DataArgs(data_parts_train=[])
    elif data_path is None:
        data_args = DataArgs(**OmegaConf.to_object(args.data))  # type: ignore
    else:
        data_args = DataArgs(data_parts_train=[DataPartArgs(path=data_path)])

    output_dir = Path(cfg.output_dir)
    model_class: ModelClass = args.model_class
    objective = args.get("objective", "nll")
    if objective == "rl":
        is_rl = True
    elif objective == "nll":
        is_rl = False
    else:
        raise ValueError(f"Unknown training objective {objective}")

    with open_dict(args):
        # gradient accumulation steps must be divisible by num_processes
        original_accum_passes = args.gradient_accumulation_passes
        if original_accum_passes % num_processes != 0:
            # round up to the next multiple of num_processes
            new_accum_passes = ((original_accum_passes + num_processes - 1) // num_processes) * num_processes
            logger.warning(
                f"Adjusting gradient_accumulation_passes from {original_accum_passes} to {new_accum_passes} "
                f"to make it divisible by {num_processes} processes"
            )
            args.gradient_accumulation_passes = new_accum_passes

        args.effective_batch_size = int(args.train_batch_size) * int(args.gradient_accumulation_passes)

    args.gradient_accumulation_passes //= num_processes
    samples_per_pass = num_processes * args.train_batch_size
    set_seed(args.seed)

    # using a subfolder makes "safe" overwriting possible
    current_dir = output_dir / "current"
    intermediate_root_dir = output_dir / "intermediate"
    training_state_dir = output_dir / "training_state"
    log_dir = output_dir / "logs"

    if args.force_restart and accelerator.is_main_process:
        remove_results(current_dir, intermediate_root_dir, training_state_dir, log_dir)

    # Logging
    setup_logging(cfg, output_dir, run=wandb_run)
    logger.info(accelerator.state)
    logger.info(f"Saving experiment to {output_dir}")
    dt = log_time(dt, "finetune/startup")

    tokenizer = load_tokenizer(args.config_name)
    model = load_model(args, model_class, current_dir)

    dt = log_time(dt, "finetune/model_load")

    forward = lambda model, batch: (model(**batch).loss, {})  # noqa: E731
    rl_data_callback = None
    if is_rl:
        rl_config = RLConfig(**args.rl)
        forward = lambda model, batch: rl_step(model, batch, rl_config)  # noqa: E731
        rl_data_callback = make_rl_data_callback(args, current_dir, rl_config, model)

    dataloader_rng = torch.Generator()
    if training_samples:
        eval_dataloader = None
        dev_dataloader = None
        train_dataloader = create_dataloader(
            training_samples,
            tokenizer=tokenizer,
            seq_length=args.seq_length,
            batch_size=args.train_batch_size,
            rl_data_callback=rl_data_callback,
            is_rl=is_rl,
        )
    else:
        train_dataloader, eval_dataloader, dev_dataloader = prepare_dataloaders(
            args,
            data_args,
            tokenizer,
            rl_data_callback,
            dataloader_rng,
            is_rl=is_rl,
        )

    accelerator.wait_for_everyone()
    dt = log_time(dt, "finetune/data_load")

    optimizer = get_optimizer(args.optim, model, args.learning_rate, args.weight_decay)
    lr_scheduler = get_scheduler(args.lr_scheduler_type, optimizer, args.num_warmup_steps, args.max_train_steps)

    # Wrap everything with HF Accelerator
    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        dev_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader, dev_dataloader, lr_scheduler)

    training_metrics = TrainingMetrics()
    rl_metrics = defaultdict(list)
    if os.path.exists(training_state_dir):
        # WARNING: In case of deepspeed this will overwrite model weights too
        training_metrics = load_training_state(training_state_dir, model, optimizer, lr_scheduler, training_metrics)
        training_metrics.lr = optimizer.param_groups[0]["lr"]
        logger.info("LR after loading training state: %.2E" % training_metrics.lr)
        dt = log_time(dt, "finetune/training_state_load")

    @contextlib.contextmanager
    def toggle_sync(sync: bool):
        """Wrap accelerate.no_sync() if sync is False."""
        if sync:
            yield  # do not enforce no_sync mode
        else:
            with accelerator.no_sync(model):
                yield

    final_train_steps = calculate_train_steps(args, interrupt_train_steps)
    if training_metrics.completed_steps == final_train_steps:
        logger.info("Training is already completed")
        return

    logger.info("Start training")
    model.train()
    dt = log_time(dt, "finetune/prepare_training")
    last_dataloader_position = training_metrics.passes % len(train_dataloader)
    while training_metrics.completed_steps < final_train_steps:
        training_metrics.epoch = training_metrics.passes // len(train_dataloader)

        # each epoch will have unique but predetermined order of batches. It allows to resume training from exactly the same position
        dataloader_rng.manual_seed(training_metrics.epoch)

        for i, batch in enumerate(train_dataloader):
            num_tokens = batch["input_ids"].numel()
            if args.resume_dataloader and last_dataloader_position != 0:
                if i < last_dataloader_position:  # rewind train dataloader to the last used position
                    continue
                last_dataloader_position = 0  # the following dataloader loop should start from 0 after rewinding
                logger.info(f"Resumed dataloader from epoch {training_metrics.epoch}, position {i}")

            time_before = time.time()
            training_metrics.passes += 1
            training_metrics.samples = training_metrics.passes * samples_per_pass

            if args.cuda_empty_cache:
                torch.cuda.empty_cache()

            do_optimizer_step = training_metrics.passes % args.gradient_accumulation_passes == 0
            with toggle_sync(do_optimizer_step):
                loss, this_step_rl_metrics = forward(model, batch)
                for k, v in this_step_rl_metrics.items():
                    rl_metrics[k].append(v)
                training_metrics.train_loss = loss.item()
                training_metrics.lr = optimizer.param_groups[0]["lr"]
                training_metrics.max_batch_len = max(batch["input_ids"].shape[1], training_metrics.max_batch_len)
                training_metrics.min_batch_len = min(batch["input_ids"].shape[1], training_metrics.min_batch_len)
                accelerator.backward(loss / args.gradient_accumulation_passes)

            if not do_optimizer_step:
                continue

            # All gradients have been accumulated, we can now do an optimizer step
            training_metrics.completed_steps += 1
            if args.gradient_clipping_threshold:
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.gradient_clipping_threshold)
                # grad_norm is None when using DeepSpeed
                training_metrics.grad_norm = grad_norm.item() if grad_norm else -1.0
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            step_took = time.time() - time_before

            metrics_dict = {}
            time_to_stop = training_metrics.completed_steps >= final_train_steps
            time_to_log = training_metrics.completed_steps % args.log_each_n_steps == 0
            time_to_save = (training_metrics.completed_steps % args.save_checkpoint_steps == 0) or (
                len(args.also_save_steps) and training_metrics.completed_steps in args.also_save_steps
            )
            time_to_save = time_to_save and not time_to_stop
            if time_to_log or time_to_save:
                dt = log_time(dt, "finetune/interim_eval")
                metrics_dict.update(
                    {
                        "stats/lr": training_metrics.lr,
                        "stats/grad_norm": training_metrics.grad_norm,
                        "stats/samples": training_metrics.passes * samples_per_pass,
                        "stats/passes": training_metrics.passes,
                        "stats/completed_steps": training_metrics.completed_steps,
                        "stats/epoch": training_metrics.epoch,
                        "throughput/tokens_perGPU_per_sec": num_tokens / step_took,
                        "throughput/tokens_per_sec": num_tokens * num_processes / step_took,
                        "throughput/passes_per_sec": 1 / step_took,
                        "throughput/steps_per_sec": 1 / args.gradient_accumulation_passes / step_took,
                        "throughput/sec_per_step": step_took / args.gradient_accumulation_passes,
                        "loss/train": training_metrics.train_loss,
                        "dataset_stats/max_batch_len": training_metrics.max_batch_len,
                        "dataset_stats/min_batch_len": training_metrics.min_batch_len,
                    }
                )

                metrics_dict.update(get_avg_rl_stats(rl_metrics))
                rl_metrics = defaultdict(list)

            if time_to_save:
                # Overwrite latest model at pytorch_model.bin (for later JGA evaluation *and* for resuming training)
                save_model_and_tokenizer(
                    current_dir,
                    model,
                    tokenizer,
                    args.lora.enabled,
                    safe_serialization=args.use_safetensors,
                )
                # Save training state to training_state.pt (for resuming).
                save_training_state(
                    training_state_dir,
                    model,
                    optimizer,
                    lr_scheduler,
                    asdict(training_metrics),
                )

                training_metrics = evaluate_and_get_metrics(
                    args, model, eval_dataloader, dev_dataloader, training_metrics
                )
                if not is_rl:
                    metrics_dict.update(
                        {
                            "loss/eval": training_metrics.eval_loss,
                            "loss/dev": training_metrics.dev_loss,
                            "loss/perplexity": np.exp(training_metrics.eval_loss),
                            "best/completed_steps": training_metrics.best_completed_steps,
                            "best/eval_loss": training_metrics.best_eval_loss,
                            "best/perplexity": np.exp(training_metrics.best_eval_loss),
                        }
                    )

                if args.keep_intermediate_checkpoints:
                    intermediate_dir = intermediate_root_dir / str(training_metrics.completed_steps)
                    save_model_and_tokenizer(
                        intermediate_dir,
                        model,
                        tokenizer,
                        args.lora.enabled,
                        safe_serialization=args.use_safetensors,
                    )
                    dt = log_time(dt, "finetune/interim_save")
                    try:
                        # run external evaluation callback on a saved checkpoint
                        if accelerator.is_main_process:
                            external_metrics = eval_fn(str(intermediate_dir))
                            metrics_dict.update({f"eval/{k}": v for k, v in external_metrics.items()})
                    except Exception as e:
                        logger.error(f"Failed to run eval on checkpoint {intermediate_dir}: {e}")
                    dt = log_time(dt, "finetune/external_eval")

                if args.cuda_empty_cache:
                    torch.cuda.empty_cache()

            accelerator.wait_for_everyone()  # wait for the main process that saves the model and runs eval
            if len(metrics_dict):
                log_metrics(logger, training_metrics.completed_steps, metrics_dict)

            if training_metrics.completed_steps >= final_train_steps:
                logger.info(f"Reached final step {final_train_steps}, stopping.")
                break

        logger.info(f"epoch {training_metrics.epoch} ended")
    dt = log_time(dt, "finetune/train_loop")

    # save the last checkpoint
    logger.info("Final model evaluation")
    training_metrics = evaluate_and_get_metrics(args, model, eval_dataloader, dev_dataloader, training_metrics)
    dt = log_time(dt, "finetune/final_eval")

    logger.info("Final model saving")
    save_model_and_tokenizer(
        current_dir,
        model,
        tokenizer,
        args.lora.enabled,
        safe_serialization=args.use_safetensors,
    )
    dt = log_time(dt, "finetune/final_save")
    if args.save_final_training_state:
        save_training_state(training_state_dir, model, optimizer, lr_scheduler, asdict(training_metrics))
        dt = log_time(dt, "finetune/final_training_state_save")

    if accelerator.is_main_process:
        with open(output_dir / "summary.json", "w") as wf:
            json.dump(asdict(training_metrics), wf, indent=4, sort_keys=True)
        if is_rl:
            with open(output_dir / "rl_summary.json", "w") as wf:
                json.dump(rl_metrics, wf, indent=4, sort_keys=True)

    torch.cuda.empty_cache()


def calculate_train_steps(args, interrupt_train_steps):
    if interrupt_train_steps == -1:
        assert args.interrupt_train_steps <= args.max_train_steps
        final_train_steps = args.max_train_steps if args.interrupt_train_steps < 0 else args.interrupt_train_steps
    else:
        assert interrupt_train_steps <= args.max_train_steps
        final_train_steps = interrupt_train_steps
    return final_train_steps
