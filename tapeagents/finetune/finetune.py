import contextlib
import json
import os
import time
from collections import defaultdict, deque
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
from accelerate.utils import gather_object

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
from .rl import RLConfig, rl_step, make_rl_data_callback
from .rl.utils import get_avg_rl_stats
from .types import DataArgs, DataPartArgs, ModelClass, TrainingMetrics


def load_config(config_name: str, config_dir: str = "../../conf/finetune", output_dir: str = "./output") -> DictConfig:
    with initialize(version_base=None, config_path=config_dir, job_name=config_name):
        config = DictConfig(dict(finetune=compose(config_name=config_name), output_dir=output_dir))
    return config


def get_batch_token_count(batch):
    """Count actual tokens in batch (excluding padding)"""
    attention_mask = batch.get('attention_mask')
    assert attention_mask is not None, "We need attention_mask for accurate token counting"
    return attention_mask.sum().item()


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
        args.effective_batch_size = int(args.train_batch_size) * int(args.gradient_accumulation_passes)
        args.output_dir = str(output_dir)

    if args.gradient_accumulation_passes % num_processes != 0:
        raise ValueError(
            f"Cannot {num_processes}-way parallelize the config with {args.gradient_accumulation_passes} accum passes"
        )
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

    # Add throughput tracking variables
    training_start_time = None
    total_tokens = 0
    total_useful_tokens = 0
    # TODO: Add the vars below to the config
    recent_throughputs = deque(maxlen=100)  # Store last 100 steps
    num_nodes = getattr(args, "num_nodes", 2)  # Default to 2 if not specified

    while training_metrics.completed_steps < final_train_steps:
        training_metrics.epoch = training_metrics.passes // len(train_dataloader)

        # each epoch will have unique but predetermined order of batches. It allows to resume training from exactly the same position
        dataloader_rng.manual_seed(training_metrics.epoch)

        for i, batch in enumerate(train_dataloader):
            if training_start_time is None:
                training_start_time = time.time()

            total_possible_tokens = batch['input_ids'].numel()
            num_tokens = get_batch_token_count(batch)
            if args.resume_dataloader and last_dataloader_position != 0:
                if i < last_dataloader_position:  # rewind train dataloader to the last used position
                    continue
                last_dataloader_position = 0  # the following dataloader loop should start from 0 after rewinding
                logger.info(f"Resumed dataloader from epoch {training_metrics.epoch}, position {i}")

            step_start_time = time.time()
            training_metrics.passes += 1
            training_metrics.samples = training_metrics.passes * samples_per_pass

            if args.cuda_empty_cache:
                torch.cuda.empty_cache()

            do_optimizer_step = training_metrics.passes % args.gradient_accumulation_passes == 0
            
            # Track forward-backward pass time separately
            forward_backward_start = time.time()
            with torch.autocast("cuda"):
                with toggle_sync(do_optimizer_step):
                    loss, this_step_rl_metrics = forward(model, batch)
                    for k, v in this_step_rl_metrics.items():
                        rl_metrics[k].append(v)
                    training_metrics.train_loss = loss.item()
                    training_metrics.lr = optimizer.param_groups[0]["lr"]
                    training_metrics.max_batch_len = max(batch["input_ids"].shape[1], training_metrics.max_batch_len)
                    training_metrics.min_batch_len = min(batch["input_ids"].shape[1], training_metrics.min_batch_len)
                    accelerator.backward(loss / args.gradient_accumulation_passes)
            forward_backward_time = time.time() - forward_backward_start

            # Gather tokens from all workers
            num_tokens_tensor = torch.tensor([num_tokens], device=accelerator.device)
            torch.distributed.all_reduce(num_tokens_tensor, op=torch.distributed.ReduceOp.SUM)
            global_tokens = num_tokens_tensor.item()
            total_possible_tokens = batch['input_ids'].numel() * accelerator.num_processes
            total_gpus = accelerator.num_processes

            # Calculate forward-backward throughput for every step
            pass_tokens_per_gpu_per_second = (total_possible_tokens / total_gpus) / forward_backward_time if forward_backward_time > 0 else 0

            if not do_optimizer_step:
                continue

            # Only calculate full step metrics when we do an optimizer step
            step_total_time = time.time() - step_start_time
            # Scale the tokens by gradient accumulation since we're measuring multiple forward passes
            accumulated_tokens = total_possible_tokens * args.gradient_accumulation_passes
            accumulated_useful_tokens = global_tokens * args.gradient_accumulation_passes
            
            # Calculate padding efficiency
            padding_efficiency = global_tokens / total_possible_tokens if total_possible_tokens > 0 else 0

            # Start timing optimizer step
            optimizer_start_time = time.time()
            
            # All gradients have been accumulated, we can now do an optimizer step
            training_metrics.completed_steps += 1
            if args.gradient_clipping_threshold:
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.gradient_clipping_threshold)
                # grad_norm is None when using DeepSpeed
                training_metrics.grad_norm = grad_norm.item() if grad_norm else -1.0
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            optimizer_time = time.time() - optimizer_start_time
            step_total_time = time.time() - step_start_time

            # Update total tokens only on main process
            if accelerator.is_main_process:
                total_tokens += accumulated_tokens
                total_useful_tokens += accumulated_useful_tokens

            # calculate per-step throughput and update moving average
            step_throughput = global_tokens / step_total_time if step_total_time > 0 else 0
            
            # stabilize metrics by tracking throughput after warmup
            recent_throughputs.append(step_throughput)
            if training_metrics.completed_steps >= args.num_warmup_steps:
                current_throughput = sum(recent_throughputs) / len(recent_throughputs)
            else:
                current_throughput = step_throughput

            # Calculate samples per second
            samples_this_step = samples_per_pass  # This is num_processes * args.train_batch_size
            samples_per_second = samples_this_step / step_total_time if step_total_time > 0 else 0
            samples_per_gpu_second = samples_per_second / total_gpus

            metrics_dict = {}
            time_to_stop = training_metrics.completed_steps >= final_train_steps
            time_to_log = training_metrics.completed_steps % args.log_each_n_steps == 0
            time_to_save = (training_metrics.completed_steps % args.save_checkpoint_steps == 0) or (
                len(args.also_save_steps) and training_metrics.completed_steps in args.also_save_steps
            )
            time_to_save = time_to_save and not time_to_stop
            
            if time_to_log or time_to_save:
                elapsed_time = time.time() - training_start_time
                elapsed_hours = elapsed_time / 3600
                elapsed_days = elapsed_time / (24 * 3600)
                node_days = elapsed_days * num_nodes

                tokens_per_second = total_tokens / elapsed_time
                useful_tokens_per_second = total_useful_tokens / elapsed_time
                tokens_per_gpu_second = tokens_per_second / total_gpus
                useful_tokens_per_gpu_second = useful_tokens_per_second / total_gpus
                tokens_per_node_second = tokens_per_second / num_nodes
                useful_tokens_per_node_second = useful_tokens_per_second / num_nodes
                tokens_per_hour = tokens_per_second * 3600
                tokens_per_day = tokens_per_second * 24 * 3600
                tokens_per_day_per_node = tokens_per_day / num_nodes
                tokens_per_day_per_gpu = tokens_per_day / total_gpus

                optimizer_step_throughput = tokens_per_second * args.gradient_accumulation_passes

                metrics_dict.update({
                    # throughput metrics
                    "throughput/tokens_per_second": tokens_per_second,
                    "throughput/useful_tokens_per_second": useful_tokens_per_second,
                    "throughput/tokens_per_hour_M": tokens_per_hour / 1e6,
                    "throughput/tokens_per_day_B": tokens_per_day / 1e9,
                    "throughput/tokens_per_day_per_node_M": tokens_per_day_per_node / 1e6,
                    "throughput/tokens_per_day_per_gpu_M": tokens_per_day_per_gpu / 1e6,
                    "throughput/tokens_per_gpu_second": tokens_per_gpu_second,
                    "throughput/tokens_per_node_second": tokens_per_node_second,
                    "throughput/useful_tokens_per_gpu_second": useful_tokens_per_gpu_second,
                    "throughput/useful_tokens_per_node_second": useful_tokens_per_node_second,
                    "throughput/total_tokens_B": total_tokens / 1e9,
                    "throughput/total_useful_tokens_B": total_useful_tokens / 1e9,
                    "throughput/optimizer_step_throughput": optimizer_step_throughput,
                    "throughput/padding_efficiency": padding_efficiency,
                    "throughput/pass_tokens_per_gpu_per_second": pass_tokens_per_gpu_per_second,
                    "throughput/samples_per_second": samples_per_second,
                    "throughput/samples_per_gpu_second": samples_per_gpu_second,
                    "throughput/current_tokens_per_second": current_throughput,
                    "throughput/raw_step_tokens_per_second": step_throughput,
                    
                    # time metrics
                    "time/total_hours": elapsed_hours,
                    "time/total_nodedays": node_days,
                    "time/elapsed_days": elapsed_days,
                    "time/step_total_seconds": step_total_time,
                    "time/step_training_seconds": optimizer_time,
                    "time/tokens_per_step": global_tokens,
                    "time/tokens_per_second_this_step": global_tokens / step_total_time if step_total_time > 0 else 0,
                    "time/forward_backward_ratio": forward_backward_time / step_total_time if step_total_time > 0 else 0,
                    "time/optimizer_ratio": optimizer_time / step_total_time if step_total_time > 0 else 0,
                    "time/step_idle": step_total_time - (forward_backward_time + optimizer_time),
                    "time/forward_backward_seconds": forward_backward_time,

                    # batch metrics
                    "batch/avg_sequence_length": batch['input_ids'].shape[1],
                    "batch/total_sequences": batch['input_ids'].shape[0] * accelerator.num_processes,
                    "batch/global_batch_size": args.train_batch_size * args.gradient_accumulation_passes * accelerator.num_processes,
                })

            # Add memory metrics if using GPU
            if torch.cuda.is_available():
                metrics_dict.update({
                    "memory/gpu_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                    "memory/gpu_reserved_gb": torch.cuda.memory_reserved() / 1e9,
                    "memory/gpu_max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                    "memory/gpu_peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9
                })

            if time_to_save:
                # Overwrite latest model at pytorch_model.bin (for later JGA evaluation *and* for resuming training)
                # conversion to HF format is done only for DeepSpeed models
                save_model_and_tokenizer(
                    current_dir,
                    model,
                    tokenizer,
                    args.lora.enabled,
                    safe_serialization=args.use_safetensors,
                )
                # Save training state to training_state.pt (for resuming)
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
    if args.cuda_empty_cache:
        torch.cuda.empty_cache()
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
