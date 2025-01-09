import logging
logger = logging.getLogger(__name__)
import contextlib
from functools import partial
import json
import os
from queue import Empty, Queue
import sys
import threading
import time
from collections import defaultdict
from datasets.arrow_dataset import Dataset
from datasets.fingerprint import Hasher
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Generator
import numpy as np
from tokenizers import Tokenizer
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data.dataloader import DataLoader
from transformers import (
    get_scheduler,
    set_seed,
)
from accelerate import Accelerator
import transformers

from tapeagents.core import TrainingText

from .checkpoints import (
    load_model,
    load_tokenizer,
    load_training_state,
    remove_results,
    save_model_and_tokenizer,
    save_training_state,
)
from .context import accelerator
from .data import collate, preprocess_fn, read_jsonl_stream
from .eval import evaluate_and_get_metrics
from .logging_ import log_metrics, log_time, setup_logging
from .optim import get_optimizer
from .rl import RL_DATA_COLUMNS, RLConfig, populate_rl_data, rl_step, make_rl_data_callback
from .rl.utils import get_avg_rl_stats
from .types import DataArgs, DataPartArgs, ModelClass, TrainingMetrics


def load_config(config_name: str, config_dir: str = "../../conf/finetune", output_dir: str = "./output") -> DictConfig:
    with initialize(version_base=None, config_path=config_dir, job_name=config_name):
        config = DictConfig(dict(finetune=compose(config_name=config_name), output_dir=output_dir))
    return config


def run_dataset_loader(
    dataset_queue: Queue, 
    data_channel: str,
    start_shard: int,
    preprocess_chunk_size: int, 
    preprocess_dataset_fn: Callable[[list[dict]], Dataset],
):
    """Incrementally load shards to populate the dataset queue."""
    shard_idx = start_shard
    while True: 
        try:
            if dataset_queue.full():
                time.sleep(1)
                continue

            shard_path = f"{data_channel}/{shard_idx}.jsonl"
            buffer = []
            for entry in read_jsonl_stream(shard_path):
                buffer.append(entry)
                if len(buffer) == preprocess_chunk_size:
                    dataset_queue.put(preprocess_dataset_fn(buffer))
                    logger.info(f"Processed {len(buffer)} samples from {shard_path} and added to the queue")
                    buffer = []
        except Exception as e:
            logger.error(f"Error in dataset loader: {e}")
            dataset_queue.put(e)
            break


def preprocess_dataset(
    data: list[dict],
    tokenizer: transformers.PreTrainedTokenizerBase,
    seq_length: int,
    rl_config: RLConfig,
) -> Dataset:
    preprocess = partial(preprocess_fn, seq_length=seq_length, tokenizer=tokenizer, is_rl=True)
    columns = ["input_ids", "labels", "attention_mask"] + RL_DATA_COLUMNS
    logger.debug(f"Instantiated preprocess function hash {Hasher.hash(preprocess)}")

    dataset = Dataset.from_list(data)
    logger.debug(f"Raw data part size: {dataset.num_rows}")
    logger.debug(f"Raw data part fingerprint: {dataset._fingerprint}")
    dataset = dataset.map(preprocess, keep_in_memory=True, load_from_cache_file=False)
    dataset = dataset.with_format(columns=columns)
    logger.debug(f"Preprocessed data part fingerprint: {dataset._fingerprint}")

    return populate_rl_data(dataset=dataset, config=rl_config)


def run_finetuning_loop(
    cfg: DictConfig,
):
    dt = time.perf_counter()
    time_stats = {}

    exp_root_dir = Path(cfg.output_dir)
    output_dir = Path(cfg.finetune.output_dir)
    num_processes = accelerator.state.num_processes  # type: ignore
    args = cfg.finetune if "finetune" in cfg else cfg

    model_class: ModelClass = args.model_class

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
    set_seed(args.seed)

    # using a subfolder makes "safe" overwriting possible
    current_dir = output_dir / "current"
    intermediate_root_dir = output_dir / "intermediate"
    training_state_dir = output_dir / "training_state"
    log_dir = output_dir / "logs"

    if args.force_restart and accelerator.is_main_process:
        remove_results(current_dir, intermediate_root_dir, training_state_dir, log_dir)

    # Logging
    setup_logging(cfg, output_dir)
    logger.info(accelerator.state)
    logger.info(f"Saving experiment to {output_dir}")
    dt = log_time(dt, time_stats, "finetune/startup")

    tokenizer = load_tokenizer(args.config_name)
    model = load_model(args, model_class, current_dir)

    dt = log_time(dt, time_stats, "finetune/model_load")

    # TODO: different processes read from different source replicas
    source_replica = 0
    data_channel = exp_root_dir / "channels" / Path(args.input) / str(source_replica) / str(accelerator.state.process_index)
    # TODO: calculate the starting shard based on the training state
    start_shard = 0

    rl_config = RLConfig(**args.rl)
    preprocess_dataset_fn = partial(
        preprocess_dataset,
        tokenizer=tokenizer,
        seq_length=args.seq_length,
        rl_config=rl_config,
    )
    dataset_queue = Queue(maxsize=3)
    dataset_loader_worker_fn = partial(
        run_dataset_loader,
        dataset_queue=dataset_queue,
        data_channel=data_channel,
        preprocess_chunk_size=64,
        preprocess_dataset_fn=preprocess_dataset_fn,
        start_shard=start_shard,
    )
    # Start the dataset loader thread using Thread
    dataset_loader_thread = threading.Thread(target=dataset_loader_worker_fn)
    dataset_loader_thread.start()
    collate_fn = partial(
        collate,
        tokenizer=tokenizer,
    )
    logger.info(f"Instantiated collate_fn hash {Hasher.hash(collate_fn)}")
    def batch_generator_fn():
        while True:
            try:
                dataset_or_exc = dataset_queue.get(timeout=5.0)
            except Empty as e:
                logger.info("Dataset queue is empty, waiting for more data...")
                continue
            if isinstance(dataset_or_exc, Exception):
                raise dataset_or_exc
            assert isinstance(dataset_or_exc, Dataset)
            dataloader = DataLoader(
                dataset_or_exc,
                batch_size=args.train_batch_size,
                collate_fn=collate_fn,
                shuffle=False,
            )
            for batch in dataloader:
                # Put all tensors in batch on GPU
                yield {k: v.to(accelerator.device) for k, v in batch.items()}

    optimizer = get_optimizer(args.optim, model, args.learning_rate, args.weight_decay)
    lr_scheduler = get_scheduler(args.lr_scheduler_type, optimizer, args.num_warmup_steps, args.max_train_steps)

    # Wrap everything with HF Accelerator
    (
        model,
        optimizer,
        lr_scheduler,
    ) = accelerator.prepare(model, optimizer, lr_scheduler)
    logger.info("Model, optimizer and lr_scheduler prepared")

    training_metrics = TrainingMetrics()
    if os.path.exists(training_state_dir):
        # WARNING: In case of deepspeed this will overwrite model weights too
        training_metrics = load_training_state(training_state_dir, model, optimizer, lr_scheduler, training_metrics)
        training_metrics.lr = optimizer.param_groups[0]["lr"]
        logger.info("LR after loading training state: %.2E" % training_metrics.lr)
        dt = log_time(dt, "finetune/training_state_load")

    logger.info("Start training")
    accelerator.wait_for_everyone()
    model.train()    
    rl_finetuning_worker(
        args,
        accelerator,
        model,
        optimizer,
        lr_scheduler,
        tokenizer,
        training_metrics,
        batch_generator_fn(),
    )
        

def rl_finetuning_worker(
    args: DictConfig,
    accelerator: Accelerator,
    # model, optimizer and scheduler can be of different types depending on what Accelerate backend we use (DeepSpeed vs FSDP)
    model: Any, 
    optimizer: Any,
    lr_scheduler: Any,
    tokenizer: Tokenizer, 
    training_metrics: TrainingMetrics,
    data_generator: Generator[dict, None, None],
):
    dt = time.perf_counter()

    output_dir = Path(args.output_dir)
    current_dir = output_dir / "current"
    intermediate_root_dir = output_dir / "intermediate"
    training_state_dir = output_dir / "training_state"
    
    rl_config = RLConfig(**args.rl)
    samples_per_pass = accelerator.state.num_processes * args.train_batch_size
    final_train_steps = calculate_train_steps(args, args.interrupt_train_steps)
    if training_metrics.completed_steps == final_train_steps:
        logger.info("Training is already completed")
        return

    while training_metrics.completed_steps < final_train_steps:
        batch = next(data_generator)
        num_tokens = batch["input_ids"].numel()

        time_before = time.time()
        training_metrics.passes += 1
        training_metrics.samples = training_metrics.passes * samples_per_pass
        rl_metrics = defaultdict(list)
        time_stats = {}

        if args.cuda_empty_cache:
            torch.cuda.empty_cache()

        do_optimizer_step = training_metrics.passes % args.gradient_accumulation_passes == 0
        @contextlib.contextmanager
        def toggle_sync(sync: bool):
            """Wrap accelerate.no_sync() if sync is False."""
            if sync:
                yield  # do not enforce no_sync mode
            else:
                with accelerator.no_sync(model):
                    yield
        with torch.autocast("cuda"):
            with toggle_sync(do_optimizer_step):
                loss, this_step_rl_metrics = rl_step(model, batch, rl_config)
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
            dt = log_time(dt, time_stats, "finetune/interim_eval")
            metrics_dict.update(
                {
                    "stats/lr": training_metrics.lr,
                    "stats/grad_norm": training_metrics.grad_norm,
                    "stats/samples": training_metrics.passes * samples_per_pass,
                    "stats/passes": training_metrics.passes,
                    "stats/completed_steps": training_metrics.completed_steps,
                    "stats/epoch": training_metrics.epoch,
                    "throughput/tokens_perGPU_per_sec": num_tokens / step_took,
                    "throughput/tokens_per_sec": num_tokens * accelerator.state.num_processes / step_took,
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

            if args.keep_intermediate_checkpoints:
                intermediate_dir = intermediate_root_dir / str(training_metrics.completed_steps)
                save_model_and_tokenizer(
                    intermediate_dir,
                    model,
                    tokenizer,
                    args.lora.enabled,
                    safe_serialization=args.use_safetensors,
                )
                dt = log_time(dt, time_stats, "finetune/interim_save")

                if args.cuda_empty_cache:
                    torch.cuda.empty_cache()

            accelerator.wait_for_everyone()  # wait for the main process that saves the model
    
        if len(metrics_dict):
            log_metrics(logger, training_metrics.completed_steps, metrics_dict)

        if training_metrics.completed_steps >= final_train_steps:
            logger.info(f"Reached final step {final_train_steps}, stopping.")
            break
    dt = log_time(dt, time_stats, "finetune/train_loop")

    logger.info("Final model saving")
    save_model_and_tokenizer(
        current_dir,
        model,
        tokenizer,
        args.lora.enabled,
        safe_serialization=args.use_safetensors,
    )
    dt = log_time(dt, time_stats, "finetune/final_save")
    if args.save_final_training_state:
        save_training_state(training_state_dir, model, optimizer, lr_scheduler, asdict(training_metrics))
        dt = log_time(dt, "finetune/final_training_state_save")

    if accelerator.is_main_process:
        with open(output_dir / "summary.json", "w") as wf:
            json.dump(asdict(training_metrics), wf, indent=4, sort_keys=True)
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
