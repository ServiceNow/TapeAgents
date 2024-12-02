import contextlib
import os
import shutil
import typing
from pathlib import Path
from typing import Any, Type

import torch
import transformers
from packaging import version
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.models.auto.modeling_auto import _BaseAutoModelClass

from .context import accelerator, logger
from .lora import has_lora_checkpoint, lora_load, lora_save, prepare_lora_model
from .types import ModelClass, TrainingMetrics


def get_auto_model_class(
    model_class: ModelClass,
) -> Type[_BaseAutoModelClass]:
    """Get the AutoModel class corresponding to the model class."""
    match model_class:
        case "causal-language-modeling":
            return AutoModelForCausalLM
        case "seq2seq-language-modeling":
            return AutoModelForSeq2SeqLM
        case _:
            raise ValueError(f"Unsupported model class: {model_class}")


def load_tokenizer(config_name):
    tokenizer = AutoTokenizer.from_pretrained(config_name, use_fast=True)
    if not isinstance(tokenizer, transformers.PreTrainedTokenizerFast):
        raise ValueError(f"tokenizer {tokenizer} is not fast")
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if isinstance(tokenizer, transformers.T5TokenizerFast):
        tokenizer.add_special_tokens({"additional_special_tokens": ["<n>", "<t>"]})  # type: ignore
        tokenizer.add_tokens(new_tokens=["▁{", "{", "▁}", "}"])
    return tokenizer


def load_model(args, model_class, current_dir):
    accelerator.wait_for_everyone()

    assert not (
        os.path.exists(current_dir / "pytorch_model.bin")
        and os.path.exists(current_dir / "pytorch_model.bin.index.json")
    ), (
        "Found pytorch_model.bin AND pytorch_model.bin.index.json in {current_dir}! "
        "Please remove one of them. "
        "This may happen if combining deepspeed and non-deepsped training"
    )

    model_to_load = args.config_name
    loading_args: dict[str, Any] = dict(
        use_safetensors=args.use_safetensors,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,  # this is essential for quick model loading as it does not spend time on a random weights initialization. It cuts loading time of a 15B params model from 100 sec to 12 sec.
    )
    if "use_flash_attention" in args.keys() and args.use_flash_attention:
        assert version.parse(transformers.__version__) >= version.parse("4.34.0"), (
            "flash_attention is only supported for transformers>=4.34.0. " "Please upgrade transformers to use it"
        )
        loading_args["use_flash_attention_2"] = args.use_flash_attention

    is_ds_zero_3 = False
    if getattr(accelerator.state, "deepspeed_plugin", None):
        del loading_args["low_cpu_mem_usage"]  # deepspeed is not compatible with this option
        is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3  # type: ignore

    if args.load_as_bf16:
        loading_args["torch_dtype"] = torch.bfloat16
    if args.lora.enabled:
        if is_ds_zero_3:
            raise Exception("LoRA is not compatible with Deepspeed zero stage 3")
        if args.lora.base_model_8bit:
            loading_args["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=False,
                load_in_8bit=True,
                llm_int8_has_fp16_weight=args.load_as_bf16,
            )
        elif args.lora.base_model_4bit:
            loading_args["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                load_in_8bit=False,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=False,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
    if args.auto_device_map:
        loading_args["device_map"] = "auto"
    model_cls = get_auto_model_class(model_class)
    if (
        os.path.exists(current_dir / "pytorch_model.bin")
        or os.path.exists(current_dir / "model.safetensors")
        or os.path.exists(current_dir / "pytorch_model.bin.index.json")
        or os.path.exists(current_dir / "model.safetensors.index.json")
    ):  # resume
        # Size mismatch errors here may be due to improper used of Deepspeed+save_pretrained()
        # instead, always call save_model_only() in all processes

        # when LoRA enabled, always preload the original model, the lora weights will be loaded later
        model_to_load = args.config_name if args.lora.enabled else str(current_dir)
        logger.info(f"Loading model {model_cls} weights from {current_dir}")
    else:  # from scratch
        logger.info(f"Initializing model {model_cls} from {args.config_name}")

    logger.info(f"Loading args: {loading_args}")
    model = model_cls.from_pretrained(model_to_load, **loading_args)

    if args.lora.enabled:
        model = prepare_lora_model(args.lora, model, args.gradient_checkpointing)
        if has_lora_checkpoint(current_dir):
            lora_load(current_dir, model)
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})

    accelerator.wait_for_everyone()
    return model


def save_training_state(
    training_state_dir: Path,
    model: transformers.PreTrainedModel,
    optimizer,
    lr_scheduler,
    extra_training_state: dict[str, typing.Any] = {},
):
    with get_temporary_folder_and_move(training_state_dir) as temp_dir:
        _save_training_state(temp_dir, model, optimizer, lr_scheduler, extra_training_state)


def _save_training_state(
    training_state_dir: Path,
    model: transformers.PreTrainedModel,
    optimizer,
    lr_scheduler,
    extra_training_state: dict[str, typing.Any] = {},
):
    """
    Checkpoint model, optimizer, lr_scheduler, scaler (?) and extra_training_state

    extra_training_state: a dictionary containing completed_steps, passes, best_loss, best_perplexity, etc.

    Compatible with and without deepspeed.
    Make sure to call on *all* accelerate processes.

    Modified from https://github.com/huggingface/accelerate/blob/main/examples/by_feature/deepspeed_with_config_support.py#L247
    More details at: https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html
    """
    assert (
        not os.path.exists(training_state_dir) or training_state_dir.is_dir()
    ), f"output_dir {training_state_dir} must be a directory"

    if model.__class__.__name__.endswith("DeepSpeedEngine"):
        # Save both model and optimizer, as well as lr_scheduler if supported by deepspeed
        logger.info("Save deepspeed training state")
        client_state = dict(extra_training_state)
        if model.lr_scheduler is None:  # lr_scheduler not handled by deepspeed
            logger.warning(
                f"Manually adding DeepSpeed-unsupported lr_scheduler of type {type(lr_scheduler).__name__} to the checkpoint"
            )
            client_state["lr_scheduler_state"] = lr_scheduler.state_dict()  # manually save state
        success = model.save_checkpoint(training_state_dir, tag="deepspeed", client_state=client_state)
        assert success, f"Failed to save deepspeed training state into {training_state_dir}"
        logger.info(f"Saved deepspeed training state to {training_state_dir}")
    else:  # multi_gpu mode (no deepspeed)
        # Only save training_state in main process
        logger.info("Save accelerate training state")
        if accelerator.is_main_process:
            training_state = dict(extra_training_state)
            training_state["optimizer_state"] = optimizer.state_dict()
            training_state["lr_scheduler_state"] = lr_scheduler.state_dict()
            accelerator.save(training_state, training_state_dir / "training_state.pt")
            logger.info(f"Saved accelerate training state to {training_state_dir}")


def load_training_checkpoint(
    training_state_dir: Path,
    model: transformers.PreTrainedModel,
    optimizer,
    lr_scheduler,
):
    """
    Load checkpoint created by save_training_checkpoint() in-place:

    - With deepspeed, this will load model, optimizer, lr_scheduler states in-place.
    - Without deepspeed, this will *only* load optimizer, lr_scheduler states in-place,
        but *not* model states!
    """
    assert (
        not os.path.exists(training_state_dir) or training_state_dir.is_dir()
    ), f"output_dir {training_state_dir} must be a directory"

    if model.__class__.__name__.endswith("DeepSpeedEngine"):
        logger.info("Load deepspeed training state")
        # This magically loads optimizer and lr_scheduler states (if they were saved)
        # (the passed optimizer and lr_scheduler arguments will be ignored)
        load_path, extra_training_state = model.load_checkpoint(
            training_state_dir,
            tag="deepspeed",
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
        )
        if load_path is None:
            raise RuntimeError(f"Loading deepspeed checkpoint from {training_state_dir} failed")
        if (
            model.lr_scheduler is None
            and extra_training_state is not None
            and "lr_scheduler_state" in extra_training_state
        ):
            # Manually load lr_scheduler states
            logger.warning(f"Manually loading ds-unsupported lr_scheduler of type {type(lr_scheduler).__name__}")
            lr_scheduler.load_state_dict(extra_training_state["lr_scheduler_state"])
        logger.info(f"Loaded deepspeed checkpoint from {training_state_dir}")
    else:  # multi_gpu (no deepspeed)
        # This needs to be called from all processes
        training_state = torch.load(training_state_dir / "training_state.pt", map_location="cpu")
        optimizer.load_state_dict(training_state["optimizer_state"])
        lr_scheduler.load_state_dict(training_state["lr_scheduler_state"])
        del training_state["optimizer_state"]
        del training_state["lr_scheduler_state"]
        extra_training_state = training_state
        logger.info(f"Loaded accelerate checkpoint from {training_state_dir}")
    return extra_training_state


@contextlib.contextmanager
def get_temporary_folder_and_move(output_dir: Path):
    """
    Context manager safe checkpointing.

    Creates temporary folder `~output_dir`, then rename to final destination
    """
    if os.path.exists(output_dir) and not os.path.isdir(output_dir):
        raise ValueError("get_temporary_folder_and_move: output_dir is not a directory")

    output_dir = output_dir.resolve()
    temporary_path = output_dir.parent / ("~" + output_dir.name)

    if accelerator.is_main_process:
        if os.path.exists(temporary_path):
            logger.info(f"Deleting temporary directory {temporary_path}")
            shutil.rmtree(temporary_path)
        logger.info(f"Creating temporary directory {temporary_path}")
        os.makedirs(temporary_path)

    accelerator.wait_for_everyone()
    yield temporary_path
    accelerator.wait_for_everyone()

    # Move to final path
    if accelerator.is_main_process:
        # delete output_dir if it exists
        if os.path.exists(output_dir):
            logger.info(
                f" -> Deleting {output_dir}. "
                f"If this fails, manually delete it and move {temporary_path} to {output_dir}"
            )
            shutil.rmtree(output_dir)
        logger.info(f" -> Renaming {temporary_path} to {output_dir}")
        os.rename(temporary_path, output_dir)
        logger.info(f"Done moving files to {output_dir}")


def save_model_and_tokenizer(
    output_dir: Path,
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
    lora: bool = False,
    safe_serialization: bool = False,
):
    logger.info("Saving model and tokenizer")
    with get_temporary_folder_and_move(output_dir) as temp_dir:
        save_model_only(
            temp_dir,
            model,
            unwrap=True,
            lora=lora,
            safe_serialization=safe_serialization,
        )
        save_tokenizer_only(temp_dir, tokenizer)


def save_model_only(
    output_dir: Path,
    model: transformers.PreTrainedModel,
    unwrap: bool = True,
    lora: bool = False,
    safe_serialization: bool = False,
):
    """
    Save model weights and config.

    Creates the following files in output_dir/ :
        - config.json
    and either:
        - pytorch_model.bin (single-file model), OR
        - pytorch_model-XXXXX-of-XXXXX.bin (multi-file model) and pytorch_model.bin.index.json

    Note that this does not save optimizer, lr_scheduler, scaler, etc.
    Use only for later JGA evaluation, not for resuming training

    Must be called on *all* accelerate processes because all of them must save their shards.
    """
    assert not os.path.exists(output_dir) or output_dir.is_dir(), f"output_dir {output_dir} must be a directory"
    accelerator.wait_for_everyone()

    logger.info(f"Save model to {output_dir}")

    unwrapped_model = accelerator.unwrap_model(model) if unwrap else model
    if lora:
        lora_save(output_dir, unwrapped_model)
        return

    if unwrapped_model.__class__.__name__.endswith("DeepSpeedEngine"):
        unwrapped_model.save_checkpoint(
            save_dir=output_dir,
        )
        logger.info(f"Saved deepspeed checkpoint to {output_dir}")
    elif isinstance(unwrapped_model, transformers.PreTrainedModel):
        unwrapped_model.save_pretrained(  # type: ignore
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
            safe_serialization=safe_serialization,
        )
        logger.info(f"Saved model to {output_dir}")
    else:
        raise ValueError(f"model is neither a deepspeed model nor a transformers.PreTrainedModel: {type(model)}")


def save_tokenizer_only(
    output_dir: Path,
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
):
    """
    Save only tokenizer to output_dir

    Can be called on *all* processes.
    """
    assert not os.path.exists(output_dir) or output_dir.is_dir(), f"output_dir {output_dir} must be a directory"
    if accelerator.is_main_process:
        logger.info(f"Save tokenizer to {output_dir}")
        tokenizer.save_pretrained(output_dir)


def remove_results(current_dir, intermediate_root_dir, training_state_dir, log_dir):
    logger.info("Cleaning up checkpoints and training state")
    if os.path.exists(current_dir):
        shutil.rmtree(current_dir)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    if os.path.exists(intermediate_root_dir):
        shutil.rmtree(intermediate_root_dir)
    if os.path.exists(training_state_dir):
        if os.path.isdir(training_state_dir):
            shutil.rmtree(training_state_dir)


def load_training_state(
    training_state_dir,
    model,
    optimizer,
    lr_scheduler,
    training_metrics: TrainingMetrics,
):
    accelerator.wait_for_everyone()
    training_state = load_training_checkpoint(training_state_dir, model, optimizer, lr_scheduler)
    if training_state is None:
        raise ValueError(f"Could not load training state from {training_state_dir}")
    training_metrics.passes = training_state["passes"]
    training_metrics.completed_steps = training_state["completed_steps"]
    training_metrics.best_eval_loss = training_state["best_eval_loss"]
    training_metrics.best_completed_steps = training_state["best_completed_steps"]
    return training_metrics
