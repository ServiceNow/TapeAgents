import logging
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer, BatchEncoding

# FIXME: remove a warnings, but might be worth investigating
os.environ["TOKENIZERS_PARALLELISM"] = "false"


logger = logging.getLogger(__name__)

RL_DATA_COLUMNS = [
    "rewards",
    "advantages",
    "old_logprobs",
    "ref_logprobs",
]


@dataclass
class StepConfig(object):
    def to_dict(self):
        output_dict = {}
        for key, value in self.__dict__.items():
            output_dict[key] = value
        return flatten_dict(output_dict)


@dataclass
class GRPOConfig(StepConfig):
    use_advantages: Optional[bool] = field(
        default=True,
        metadata={"help": "Use advantages instead of rewards to compute the loss"},
    )
    epsilon: Optional[float] = field(default=0.2, metadata={"help": "Clip parameter for the ration of log probs"})
    kl_coef: Optional[float] = field(
        default=0.1,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )
    ratio_threshold: Optional[float] = field(
        default=10.1,
        metadata={"help": "Skip mini-batches with high PPO ratios that can cause loss spike"},
    )


def masked_sum(values: torch.Tensor, mask: torch.Tensor, axis: Optional[bool] = None) -> torch.Tensor:
    """Compute sum of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis)
    else:
        return (values * mask).sum()


def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis: Optional[bool] = None) -> torch.Tensor:
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()


def make_rl_data_callback(args, current_dir, ppo_config, model):
    if ppo_config:
        populate_rl_data_ = partial(
            populate_rl_data,
            config=ppo_config,
        )
    else:
        populate_rl_data_ = None
    return populate_rl_data_


def flatten_dict(nested: Dict, sep: str = "/") -> Dict:
    """Flatten dictionary and concatenate nested keys with separator."""

    def recurse(nest: Dict, prefix: str, into: Dict) -> None:
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, Mapping):
                recurse(v, prefix + k + sep, into)
            else:
                into[prefix + k] = v

    flat = {}
    recurse(nested, "", flat)
    return flat


def grpo_step(model, batch, config: GRPOConfig) -> tuple[torch.Tensor, dict[str, float]]:
    """
    GRPO is based on https://arxiv.org/pdf/2402.03300
    model: model that is updated

    """
    rewards = batch.pop("rewards")[:, 1:]
    advantages = batch.pop("advantages")[:, 1:]
    masks = batch["labels"] != -100
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )

    new_log_probs = torch.gather(
        F.log_softmax(outputs.logits[:, :-1, :], dim=-1),  # the last log probs has no target
        dim=2,
        index=batch["input_ids"][:, 1:].unsqueeze(2),
    ).squeeze(2)

    masks_ = masks[:, 1:]
    ref_logprobs = batch["ref_logprobs"][:, 1:]
    old_logprobs = batch["old_logprobs"][:, 1:]
    # assert torch.isfinite(rewards).all(), "rewards contains NaN or inf"
    # assert torch.isfinite(advantages).all(), "advantages contains NaN or inf"
    # assert torch.isfinite(ref_logprobs).all(), "ref_logprobs contains NaN or inf"
    # assert torch.isfinite(old_logprobs).all(), "old_logprobs contains NaN or inf"
    # assert torch.isfinite(new_log_probs).all(), "new_log_probs contains NaN or inf"

    assert new_log_probs.shape == ref_logprobs.shape

    # First compute the PPO surrogate loss, see https://arxiv.org/pdf/2402.03300 eq 3
    log_ratio_new_old = new_log_probs - old_logprobs
    log_ratio_new_old = new_log_probs - old_logprobs
    ratio_new_old = torch.exp(log_ratio_new_old)
    weights = advantages if config.use_advantages else rewards
    # assert torch.isfinite(log_ratio_new_old).all(), "log_ratio_new_old contains NaN or inf"
    # assert torch.isfinite(ratio_new_old).all(), "ratio_new_old contains NaN or inf"
    # assert torch.isfinite(weights).all(), "weights contains NaN or inf"
    # assert ratio_new_old.shape == weights.shape
    surr1 = ratio_new_old * weights
    # assert torch.isfinite(surr1).all(), "surr1 contains NaN or inf"

    clamped_ratio = torch.clamp(ratio_new_old, 1 - config.epsilon, 1 + config.epsilon)
    # assert torch.isfinite(clamped_ratio).all(), "clamped_ratio contains NaN or inf"

    surr2 = clamped_ratio * weights
    # assert torch.isfinite(surr2).all(), "surr2 contains NaN or inf"

    surrogate_loss = torch.min(surr1, surr2)
    # assert torch.isfinite(surrogate_loss).all(), "surrogate_loss contains NaN or inf"

    # Second compute the approximated KL, see https://arxiv.org/pdf/2402.03300 eq 4
    log_ratio_ref_new = ref_logprobs - new_log_probs
    approx_kl = torch.exp(log_ratio_ref_new) - log_ratio_ref_new - 1  # Schulman KL approx
    assert approx_kl.shape == masks_.shape
    assert approx_kl.shape == surrogate_loss.shape
    # assert torch.isfinite(surrogate_loss).all(), "surrogate_loss contains NaN or inf"
    # assert torch.isfinite(approx_kl).all(), "approx_kl contains NaN or inf"
    loss = -masked_mean(surrogate_loss - config.kl_coef * approx_kl, masks_)
    assert torch.isfinite(loss).all(), "loss contains NaN or inf"
    # if not torch.isfinite(loss):
    #    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

    if (
        masked_mean(ratio_new_old, masks_) > config.ratio_threshold
    ):  # https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1236
        loss = loss * 0

    stats = {
        "max_new_log_probs": (masks_ * new_log_probs).max().item(),
        "max_ratio_new_old": (masks_ * ratio_new_old).max().item(),
        "max_loss": loss.max().item(),
        "reward": masked_mean(rewards, masks_).item(),
        "max_reward": (rewards * masks_).max().item(),
        "min_reward": (rewards * masks_).min().item(),
        "mean_old_logprobs": masked_mean(old_logprobs, masks_).item(),
        "mean_new_logprobs": masked_mean(new_log_probs, masks_).item(),
        "mean_ref_logprobs": masked_mean(ref_logprobs, masks_).item(),
        "advantage": masked_mean(advantages, masks_).item(),
        "max_advantage": advantages.max().item(),
        "min_advantage": advantages.min().item(),
        "loss": loss.item(),
        "kl": masked_mean(approx_kl, masks_).item(),
        "max_kl": (masks_ * approx_kl).max().item(),
        "min_kl": (masks_ * approx_kl).min().item(),
        "surr1": masked_mean(surr1, masks_).item(),
        "surr2": masked_mean(surr2, masks_).item(),
        "ratio_new_old": masked_mean(ratio_new_old, masks_).item(),
        "ratio_ref_new": masked_mean(torch.exp(log_ratio_ref_new), masks_).item(),
        "ratio_ref_old": masked_mean(torch.exp(ref_logprobs - old_logprobs), masks_).item(),
    }
    return loss, stats


def replace_non_zero(reward_list, value):
    """
    Replace the non-zero entry in the reward list with the given value.
    If all entries are zero, return the original list.
    """
    for i, r in enumerate(reward_list):
        if r != 0:
            new_list = reward_list.copy()
            new_list[i] = value
            return new_list
    return reward_list  # Return original list if all entries are zero


def update_advantages(dataset: Dataset, config: GRPOConfig) -> Dataset:
    """
    Updates the advantages column in the given dataset based on reward statistics.

    Args:
        dataset (Dataset): The input dataset containing rewards and placeholder advantages.

    Returns:
        Dataset: The updated dataset with the updated advantages column.

    """
    df = dataset.to_pandas()

    # Group by fork_id and compute mean and std of reward
    # new_df = expand_rewards_column(df, "rewards")
    df["reward"] = df["rewards"].apply(np.mean)
    grouped = df.groupby("fork_id")["reward"].agg(["mean", "std", "count"]).reset_index()

    # Rename columns for clarity
    grouped.columns = ["fork_id", "reward_mean", "reward_std", "count"]

    # Merge the computed statistics back to the original dataset
    df_with_stats = pd.merge(df, grouped, on="fork_id", how="left")

    def calculate_advantage(row):
        rewards = row["rewards"]
        mean = row["reward_mean"]
        std = row["reward_std"]
        return [(reward - mean) / (np.nan_to_num(std) + 1e-4) for reward in rewards]

    df_with_stats["advantages"] = df_with_stats.apply(calculate_advantage, axis=1)

    # replace advantages entry
    dataset = replace_dataset_column(dataset, "advantages", df_with_stats["advantages"].tolist())

    # Convert back to a Hugging Face Dataset
    return dataset


def populate_rl_data(
    dataset: Dataset,
    columns: list[str],
    collate_fn: Callable,
    config: GRPOConfig,
) -> Dataset:
    """
    Prepares the dataset for RL by performing forward passes and updating the dataset.

    Args:
        dataset (Dataset): Rollout dataset
        config (StepConfig): The configuration settings for the RL config.

    Returns:
        Dataset: The prepared dataset with added columns for advantages and updated advantages.

    Note:
        The function performs SFT and RL forward passes, updates the dataset, and calculates advantages for RL. If the
        RL weights are different from the SFT weights, it loads the RL weights before the RL forward passes. The function
        then drops unnecessary columns from the dataset and adds a new column for advantages.
    """
    logger.info("Populate RL Data")

    # TODO: make it an option if using advantage or reward
    dataset = update_advantages(dataset, config)

    logger.info("Finish Populate RL Data")
    return dataset


def replace_dataset_column(dataset: Dataset, column_name: str, new_column: List[List[float]]) -> Dataset:
    """
    Replace a column in the dataset with a new column.
    """
    if column_name in dataset.features:
        dataset = dataset.map(remove_columns=[column_name])
    dataset = dataset.add_column(name=column_name, column=new_column)  # type: ignore

    return dataset


def prepare_rl_fields(
    encoding: BatchEncoding,
    rewards_per_line: list[float],
    old_logprobs: list[float],
    ref_logprobs: list[float],
    spans: list[tuple[int, int]],
    seq_length: int,
    tokenizer: AutoTokenizer,
) -> BatchEncoding:
    """
    Convert reward per lines to reward per token and add returns and advantages placeholders

    inputs:
        encoding: BatchEncoding
        rewards_per_line: list of rewards per line
        spans: list of spans
        seq_length: length of the sequence

    outputs:
        encoding: BatchEncoding with rewards per token and returns and advantages placeholders

    """
    target_tokens = [token for token in encoding["labels"] if token != -100]
    assert len(target_tokens) == len(
        old_logprobs
    ), f"Target tokens: {len(target_tokens)}, old logprobs: {len(old_logprobs)}"

    encoding["rewards"] = [0.0] * (len(encoding["labels"]) - len(old_logprobs)) + rewards_per_line[:1] * len(
        old_logprobs
    )
    encoding["advantages"] = [0.0] * len(encoding["labels"])  # place holder
    encoding["old_logprobs"] = [0.0] * (len(encoding["labels"]) - len(old_logprobs)) + old_logprobs
    encoding["ref_logprobs"] = [0.0] * (len(encoding["labels"]) - len(old_logprobs)) + ref_logprobs
    return encoding
