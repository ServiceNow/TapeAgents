import logging
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Dict, Mapping, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import BatchEncoding

from .utils import (
    StepConfig,
    calculate_advantage,
    calculate_reward_with_implicit_kl,
    masked_mean,
    replace_dataset_column,
)

# FIXME: remove a warnings, but might be worth investigating
os.environ["TOKENIZERS_PARALLELISM"] = "false"


logger = logging.getLogger(__name__)

RL_DATA_COLUMNS = [
    "reward",
    "rewards",
    "advantages",
    "old_logprobs",
    "ref_logprobs",
]


@dataclass
class RLConfig(StepConfig):
    algo: Optional[str] = field(
        default="grpo", metadata={"help": "Algorithm to use for RL", "choices": ["grpo", "reinforce"]}
    )
    use_advantages: Optional[bool] = field(
        default=True,
        metadata={"help": "Use advantages instead of rewards to compute the loss"},
    )
    epsilon: Optional[float] = field(default=0.2, metadata={"help": "Clip parameter for the ration of log probs"})
    implicit_kl_coef: Optional[float] = field(
        default=0.,
        # https://arxiv.org/abs/2402.14740
        metadata={"help": "Implicit KL coefficient similar to the RLOO paper"},
    )
    kl_coef: Optional[float] = field(
        default=0.1,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )
    ratio_threshold: Optional[float] = field(
        default=10.1,
        metadata={"help": "Skip mini-batches with high PPO ratios that can cause loss spike"},
    )
    relu_weights: Optional[bool] = field(
        default=False,
        metadata={"help": "ReLU the weights before updating the model"},
    )



def make_rl_data_callback(args, current_dir, rl_config, model):
    if rl_config:
        populate_rl_data_ = partial(
            populate_rl_data,
            config=rl_config,
        )
    else:
        populate_rl_data_ = None
    return populate_rl_data_


def rl_step(model, batch, config: RLConfig) -> tuple[torch.Tensor, dict[str, float]]:
    """
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
    assert new_log_probs.shape == ref_logprobs.shape

    # First compute the PPO surrogate loss, see https://arxiv.org/pdf/2402.03300 eq 3
    log_ratio_new_old = new_log_probs - old_logprobs
    ratio_new_old = torch.exp(log_ratio_new_old)
    weights = advantages if config.use_advantages else rewards
    weights = torch.clamp(weights, min=0) if config.relu_weights else weights
    # Second compute the approximated KL, see https://arxiv.org/pdf/2402.03300 eq 4
    log_ratio_ref_new = ref_logprobs - new_log_probs
    approx_kl = torch.exp(log_ratio_ref_new) - log_ratio_ref_new - 1  # Schulman KL approx
    match config.algo:
        case "grpo":
            # GRPO is based on https://arxiv.org/pdf/2402.03300
            surr1 = ratio_new_old * weights

            clamped_ratio = torch.clamp(ratio_new_old, 1 - config.epsilon, 1 + config.epsilon)

            surr2 = clamped_ratio * weights

            surrogate_loss = torch.min(surr1, surr2)

            assert approx_kl.shape == masks_.shape
            assert approx_kl.shape == surrogate_loss.shape
            loss = -masked_mean(surrogate_loss - config.kl_coef * approx_kl, masks_)
        case "reinforce":
            surr1 = torch.zeros_like(ratio_new_old)
            surr2 = torch.zeros_like(ratio_new_old)
            loss = -masked_mean(new_log_probs * weights - config.kl_coef * approx_kl, masks_)
        case _:
            raise ValueError(f"Unknown algorithm {config.algo}")
    assert torch.isfinite(loss).all(), "loss contains NaN or inf"

    if (
        masked_mean(ratio_new_old, masks_) > config.ratio_threshold
    ):  # https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1236
        loss = loss * 0

    stats = {
        "max_new_log_probs": new_log_probs[masks_].max().item(),
        "max_ratio_new_old": ratio_new_old[masks_].max().item(),
        "max_loss": loss.max().item(),
        "reward": masked_mean(rewards, masks_).item(),
        "max_reward": rewards[masks_].max().item(),
        "min_reward": rewards[masks_].min().item(),
        "mean_old_logprobs": masked_mean(old_logprobs, masks_).item(),
        "mean_new_logprobs": masked_mean(new_log_probs, masks_).item(),
        "mean_new_logprobs_positive_weights": masked_mean(new_log_probs[weights > 0], masks_[weights > 0]).item() if (weights > 0).any() else 0,
        "mean_new_logprobs_negative_weights": masked_mean(new_log_probs[weights < 0], masks_[weights < 0]).item() if (weights < 0).any() else 0,
        "mean_ref_logprobs": masked_mean(ref_logprobs, masks_).item(),
        "advantage": masked_mean(advantages, masks_).item(),
        "max_advantage": advantages[masks_].max().item(),
        "min_advantage": advantages[masks_].min().item(),
        "loss": loss.item(),
        "kl": masked_mean(approx_kl, masks_).item(),
        "max_kl": approx_kl[masks_].max().item(),
        "min_kl": approx_kl[masks_].min().item(),
        "surr1": masked_mean(surr1, masks_).item(),
        "surr2": masked_mean(surr2, masks_).item(),
        "ratio_new_old": masked_mean(ratio_new_old, masks_).item(),
        "ratio_ref_new": masked_mean(torch.exp(log_ratio_ref_new), masks_).item(),
        "ratio_ref_old": masked_mean(torch.exp(ref_logprobs - old_logprobs), masks_).item(),
    }
    return loss, stats


def update_rewards_and_advantages(dataset: Dataset, config: RLConfig) -> Dataset:
    """
    Updates the advantages column in the given dataset based on reward statistics.

    Args:
        dataset (Dataset): The input dataset containing rewards and placeholder advantages.

    Returns:
        Dataset: The updated dataset with the updated advantages column.

    """
    df = dataset.to_pandas()

    if config.implicit_kl_coef > 0:
        logger.info("Updating Reward with Implicit KL")
        calculate_reward_with_implicit_kl_ = partial(
            calculate_reward_with_implicit_kl, implicit_kl_coef=config.implicit_kl_coef
        )
        df["reward"] = df.apply(calculate_reward_with_implicit_kl_, axis=1)

    # Group by group_id and compute mean and std of reward
    grouped = df.groupby("group_id")["reward"].agg(["mean", "std", "count"]).reset_index()

    # Rename columns for clarity
    grouped.columns = ["group_id", "reward_mean", "reward_std", "count"]

    # Merge the computed statistics back to the original dataset
    df_with_stats = pd.merge(df, grouped, on="group_id", how="left")

    df_with_stats["advantages"] = df_with_stats.apply(calculate_advantage, axis=1)

    # replace advantages entry
    dataset = replace_dataset_column(dataset, "advantages", df_with_stats["advantages"].tolist())

    # Convert back to a Hugging Face Dataset
    return dataset


def populate_rl_data(
    dataset: Dataset,
    columns: list[str],
    collate_fn: Callable,
    config: RLConfig,
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

    dataset = update_rewards_and_advantages(dataset, config)

    logger.info("Finish Populate RL Data")
    return dataset


def prepare_rl_fields(
    encoding: BatchEncoding,
    reward: float,
    old_logprobs: list[float],
    ref_logprobs: list[float],
) -> BatchEncoding:
    """
    Convert reward per agent step to reward per token and add returns and advantages placeholders
    """
    target_tokens = [token for token in encoding["labels"] if token != -100]
    assert len(target_tokens) == len(
        old_logprobs
    ), f"Target tokens: {len(target_tokens)}, old logprobs: {len(old_logprobs)}"

    encoding["rewards"] = [reward] * len(encoding["labels"])
    encoding["advantages"] = [0.0] * len(encoding["labels"])  # place holder
    encoding["old_logprobs"] = [0] * (len(encoding["labels"]) - len(old_logprobs)) + old_logprobs
    encoding["ref_logprobs"] = [0] * (len(encoding["labels"]) - len(ref_logprobs)) + ref_logprobs
    return encoding
