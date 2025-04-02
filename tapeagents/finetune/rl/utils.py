from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from datasets import Dataset

from tapeagents.finetune.logging_ import flatten_dict_config


@dataclass
class StepConfig(object):
    def to_dict(self):
        output_dict = {}
        for key, value in self.__dict__.items():
            output_dict[key] = value
        return flatten_dict_config(output_dict)


def get_avg_rl_stats(rl_stats):
    avg_rl_stats: dict[str, float] = {}
    for k, v in rl_stats.items():
        if "min" in k:
            op = torch.min
        elif "max" in k:
            op = torch.max
        else:
            op = torch.mean
        avg_rl_stats["rl/" + k] = op(torch.Tensor(v)).item()
    return avg_rl_stats


def masked_sum(values: torch.Tensor, mask: torch.Tensor, axis: Optional[bool] = None) -> torch.Tensor:
    """Compute sum of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis)  # type: ignore
    else:
        return (values * mask).sum()


def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis: Optional[bool] = None) -> torch.Tensor:
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)  # type: ignore
    else:
        return (values * mask).sum() / mask.sum()


def calculate_rewards_with_implicit_kl(row, reward_minus_kl_coef):
    """
    Calculate reward with implicit KL penalty.

    Args:
        row (dict): Dictionary containing reward and log probability data with keys:

            - reward: Base reward value
            - old_logprobs: Log probabilities from old policy
            - ref_logprobs: Reference log probabilities
        reward_minus_kl_coef (float): Coefficient for implicit KL penalty term

    Returns:
        (float): Reward value adjusted by implicit KL penalty, calculated as:
            reward - reward_minus_kl_coef * KL(ref||old)

        The KL divergence is approximated using the Schulman approximation:
            KL ≈ exp(log_ratio) - log_ratio - 1
            where log_ratio = ref_logprobs - old_logprobs
    """
    rewards = row["rewards"]
    old_logprobs = row["old_logprobs"]
    ref_logprobs = row["ref_logprobs"]
    log_ratio_ref_old = ref_logprobs - old_logprobs
    kl = (np.exp(log_ratio_ref_old) - log_ratio_ref_old - 1).sum()  # Schulman KL approx
    return [reward - reward_minus_kl_coef * kl for reward in rewards]


def calculate_advantage(row):
    """
    Calculate advantage values for a row of data.

    Args:
        row (dict): Dictionary containing rewards and statistics with keys:

            - rewards: List of reward values
            - reward_mean: Mean reward value
            - reward_std: Standard deviation of rewards

    Returns:
       (list[float]): List of advantage values calculated as (reward - mean)/(std + eps)
            where eps=1e-4 is added for numerical stability
    """
    rewards = row["rewards"]
    mean = row["reward_mean"]
    std = row["reward_std"]
    advantages = [(reward - mean) / (np.nan_to_num(std) + 1e-4) for reward in rewards]
    return advantages


def replace_dataset_column(dataset: Dataset, column_name: str, new_column: List[List[float]]) -> Dataset:
    """
    Replace a column in the dataset with a new column.
    """
    if column_name in dataset.features:
        dataset = dataset.map(remove_columns=[column_name])
    dataset = dataset.add_column(name=column_name, column=new_column)  # type: ignore

    return dataset
