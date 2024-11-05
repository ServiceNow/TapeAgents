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


def calculate_reward_with_implicit_kl(row, implicit_kl_coef):
    reward = row["reward"]
    old_logprobs = row["old_logprobs"]
    ref_logprobs = row["ref_logprobs"]
    log_ratio_ref_old = ref_logprobs - old_logprobs
    kl = (np.exp(log_ratio_ref_old) - log_ratio_ref_old - 1).sum()  # Schulman KL approx
    return reward - implicit_kl_coef * kl


def calculate_advantage(row):
    rewards = row["rewards"]
    mean = row["reward_mean"]
    std = row["reward_std"]
    return [(reward - mean) / (np.nan_to_num(std) + 1e-4) for reward in rewards]


def replace_dataset_column(dataset: Dataset, column_name: str, new_column: List[List[float]]) -> Dataset:
    """
    Replace a column in the dataset with a new column.
    """
    if column_name in dataset.features:
        dataset = dataset.map(remove_columns=[column_name])
    dataset = dataset.add_column(name=column_name, column=new_column)  # type: ignore

    return dataset
