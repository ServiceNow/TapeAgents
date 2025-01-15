import os
import time
from functools import partial
from typing import Any, Callable, Iterable, Sequence

import datasets
import torch
import transformers
from datasets.arrow_dataset import Dataset
from datasets.combine import interleave_datasets
from datasets.fingerprint import Hasher
from datasets.load import load_dataset
from omegaconf import DictConfig
from torch.utils.data.dataloader import DataLoader
from transformers import BatchEncoding

from tapeagents.core import TrainingText

from .context import accelerator, logger
from .logging_ import log_time
from .rl import RL_DATA_COLUMNS, prepare_rl_fields
from .types import DataArgs, DataPartArgs

datasets.builder.has_sufficient_disk_space = (
    lambda needed_bytes, directory=".": True
)  # hack for NFS filesystem with 0 disk space reported

# -100 is the default "ignore_index" in nn.CrossEntropyLoss,
# see https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
MASKED_TOKEN_ID = -100


def save_samples(training_samples: list[TrainingText], jsonl_filename: str):
    assert jsonl_filename.endswith(".jsonl"), f"Filename {jsonl_filename} must end with .jsonl"
    with open(jsonl_filename, "w") as f:
        for sample in training_samples:
            f.write(sample.model_dump_json() + "\n")


def load_samples(file: str) -> list[TrainingText]:
    samples = []
    with open(file) as f:
        for line in f.readlines():
            samples.append(TrainingText.model_validate_json(line))
    return samples


def mask_labels(
    input_ids: Sequence[int],
    offset_mapping: Iterable[tuple[int, int]],
    predicted_spans: Iterable[Iterable[int]],
    masked_token_id: int = MASKED_TOKEN_ID,
) -> tuple[list[int], list[int]]:
    """
    This function creates labels from a sequence of input ids by masking
    the tokens that do not have any overlap with the character spans that
    are designated for prediction. The labels can then be used to train
    a model to predict everything except the masked tokens.

    The function also returns a list of midpoints for splitting the
    labels into a source and a target. The source is the part of the
    labels that is used to predict the target. There is one midpoint
    for each span that is designated for prediction. Each midpoint is
    the index of the first token that overlaps with the corresponding
    span.

    Args:
        input_ids (Sequence[int]): A sequence of token ids.
        offset_mapping (Iterable[tuple[int, int]]): The offset mapping
            returned by the tokenizer.
        predicted_spans (Iterable[Iterable[int]]): The character spans
            that are designated for prediction. The spans are given as
            a sequence of two-element sequences, where the first element
            is the beginning of the span (inclusive) and the second
            element is the end of the span (not inclusive).

    Returns:
        tuple[list[int], list[int]]: A tuple of masked labels and
            corresponding midpoints for splitting the labels into
            a source and a target.
    """
    labels = [masked_token_id] * len(input_ids)
    midpoints = []
    # TODO: Make this O(n_tokens) instead of O(n_tokens * n_spans)
    for span_begin, span_end in predicted_spans:
        midpoint_found = False
        for i, (offset_begin, offset_end) in enumerate(offset_mapping):
            # visual inspection of the results shows that this is the correct way to check
            if offset_begin < span_end and span_begin < offset_end:
                if not midpoint_found:
                    midpoints.append(i)
                    midpoint_found = True
                labels[i] = input_ids[i]
    return labels, midpoints


def validate_spans(text: str, predicted_spans: list[tuple[int, int]]) -> None:
    """Make sure the spans are valid, don't overlap, and are in order."""
    for start, end in predicted_spans:
        if start < 0 or end > len(text):
            raise ValueError(f"Span {start}:{end} is out of bounds for text {text!r}")
        if start > end:
            raise ValueError(f"Span {start}:{end} is invalid")
    for (start1, end1), (start2, end2) in zip(predicted_spans, predicted_spans[1:]):
        # Make sure the second span starts after the first one ends.
        if start2 < end1:
            raise ValueError(
                f"Spans {start1}:{end1} ({text[start1:end1]!r}) and {start2}:{end2} ({text[start2:end2]!r}) overlap"
            )


def preprocess_fn(
    entry: dict[str, Any],
    tokenizer: transformers.PreTrainedTokenizerBase,
    seq_length: int,
    is_rl: bool = False,
) -> BatchEncoding:
    if "input_ids" in entry and entry["input_ids"]:
        # build the `encoding` object from the given tokenization
        encoding = BatchEncoding()
        encoding["input_ids"] = entry["input_ids"]
        encoding["labels"] = entry["labels"]
        encoding["attention_mask"] = [1] * len(entry["input_ids"])
    else:
        # tokenize text to build the `encoding` object
        encoding = tokenizer(
            entry["text"],
            return_offsets_mapping=True,
            max_length=seq_length,
            truncation=True,
        )
        if "predicted_spans" in entry:
            predicted_spans = entry["predicted_spans"]
        else:
            text_length = len(entry["text"])
            predicted_chars = entry.get("n_predicted", text_length)
            predicted_spans = [(text_length - predicted_chars, text_length)]
        validate_spans(entry["text"], predicted_spans)
        encoding["labels"], _ = mask_labels(
            encoding["input_ids"],  # type: ignore
            encoding["offset_mapping"],  # type: ignore
            predicted_spans,
        )
    if is_rl:
        encoding = prepare_rl_fields(
            encoding,
            entry["reward"],
            entry["logprobs"],
            entry["ref_logprobs"],
        )
    return encoding


def collate(
    examples: list[dict[str, list[int]]],
    tokenizer: transformers.PreTrainedTokenizerBase,
    label_mask_value: int = MASKED_TOKEN_ID,
    pad_to_multiple_of: int = 16,
) -> BatchEncoding:
    # turn list of dicts with the same keys into a dict of lists
    example_dict = {key: [example[key] for example in examples] for key in examples[0].keys()}
    seq_length = max(len(i) for i in example_dict["input_ids"])
    if seq_length % pad_to_multiple_of:
        seq_length += pad_to_multiple_of - (seq_length % pad_to_multiple_of)
    result = {}
    for k, seq_list in example_dict.items():
        padded_sequences = []
        pad_value = label_mask_value if k == "labels" else (0.0 if k in RL_DATA_COLUMNS else 0)
        for seq in seq_list:
            if not isinstance(seq, list):
                seq = [seq]
            padding = [pad_value] * (seq_length - len(seq))
            padded = (seq + padding) if tokenizer.padding_side == "right" else (padding + seq)
            padded_sequences.append(padded)
        result[k] = torch.tensor(padded_sequences)
    return BatchEncoding(result, tensor_type="pt")


def create_dataloader(
    data_parts: list[DataPartArgs] | list[TrainingText],
    tokenizer: transformers.PreTrainedTokenizerBase,
    batch_size: int,
    seq_length: int,
    is_rl: bool = False,
    rng: torch.Generator | None = None,
    shuffle: bool = False,
    rl_data_callback: Callable | None = None,
    n_examples: int | None = None,
) -> DataLoader:
    preprocess = partial(preprocess_fn, seq_length=seq_length, tokenizer=tokenizer, is_rl=is_rl)
    columns = ["input_ids", "labels", "attention_mask"]
    if is_rl:
        columns += RL_DATA_COLUMNS

    logger.info(f"Instantiated preprocess function hash {Hasher.hash(preprocess)}")
    collate_fn = partial(
        collate,
        tokenizer=tokenizer,
    )
    logger.info(f"Instantiated collate_fn hash {Hasher.hash(collate_fn)}")

    datasets = []
    weights = []
    stop = False
    for part in data_parts:
        if isinstance(part, TrainingText):
            dataset_part = Dataset.from_list([s.model_dump() for s in data_parts])
            weights.append(1.0)
            stop = True
        else:
            # The path must point to the directory containing the data files
            # for one split of interest. `load_dataset` will automatically call
            # this split "train".
            dataset_part = load_dataset(part.path, split="train", data_files=part.files)
            assert isinstance(dataset_part, Dataset)
            weights.append(part.weight)

        logger.info(f"Raw data part size: {dataset_part.num_rows}")
        logger.info(f"Raw data part fingerprint: {dataset_part._fingerprint}")

        # dataset_part = dataset_part.shard(
        #    num_shards=accelerator.num_processes,
        #    index=accelerator.process_index,
        # )

        # Each process preprocesses its shard
        if accelerator.is_main_process:
            dataset_part = dataset_part.map(
                preprocess, keep_in_memory=True, load_from_cache_file=False, num_proc=os.cpu_count()
            )
            dataset_part = dataset_part.with_format(columns=columns)

        # Wait for all processes to finish preprocessing
        # accelerator.wait_for_everyone()

        logger.info(f"Preprocessed data part fingerprint: {dataset_part._fingerprint}")
        datasets.append(dataset_part)
        if stop:
            break
    total_weight = sum(weights)
    probs = [w / total_weight for w in weights]
    data = interleave_datasets(
        datasets,
        probabilities=probs,
        stopping_strategy="all_exhausted",
        seed=rng.initial_seed() if rng is not None else None,
    )
    logger.info(f"Merged data size: {data.num_rows}")
    logger.info(f"Merged data fingerprint: {data._fingerprint}")

    if rl_data_callback is not None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            num_cpus = os.cpu_count() or 1
            logger.info(f"Populate RL Data with {num_cpus} workers")
            # Group data by group_id to keep related samples together
            group_ids = data["group_id"]
            unique_groups = list(set(group_ids))
            
            # Assign groups to CPU processes
            process_assignments = {}
            for group in unique_groups:
                # Use consistent hash to ensure same group always goes to same process
                process_id = hash(str(group)) % num_cpus
                process_assignments[group] = process_id
            
            # Split data into shards by process assignment
            shards = [[] for _ in range(num_cpus)]
            for i, group in enumerate(group_ids):
                process_id = process_assignments[group]
                shards[process_id].append(i)
                
            # Convert index lists to datasets
            shard_datasets = [data.select(indices) for indices in shards if indices]
            from concurrent import futures
            rl_data_callback = partial(rl_data_callback, columns=columns, collate_fn=collate_fn)
            with futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
                shard_datasets = list(executor.map(rl_data_callback, shard_datasets))
            # merge the data back together
            data = Dataset.from_list(shard_datasets)
            logger.info("Finish Populate RL Data")

            #data = rl_data_callback(dataset=data, columns=columns, collate_fn=collate_fn)

    if n_examples:
        data = data.select(range(n_examples))

    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        generator=rng,
    )


def prepare_dataloaders(
    args: DictConfig,
    data_args: DataArgs,
    tokenizer: transformers.PreTrainedTokenizerBase,
    rl_data_callback: Callable | None,
    dataloader_rng: torch.Generator | None,
    is_rl: bool = False,
) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
    _create_dataloader = partial(
        create_dataloader,
        tokenizer=tokenizer,
        seq_length=args.seq_length,
        rng=dataloader_rng,
        rl_data_callback=rl_data_callback,
        is_rl=is_rl,
    )

    # Load dataset and dataloader
    train_dataloader = _create_dataloader(
        data_parts=data_args.data_parts_train,
        batch_size=args.train_batch_size,
        n_examples=args.n_examples,
        shuffle=True,
    )

    eval_dataloader = (
        _create_dataloader(
            data_parts=data_args.data_parts_valid,
            batch_size=args.valid_batch_size,
            shuffle=False,
        )
        if data_args.data_parts_valid
        else None
    )

    dev_dataloader = (
        _create_dataloader(
            data_parts=data_args.data_parts_dev,
            batch_size=args.valid_batch_size,
            shuffle=False,
        )
        if data_args.data_parts_dev
        else None
    )

    return train_dataloader, eval_dataloader, dev_dataloader
