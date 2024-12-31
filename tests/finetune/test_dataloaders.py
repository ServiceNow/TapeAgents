from pathlib import Path
from typing import Dict, List

import pytest
import torch
import transformers

from tapeagents.finetune.data import collate, preprocess_fn

res_path = Path(__file__).parent.resolve() / "res"


@pytest.fixture
def tokenizer():
    return transformers.AutoTokenizer.from_pretrained(res_path / "tokenizer/starcoder/")


def test_preprocess_text(tokenizer):
    short_string = "hello world"
    sample = preprocess_fn({"text": short_string}, tokenizer, seq_length=2048)
    assert sample == {
        "input_ids": [7656, 5788],
        "attention_mask": [1, 1],
        "offset_mapping": [(0, 5), (5, 11)],
        "labels": [7656, 5788],
    }


def test_preprocess_spans(tokenizer):
    short_string = "hello world over here"
    sample = preprocess_fn({"text": short_string, "predicted_spans": [[6, 11], [18, 21]]}, tokenizer, seq_length=2048)
    assert sample == {
        "input_ids": [7656, 5788, 2288, 2442],
        "attention_mask": [1, 1, 1, 1],
        "offset_mapping": [(0, 5), (5, 11), (11, 16), (16, 21)],
        "labels": [-100, 5788, -100, 2442],
    }


def test_preprocess_wrong_span(tokenizer):
    short_string = "hello world over here"
    with pytest.raises(ValueError):
        preprocess_fn({"text": short_string, "predicted_spans": [[6, 11], [18, 22]]}, tokenizer, seq_length=2048)


def test_preprocess_truncation(tokenizer):
    string = "hello world over here" * 15
    seq_length = 20
    sample = preprocess_fn({"text": string}, tokenizer, seq_length=seq_length)
    for k, v in sample.items():
        assert len(v) == seq_length, f"Truncation failed for field {k}"


def test_preprocess_long(tokenizer):
    long_string = "a = 1 + 2 * 3 - 4 / 5\n" * 500
    seq_length = 2048
    sample = preprocess_fn({"text": long_string}, tokenizer, seq_length=seq_length)
    for k, v in sample.items():
        assert len(v) == seq_length, f"Truncation failed for field {k}"


def get_samples(tokenizer, seq_length) -> List[Dict[str, List[int]]]:
    short_string = "hello world"
    mid_string = "hello world over here"
    long_string = "this is the test of the padding on the longest string"
    samples = [
        preprocess_fn({"text": text}, tokenizer, seq_length=seq_length)
        for text in [short_string, long_string, mid_string]
    ]
    for s in samples:
        s.pop("offset_mapping")
    return samples  # type: ignore


def test_collate_clm(tokenizer):
    samples = get_samples(tokenizer, seq_length=16)
    batch = collate(samples, tokenizer, pad_to_multiple_of=4)
    for v in batch.values():
        assert isinstance(v, torch.Tensor)
    batch = {k: v.tolist() for k, v in batch.items()}
    assert batch == {
        "input_ids": [
            [7656, 5788, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [578, 438, 322, 894, 432, 322, 3714, 544, 322, 30698, 802, 0],
            [7656, 5788, 2288, 2442, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        "attention_mask": [
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        "labels": [
            [7656, 5788, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100],
            [578, 438, 322, 894, 432, 322, 3714, 544, 322, 30698, 802, -100],
            [7656, 5788, 2288, 2442, -100, -100, -100, -100, -100, -100, -100, -100],
        ],
    }


def test_collate_left_padding(tokenizer):
    tokenizer.padding_side = "left"
    left_pad_samples = get_samples(tokenizer, seq_length=16)
    batch = collate(left_pad_samples, tokenizer)
    for v in batch.values():
        assert isinstance(v, torch.Tensor)
    batch = {k: v.tolist() for k, v in batch.items()}
    assert batch == {
        "input_ids": [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7656, 5788],
            [0, 0, 0, 0, 0, 578, 438, 322, 894, 432, 322, 3714, 544, 322, 30698, 802],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7656, 5788, 2288, 2442],
        ],
        "attention_mask": [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        ],
        "labels": [
            [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 7656, 5788],
            [-100, -100, -100, -100, -100, 578, 438, 322, 894, 432, 322, 3714, 544, 322, 30698, 802],
            [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 7656, 5788, 2288, 2442],
        ],
    }


def test_collate_multiple_of_16(tokenizer):
    non_16_samples = get_samples(tokenizer, seq_length=5)
    batch = collate(non_16_samples, tokenizer)
    for v in batch.values():
        assert isinstance(v, torch.Tensor)
    batch = {k: v.tolist() for k, v in batch.items()}
    assert batch == {
        "input_ids": [
            [7656, 5788, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [578, 438, 322, 894, 432, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [7656, 5788, 2288, 2442, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        "attention_mask": [
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        "labels": [
            [7656, 5788, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100],
            [578, 438, 322, 894, 432, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100],
            [7656, 5788, 2288, 2442, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100],
        ],
    }


if __name__ == "__main__":
    _tokenizer = transformers.AutoTokenizer.from_pretrained(res_path / "tokenizer/starcoder/")
    test_preprocess_text(_tokenizer)
    test_preprocess_spans(_tokenizer)
    test_preprocess_wrong_span(_tokenizer)
    test_preprocess_truncation(_tokenizer)
    test_preprocess_long(_tokenizer)
    test_collate_clm(_tokenizer)
    test_collate_left_padding(_tokenizer)
    test_collate_multiple_of_16(_tokenizer)
