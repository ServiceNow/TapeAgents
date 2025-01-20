import json
import os
import sys
import tempfile
from pathlib import Path
from pprint import pprint

import pytest
import torch

res_path = Path(__file__).parent.resolve() / "res"


def _almost_equal(a, b, eps=1e-2):
    return abs(a - b) < eps


def _check_tuning_artifacts(artifact_paths, deepspeed=False, lora=False):
    artifact_paths_set = set(artifact_paths)
    expected_paths = [
        "current",
        "intermediate",
        "intermediate/16",
        "intermediate/32",
        "log",
        "log/info_0.log",
        "summary.json",
        "training_state",
        "wandb_info.json",
    ]
    if deepspeed:
        expected_paths.append("training_state/deepspeed")
        expected_paths.append("training_state/latest")
    else:
        expected_paths.append("training_state/training_state.pt")

    checkpoint_files = [
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ]
    if lora:
        checkpoint_files.append("adapter_config.json")
    else:
        checkpoint_files.append("config.json")
        checkpoint_files.append("model.safetensors")
        checkpoint_files.append("generation_config.json")
    for prefix in ["current/", "intermediate/16/", "intermediate/32/"]:
        for checkpoint_file in checkpoint_files:
            expected_paths.append(prefix + checkpoint_file)

    pprint(artifact_paths, stream=sys.stderr)
    for path in expected_paths:
        assert path in artifact_paths_set, f'Path "{path}" not found in artifacts'


def _check_tuning_result(result, golden_result_name, update_golden_result):
    golden_result_filename = res_path / f"finetune/results/{golden_result_name}.json"
    if update_golden_result:
        with open(golden_result_filename, "w") as wf:
            json.dump(result, wf, indent=4, sort_keys=True)
        return

    with open(golden_result_filename) as f:
        golden_result = json.load(f)

    pprint(result, stream=sys.stderr)
    result_keys = set(result.keys())
    assert all(k in result_keys for k in golden_result.keys())
    not_equals = []
    for k, golden in golden_result.items():
        v = result[k]
        same = _almost_equal(v, golden)
        if not same:
            not_equals.append(f"{k} = {v} (expected {golden})")

    assert not len(not_equals), ". ".join(not_equals)


def _run_isolated_tuning(config_name, overrides="", mixed_precision="no", distributed_mode="no"):
    with tempfile.TemporaryDirectory() as tmpdirname:
        _run_tuning(config_name, overrides, tmpdirname, mixed_precision, distributed_mode)
        tmpdirpath = Path(tmpdirname)
        artifacts = _get_run_artifacts(tmpdirpath)
        with open(tmpdirpath / "summary.json") as f:
            result = json.load(f)
    return result, artifacts


def _run_tuning(config_name, overrides, tmpdirname, mixed_precision="no", distributed_mode="no"):
    config_dir = res_path / "conf"
    accelerate_args = _prepare_accelerate_args(mixed_precision, distributed_mode)
    cmd = f"accelerate launch {accelerate_args} {res_path}/../run_finetune.py --config-dir={config_dir} finetune={config_name} {overrides} hydra.run.dir={tmpdirname}"
    print(cmd)
    exitcode = os.system(cmd)
    assert exitcode == 0, f"Tuning failed with code {exitcode}"


def _get_run_artifacts(tmpdirpath):
    artifacts = sorted([str(path).replace(str(tmpdirpath) + "/", "") for path in tmpdirpath.glob("**/*")])

    return artifacts


def _prepare_accelerate_args(mixed_precision, distributed_mode):
    accelerate_args = [f"--mixed_precision={mixed_precision}"]
    if distributed_mode == "multi_gpu":
        accelerate_args += [
            "--config_file conf/accelerate/accelerate_base.yaml",
            "--num_processes 2",
            "--multi_gpu",
        ]
    elif distributed_mode == "deepspeed":
        accelerate_args += [
            "--config_file conf/accelerate/accelerate_base.yaml",
            "--use_deepspeed",
            "--deepspeed_config_file conf/accelerate/deepspeed_stage3_bf16.json",
        ]
    else:
        accelerate_args.append(f"--config_file {res_path}/conf/accelerate/accelerate_local.yaml")
    return " ".join(accelerate_args)


# All the following tests use separate test-time only config test.yaml


@pytest.mark.slow
def test_tuning(update_golden_result=False):
    result, artifacts = _run_isolated_tuning("test")
    _check_tuning_artifacts(artifacts)
    _check_tuning_result(result, "regular", update_golden_result)


@pytest.mark.slow
def test_tuning_grad_checkpoints(update_golden_result=False):
    result, artifacts = _run_isolated_tuning("test", overrides="finetune.gradient_checkpointing=true")
    _check_tuning_artifacts(artifacts)
    _check_tuning_result(result, "gradient_checkpointing", update_golden_result)


@pytest.mark.slow
@pytest.mark.gpu
def test_tuning_bf16(update_golden_result=False):
    result, artifacts = _run_isolated_tuning("test", mixed_precision="bf16")
    _check_tuning_artifacts(artifacts)
    _check_tuning_result(result, "bf16", update_golden_result)


@pytest.mark.slow
@pytest.mark.gpu
def test_tuning_fp16(update_golden_result=False):
    result, artifacts = _run_isolated_tuning("test", mixed_precision="fp16")
    _check_tuning_artifacts(artifacts)
    _check_tuning_result(result, "fp16", update_golden_result)


@pytest.mark.slow
@pytest.mark.gpu
def test_tuning_load_as_bf16(update_golden_result=False):
    result, artifacts = _run_isolated_tuning("test", overrides="finetune.load_as_bf16=true")
    _check_tuning_artifacts(artifacts)
    _check_tuning_result(result, "half_precision", update_golden_result)


@pytest.mark.slow
@pytest.mark.multi_gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="multi GPU test requires 2+ GPUs")
def test_tuning_multi_gpu(update_golden_result=False):
    assert torch.cuda.device_count() >= 2
    result, artifacts = _run_isolated_tuning("test", distributed_mode="multi_gpu")
    _check_tuning_artifacts(artifacts)
    _check_tuning_result(result, "multi_gpu", update_golden_result)


@pytest.mark.slow
@pytest.mark.gpu
def test_tuning_deepspeed(update_golden_result=False):
    result, artifacts = _run_isolated_tuning("test", distributed_mode="deepspeed")
    _check_tuning_artifacts(artifacts, deepspeed=True)
    _check_tuning_result(result, "deepspeed", update_golden_result)


@pytest.mark.slow
def test_tuning_lora(update_golden_result=False):
    result, artifacts = _run_isolated_tuning(
        "test",
        overrides=(
            "finetune.learning_rate=5e-3 finetune.lora.enabled=true finetune.lora.target_modules=[c_proj,c_attn]"
        ),
    )
    _check_tuning_artifacts(artifacts, lora=True)
    _check_tuning_result(result, "lora", update_golden_result)


@pytest.mark.slow
def test_tuning_resumption(update_golden_result=False):
    with tempfile.TemporaryDirectory() as tmpdirname:
        _run_tuning("test", "finetune.interrupt_train_steps=20", tmpdirname)
        tmpdirpath = Path(tmpdirname)
        with open(tmpdirpath / "summary.json") as f:
            intermediate_result = json.load(f)
        intermediate_artifacts = _get_run_artifacts(tmpdirpath)
        assert "current" in intermediate_artifacts
        assert intermediate_result["completed_steps"] == 20
        # resume training in the same folder
        _run_tuning("test", "", tmpdirname)
        artifacts = _get_run_artifacts(tmpdirpath)
        with open(tmpdirpath / "summary.json") as f:
            result = json.load(f)
        with open(tmpdirpath / "log/info_0.log") as f:
            log_data = f.read()

    assert "Loading model " in log_data
    _check_tuning_artifacts(artifacts)
    _check_tuning_result(result, "resumption", update_golden_result)


if __name__ == "__main__":
    update_golden_result = False
    test_tuning(update_golden_result)
    test_tuning_bf16(update_golden_result)
    test_tuning_fp16(update_golden_result)
    test_tuning_load_as_bf16(update_golden_result)
    test_tuning_grad_checkpoints(update_golden_result)
    test_tuning_deepspeed(update_golden_result)
    test_tuning_lora(update_golden_result)
    test_tuning_resumption(update_golden_result)
    test_tuning_multi_gpu(update_golden_result)
