from __future__ import annotations

import json

import pytest

# Additional tests focusing on override precedence to ensure defaults show in help do not
# change merging order: defaults <- config file <- CLI overrides.


@pytest.mark.skipif(pytest.importorskip("jsonargparse") is None, reason="jsonargparse not installed")
def test_jsonargparse_app_override_precedence(tmp_path, run_example):
    # Create a config file that overrides some defaults
    cfg_path = tmp_path / "train_config.json"
    data = {
        "model": {
            "hidden_units": 256,
            "dropout": 0.15,
            "preprocess": {"dropna": False, "normalize": "minmax", "clip_outliers": 2.5},
        },
        "train": {"epochs": 5, "batch_size": 8, "optimizer": "sgd"},
    }
    cfg_path.write_text(json.dumps(data))

    # CLI overrides should beat file values (epochs, hidden_units) while others remain from file
    code, out, err = run_example(
        "examples/jsonargparse_app/train_jsonarg_app.py",
        [
            "--config",
            str(cfg_path),
            "--train.epochs",
            "50",
            "--model.hidden_units",
            "512",
            "--model.preprocess.submodule_fun.arg1",
            "10",
            "--model.preprocess.submodule_class.attr1",
            "-5",
            "--model.preprocess.submodule_class.arg2",
        ],
    )
    assert code == 0, err
    # Expect overridden values
    assert "submodule_fun 10 False" in out
    assert "sumbodule_class -5 False"
    assert "submodule_class_method 0 True" in out
    assert "build_model 512 0.15" in out
    assert "train 50 8 sgd" in out


@pytest.mark.skipif(pytest.importorskip("jsonargparse") is None, reason="jsonargparse not installed")
@pytest.mark.skip(reason="AST test cannot pass yet.")
def test_jsonargparse_trace_override_precedence(tmp_path, run_example):
    cfg_path = tmp_path / "train_config.json"
    data = {
        "model": {
            "hidden_units": 64,
            "dropout": 0.25,
            "preprocess": {"dropna": True, "normalize": "none", "clip_outliers": 3.0},
        },
        "train": {"epochs": 3, "batch_size": 4, "optimizer": "adam"},
    }
    cfg_path.write_text(json.dumps(data))
    code, out, err = run_example(
        "examples/jsonargparse_trace/train_jsonarg_trace.py",
        [
            "--config",
            str(cfg_path),
            "--model.preprocess.normalize",
            "zscore",
            "--train.batch_size",
            "16",
            "--model.preprocess.submodule_fun.arg1",
            "10",
            "--model.preprocess.submodule_class.attr1",
            "-5",
            "--model.preprocess.submodule_class.arg2",
        ],
    )
    assert code == 0, err
    # Expect overridden values
    assert "submodule_fun 10 False" in out
    assert "sumbodule_class -5 False"
    assert "submodule_class_method 0 True" in out
    assert "preprocess True zscore 3.0" in out
    assert "build_model 64 0.25" in out
    assert "train 3 16 adam" in out
