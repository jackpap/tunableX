from __future__ import annotations

import json

import pytest


@pytest.mark.skipif(pytest.importorskip("jsonargparse") is None, reason="jsonargparse not installed")
def test_centralized_pipeline_defaults_and_overrides(run_example):
    code, out, err = run_example(
        "examples/jsonargparse_app/train_jsonarg_params.py",
        [
            "--train.epochs",
            "21",
            "--model.hidden_sizes",
            "300",
            "250",
            "--model.preprocess.normalize",
            "minmax",
        ],
    )
    assert code == 0, err
    # These asserts verify that functions read values via Class.attr
    assert "build_model [300, 250]" in out
    assert "train 21 32 adam" in out


@pytest.mark.skipif(pytest.importorskip("jsonargparse") is None, reason="jsonargparse not installed")
def test_centralized_pipeline_file_plus_overrides(tmp_path, run_example):
    cfg = tmp_path / "train_config.json"
    cfg.write_text(
        json.dumps(
            {
                "model": {
                    "dropout": 0.2,
                    "hidden_sizes": 256,
                    "preprocess": {"dropna": False, "normalize": "minmax", "clip_outliers": 2.5},
                },
                "train": {"epochs": 5, "batch_size": 8, "optimizer": "sgd"},
            }
        )
    )
    code, out, err = run_example(
        "examples/jsonargparse_app/train_jsonarg_params.py",
        ["--config", str(cfg), "--train.epochs", "50", "--model.hidden_sizes", "512", "--model.dropout", "0.15"],
    )
    assert code == 0, err
    assert "build_model [512] 0.15" in out
    assert "train 50 8 sgd" in out
