from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import pathlib


def make_train_config(
    path: pathlib.Path,
    epochs=10,
    batch_size=32,
    optimizer="adam",
    hidden_units=128,
    dropout=0.2,
    dropna=True,
    normalize="zscore",
    clip_outliers=3.0,
):
    data = {
        "model": {
            "hidden_units": hidden_units,
            "dropout": dropout,
            "preprocess": {"dropna": dropna, "normalize": normalize, "clip_outliers": clip_outliers},
        },
        "train": {"epochs": epochs, "batch_size": batch_size, "optimizer": optimizer},
    }
    path.write_text(json.dumps(data))
    return path


def test_argparse_app_with_config(tmp_path, run_example):
    cfg = make_train_config(tmp_path / "train_config.json", epochs=11, batch_size=64, optimizer="sgd")
    code, out, err = run_example(
        "examples/argparse_app/train_app.py",
        ["--config", str(cfg)],
    )
    assert code == 0, err
    assert "preprocess" in out
    assert "build_model" in out
    assert "train 11 64 sgd" in out


def test_argparse_trace_with_config(tmp_path, run_example):
    cfg = make_train_config(tmp_path / "train_config_traced.json", epochs=7, batch_size=16, optimizer="adam")
    code, out, err = run_example(
        "examples/argparse_trace/train_trace.py",
        ["--config", str(cfg)],
    )
    assert code == 0, err
    assert "train 7 16 adam" in out


@pytest.mark.skipif(pytest.importorskip("jsonargparse") is None, reason="jsonargparse not installed")
def test_jsonargparse_app_defaults_and_overrides(run_example):
    code, out, err = run_example(
        "examples/jsonargparse_app/train_jsonarg_app.py",
        ["--train.epochs", "20", "--model.hidden_units", "256", "--model.preprocess.normalize", "minmax"],
    )
    assert code == 0, err
    assert "build_model 256" in out
    assert "train 20 32 adam" in out  # batch_size default remains


@pytest.mark.skipif(pytest.importorskip("jsonargparse") is None, reason="jsonargparse not installed")
def test_jsonargparse_app_file_plus_overrides(tmp_path, run_example):
    cfg = make_train_config(tmp_path / "train_config.json", epochs=5, batch_size=8, optimizer="sgd", hidden_units=128)
    code, out, err = run_example(
        "examples/jsonargparse_app/train_jsonarg_app.py",
        ["--config", str(cfg), "--train.epochs", "50", "--model.hidden_units", "512"],
    )
    assert code == 0, err
    assert "build_model 512" in out
    assert "train 50 8 sgd" in out


@pytest.mark.skipif(pytest.importorskip("jsonargparse") is None, reason="jsonargparse not installed")
def test_jsonargparse_trace_defaults_cli(run_example):
    code, out, err = run_example(
        "examples/jsonargparse_trace/train_jsonarg_trace.py",
        ["--no-model.preprocess.dropna", "--train.batch_size", "64"],
    )
    assert code == 0, err
    assert "train " in out  # epochs default, batch overridden
    assert "64" in out
