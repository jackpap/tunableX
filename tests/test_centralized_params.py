from __future__ import annotations

import json

import pytest


@pytest.mark.skipif(pytest.importorskip("jsonargparse") is None, reason="jsonargparse not installed")
def test_app_with_centralized_params_defaults_and_overrides(run_example):
    code, out, err = run_example(
        "examples/jsonargparse_app/train_jsonarg_params.py",
        ["--train.epochs", "20", "--model.hidden_sizes", "128", "256", "--model.preprocess.normalize", "minmax"],
    )
    assert code == 0, err
    assert "build_model [128, 256] 0.2 sum True root advanced_root" in out
    assert "train 20 32 adam" in out


@pytest.mark.skipif(pytest.importorskip("jsonargparse") is None, reason="jsonargparse not installed")
def test_help_shows_defaults_from_centralized_params(run_example):
    code, out, err = run_example(
        "examples/jsonargparse_app/train_jsonarg_params.py",
        ["--help"],
    )
    assert code == 0, err
    assert "(default: true)" in out
    assert "(default: (128, 128))" in out
    assert "(default: 10)" in out


@pytest.mark.skipif(pytest.importorskip("jsonargparse") is None, reason="jsonargparse not installed")
def test_file_plus_overrides_with_centralized_params(tmp_path, run_example):
    cfg = tmp_path / "train_config.json"
    cfg.write_text(
        json.dumps({
            "batch_norm": False,
            "root_param": "newRoot",
            "model": {
                "dropout": 0.15,
                "hidden_units": 256,
                "preprocess": {"dropna": False, "normalize": "minmax", "clip_outliers": 2.5},
            },
            "train": {"epochs": 5, "batch_size": 8, "optimizer": "sgd"},
        })
    )
    code, out, err = run_example(
        "examples/jsonargparse_app/train_jsonarg_params.py",
        ["--config", str(cfg), "--train.epochs", "50", "--model.hidden_sizes", "512", "--model.dropout", "0.25"],
    )
    assert code == 0, err
    assert "build_model [512] 0.25 sum False newRoot" in out
    assert "train 50 8 sgd" in out


@pytest.mark.skipif(pytest.importorskip("jsonargparse") is None, reason="jsonargparse not installed")
def test_other_argnames_from_centralized_params(tmp_path, run_example):
    cfg = tmp_path / "train_config.json"
    cfg.write_text(
        json.dumps({
            "batch_norm": False,
            "root_param": "newRoot",
            "model": {
                "dropout": 0.15,
                "hidden_units": 256,
                "preprocess": {"dropna": False, "normalize": "minmax", "clip_outliers": 2.5},
            },
            "train": {"epochs": 5, "batch_size": 8, "optimizer": "sgd"},
        })
    )
    code, out, err = run_example(
        "examples/jsonargparse_app/train_jsonarg_params.py",
        ["--config", str(cfg)],
    )
    assert code == 0, err
    assert "build_model [128, 128] 0.15 sum False newRoot" in out
    assert "train 5 8 sgd" in out
    assert "other_train 5 8 sgd" in out


def test_get_description_from_docstring(tmp_path, run_example):
    cfg_path = tmp_path / "test_config"
    run_example(
        "examples/app_generate_schema.py",
        ["--prefix", str(cfg_path)],
    )

    schema_path = cfg_path.with_suffix(".schema.json")
    with schema_path.open() as f:
        cfg = json.load(f)
    assert "clip_outliers" in str(cfg)
    assert "Description in docstring." in str(cfg)


def test_same_name_args(tmp_path, run_example):
    cfg_path = tmp_path / "test_config"
    run_example(
        "examples/app_generate_schema.py",
        ["--prefix", str(cfg_path)],
    )
    schema_path = cfg_path.with_suffix(".schema.json")
    with schema_path.open() as f:
        cfg = json.load(f)
    # Check that both args appear in the config
    assert "same_name_arg" in str(cfg["$defs"]["Model_Config"]["properties"])
    assert "same_name_arg" in str(cfg["$defs"]["Train_Config"]["properties"])
    # Check that the correct values are used during function call
    code, out, err = run_example(
        "examples/jsonargparse_app/train_jsonarg_params.py",
        ["--config", str(schema_path)],
    )
    assert code == 0, err
    print(out)
    assert "build_model [128, 128] 0.2 sum True root advanced_root same_name_arg_model" in out
    assert "train 10 32 adam same_name_arg_train" in out
