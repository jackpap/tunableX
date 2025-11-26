from __future__ import annotations

import json
from pathlib import Path

import pytest

# Updated tests for static AST analysis replacing runtime tracing.


@pytest.mark.skipif(pytest.importorskip("yaml") is None, reason="PyYAML not installed")
def test_analyze_generate_default_yaml(tmp_path, run_example):
    out_prefix = tmp_path / "train_defaults"
    code, _, err = run_example(
        "examples/trace_generate_schema.py",
        ["--entry", "train", "--prefix", str(out_prefix)],
    )
    assert code == 0, err
    schema_path = Path(str(out_prefix) + ".schema.json")
    json_path = Path(str(out_prefix) + ".json")
    assert schema_path.exists()
    assert json_path.exists()

    data = json.loads(json_path.read_text())
    assert "model" in data
    assert "train" in data
    assert "preprocess" in data["model"]
    assert "submodule_fun" in data["model"]["preprocess"]
    assert "submodule_class" in data["model"]["preprocess"]


@pytest.mark.skipif(pytest.importorskip("yaml") is None, reason="PyYAML not installed")
def test_analyze_generate_default_json_and_use(tmp_path, run_example):
    out_prefix = tmp_path / "train_cfg"
    code, _, err = run_example(
        "examples/trace_generate_schema.py",
        ["--entry", "train", "--prefix", str(out_prefix)],
    )
    assert code == 0, err

    cfg_json = Path(str(out_prefix) + ".json")
    code, _, err = run_example(
        "examples/argparse_trace/train_trace.py",
        ["--config", str(cfg_json)],
    )
    assert code == 0, err


@pytest.mark.skipif(pytest.importorskip("yaml") is None, reason="PyYAML not installed")
def test_analyze_generate_schema_only(tmp_path, run_example):
    out_prefix = tmp_path / "serve_cfg"
    code, _, err = run_example(
        "examples/trace_generate_schema.py",
        ["--entry", "serve", "--prefix", str(out_prefix)],
    )
    assert code == 0, err

    schema_path = Path(str(out_prefix) + ".schema.json")
    schema = json.loads(schema_path.read_text())
    assert schema.get("type") == "object"
    assert "properties" in schema
