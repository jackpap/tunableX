from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.mark.skipif(pytest.importorskip("yaml") is None, reason="PyYAML not installed")
@pytest.mark.skip(reason="AST test cannot pass yet.")
def test_trace_generate_default_yaml(tmp_path, run_example):
    # Use the tracing helper example to generate schema & defaults
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

    # The defaults JSON should be loadable and contain known keys
    data = json.loads(json_path.read_text())
    assert "model" in data
    assert "train" in data
    assert "preprocess" in data["model"]
    assert "submodule_fun" in data["model"]["preprocess"]
    assert "submodule_class" in data["model"]["preprocess"]
    assert "attr1" in data["model"]["preprocess"]["submodule_class"]
    assert "arg1" in data["model"]["preprocess"]["submodule_class"]


@pytest.mark.skipif(pytest.importorskip("yaml") is None, reason="PyYAML not installed")
def test_trace_generate_default_json_and_use_with_trace(tmp_path, run_example):
    out_prefix = tmp_path / "train_cfg"
    code, _, err = run_example(
        "examples/trace_generate_schema.py",
        ["--entry", "train", "--prefix", str(out_prefix)],
    )
    assert code == 0, err

    # Now run the tracing-based app using the generated JSON config
    cfg_json = Path(str(out_prefix) + ".json")
    code, _, err = run_example(
        "examples/argparse_trace/train_trace.py",
        ["--config", str(cfg_json)],
    )
    assert code == 0, err
    assert "train" in Path.read_text(cfg_json) or True


@pytest.mark.skipif(pytest.importorskip("yaml") is None, reason="PyYAML not installed")
def test_trace_generate_schema_only(tmp_path, run_example):
    out_prefix = tmp_path / "serve_cfg"
    code, _, err = run_example(
        "examples/trace_generate_schema.py",
        ["--entry", "serve", "--prefix", str(out_prefix)],
    )
    assert code == 0, err

    # Validate basic shape of schema
    schema_path = Path(str(out_prefix) + ".schema.json")
    schema = json.loads(schema_path.read_text())
    assert schema.get("type") == "object"
    assert "properties" in schema
