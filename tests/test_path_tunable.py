from __future__ import annotations

import json

import pytest

# Test that a tunable Path parameter serializes correctly in defaults and can be overridden.


@pytest.mark.skipif(pytest.importorskip("jsonargparse") is None, reason="jsonargparse not installed")
def test_path_tunable_defaults_and_override(tmp_path, repo_root, run_example):
    # Dynamically create a small example with a Path tunable
    example_dir = tmp_path / "path_example"
    example_dir.mkdir()
    (example_dir / "__init__.py").write_text("")
    code = """
from pathlib import Path
from pydantic import Field
from tunablex import tunable, use_config
from tunablex.runtime import make_config_for_app
from tunablex.cli_helpers import add_flags_by_app, build_cfg_from_file_and_args

@tunable("ckpt_path", namespace="train.path", apps=("train",))
def step(ckpt_path: Path = Field(Path("ckpt"), description="Path to checkpoint directory")):
    print("STEP", ckpt_path)

if __name__ == "__main__":
    APP = "train"
    AppConfig = make_config_for_app(APP)
    from jsonargparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument("--config", help="optional config")
    add_flags_by_app(p, APP)
    args = p.parse_args()
    cfg_dict = build_cfg_from_file_and_args(AppConfig, args)
    cfg = AppConfig.model_validate(cfg_dict)
    with use_config(cfg):
        step()
"""
    mod_path = example_dir / "path_app.py"
    mod_path.write_text(code)

    # Determine script path (relative if possible, else absolute)
    try:
        script_path = str(mod_path.relative_to(repo_root))
    except ValueError:
        script_path = str(mod_path)

    # 1. Check help contains default path as string
    rc, out, err = run_example(script_path, ["--help"])
    assert rc == 0, err
    assert "Path to checkpoint directory" in out
    assert "(default: 'ckpt')" in out or "(default: ckpt)" in out

    # 2. Run without overrides -> default path
    rc, out, err = run_example(script_path, [])
    assert rc == 0, err
    assert "STEP ckpt" in out

    # 3. Provide override
    rc, out, err = run_example(script_path, ["--train.path.ckpt_path", str(tmp_path / "newckpt")])
    assert rc == 0, err
    assert f"STEP {tmp_path / 'newckpt'}" in out

    # 4. Provide via config file
    cfg_file = tmp_path / "cfg.json"
    cfg_file.write_text(json.dumps({"train": {"path": {"ckpt_path": str(tmp_path / "cfgckpt")}}}))
    rc, out, err = run_example(script_path, ["--config", str(cfg_file)])
    assert rc == 0, err
    assert f"STEP {tmp_path / 'cfgckpt'}" in out
