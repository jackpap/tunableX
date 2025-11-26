from __future__ import annotations

# Test chaining two @tunable decorators to split parameters across namespaces and
# ensure help text (descriptions + defaults) is preserved for each parameter.
import pytest
from jsonargparse import ArgumentParser
from pydantic import Field

from tunablex import add_flags_by_app
from tunablex import build_cfg_from_file_and_args
from tunablex import tunable
from tunablex.context import use_config
from tunablex.runtime import make_config_for_app

pytestmark = pytest.mark.skipif(pytest.importorskip("jsonargparse") is None, reason="jsonargparse not installed")


@tunable("x", namespace="chain.alpha", apps=("chainapp",))
@tunable("y", "z", namespace="chain.beta", apps=("chainapp",))
def chained_func(
    x: int = Field(5, description="Alpha value"),
    y: float = Field(0.1, description="Learning rate"),
    z: str = Field("adam", description="Optimizer choice"),
):
    return x, y, z


def test_chained_tunable_decorators_help_output(capsys):
    p = ArgumentParser()
    p.add_argument("--config", help="optional config")
    add_flags_by_app(p, "chainapp")
    with pytest.raises(SystemExit):  # argparse exits after printing help
        p.parse_args(["--help"])  # triggers help
    out = capsys.readouterr().out
    assert "Alpha value" in out and "(default: 5)" in out
    assert "Learning rate" in out and "(default: 0.1)" in out
    assert "Optimizer choice" in out and "(default: 'adam')" in out


def test_chained_tunable_decorators_runtime_injection():
    AppConfig = make_config_for_app("chainapp")  # noqa: N806
    cfg = AppConfig()
    with use_config(cfg):
        vals = chained_func()
    assert vals == (5, 0.1, "adam")


def test_chained_tunable_decorators_cli_override(tmp_path):
    cfg_json = tmp_path / "cfg.json"
    cfg_json.write_text("{}")

    p = ArgumentParser()
    p.add_argument("--config")
    app_model = add_flags_by_app(p, "chainapp")

    args = p.parse_args([
        "--config",
        str(cfg_json),
        "--chain.alpha.x",
        "10",
        "--chain.beta.y",
        "0.01",
        "--chain.beta.z",
        "sgd",
    ])
    merged = build_cfg_from_file_and_args(app_model, args)
    assert merged["chain"]["alpha"]["x"] == 10
    assert merged["chain"]["beta"]["y"] == 0.01
    assert merged["chain"]["beta"]["z"] == "sgd"
