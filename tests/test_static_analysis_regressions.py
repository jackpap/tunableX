from __future__ import annotations

import importlib
import inspect
import json
import subprocess
import sys

import pytest

from tunablex import make_config_for_entry, schema_for_entrypoint, tunable
from tunablex.registry import REGISTRY, Node


@pytest.fixture(autouse=True)
def isolate_registry(monkeypatch):
    monkeypatch.setattr(REGISTRY, "entry_tree", Node(""))


def test_import_aliases_constructors_methods_and_name_collisions(tmp_path, monkeypatch):
    (tmp_path / "tx_left.py").write_text("""
from tunablex import tunable
@tunable(namespace="left")
def same(value: int = 1):
    raise AssertionError("must not execute")
class Worker:
    @tunable(namespace="worker")
    def __init__(self, size: int = 2):
        raise AssertionError("must not execute")
    @tunable(namespace="method")
    def execute(self, times: int = 3):
        raise AssertionError("must not execute")
""")
    (tmp_path / "tx_right.py").write_text("""
from tunablex import tunable
@tunable(namespace="right")
def same(value: int = 99):
    pass
""")
    (tmp_path / "tx_entry.py").write_text("""
import tx_left as lib
from tx_left import same as renamed
import tx_right

def entry():
    alias = renamed
    alias()
    worker = lib.Worker()
    worker.execute()
""")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        entry = importlib.import_module("tx_entry").entry
        assert make_config_for_entry(entry)().model_dump() == {
            "left": {"value": 1},
            "worker": {"size": 2},
            "method": {"times": 3},
        }
    finally:
        for name in ("tx_left", "tx_right", "tx_entry"):
            sys.modules.pop(name, None)


def test_nested_entry_closures_async_and_recursive_calls():
    @tunable(namespace="nested")
    def step(count: int = 5):
        pass

    def recursive():
        step()
        recursive()

    async def entry():
        recursive()

    assert schema_for_entrypoint(entry)[1] == {"nested": {"count": 5}}


def test_static_analysis_does_not_execute_properties_or_getattr():
    class Trap:
        @property
        def dangerous(self):
            pytest.fail("property executed during analysis")

        def __getattr__(self, name):
            pytest.fail("dynamic attribute executed during analysis")

    obj = Trap()

    def entry():
        obj.dangerous()
        obj.missing()

    assert make_config_for_entry(entry)().model_dump() == {}


def test_unavailable_root_source_has_actionable_error(monkeypatch):
    def entry():
        pass

    def unavailable(_):
        raise OSError("no source")

    monkeypatch.setattr(inspect, "getsource", unavailable)
    with pytest.raises(ValueError, match="app tags"):
        make_config_for_entry(entry)


def test_cli_analyze_json_output_and_invalid_entry(repo_root):
    result = subprocess.run(
        [sys.executable, "-m", "tunablex", "analyze", "--entry", "examples.myapp.pipeline:train_main"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    assert data["defaults"]["model"]["preprocess"]["submodule_class"]["attr1"] == 0
    invalid = subprocess.run(
        [sys.executable, "-m", "tunablex", "analyze", "--entry", "broken"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert invalid.returncode == 2
    assert "module:function" in invalid.stderr
