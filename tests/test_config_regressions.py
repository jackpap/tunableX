from __future__ import annotations

import argparse
import builtins
import json
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

import pytest
from pydantic import Field, PlainSerializer, ValidationError

from tunablex import add_flags_by_app, build_cfg_from_file_and_args, make_config_for_app, schema_for_app, tunable
from tunablex.io import load_structured_config
from tunablex.registry import REGISTRY, Node
from tunablex.runtime import write_schema


class Mode(str, Enum):
    FAST = "fast"
    SAFE = "safe"


@pytest.fixture(autouse=True)
def isolate_registry(monkeypatch):
    monkeypatch.setattr(REGISTRY, "entry_tree", Node(""))


@pytest.fixture(params=["argparse", "jsonargparse"])
def parser(request):
    if request.param == "jsonargparse":
        return pytest.importorskip("jsonargparse").ArgumentParser()
    return argparse.ArgumentParser()


def test_required_settings_can_come_from_file_and_cli(parser, tmp_path):
    @tunable("count", "path", namespace="job", apps="test")
    def run(count: int, path: Path):
        return count, path

    parser.add_argument("--config")
    model = add_flags_by_app(parser, "test")
    file = tmp_path / "config.json"
    file.write_text('{"job": {"count": 2, "path": "data"}}')
    args = parser.parse_args(["--config", str(file), "--job.count", "3"])
    assert build_cfg_from_file_and_args(model, args) == {"job": {"count": 3, "path": "data"}}
    schema, defaults = schema_for_app("test")
    assert defaults == {"job": {}}
    assert schema["required"] == ["job"]
    assert schema["$defs"]["Job_Config"]["required"] == ["count", "path"]
    with pytest.raises(ValidationError) as error:
        build_cfg_from_file_and_args(model, parser.parse_args([]))
    assert error.value.errors()[0]["loc"] == ("job",)


def test_cli_collections_literals_enums_nullable_and_double_underscores(parser, tmp_path):
    @tunable(namespace="group__name", apps="test")
    def run(
        values: list[int] = Field(default_factory=list),
        raw: list = Field(default_factory=list),
        pair: tuple[int, str] = (1, "a"),
        mapping: dict[str, int] = Field(default_factory=dict),
        number: Literal[1, 2] = 1,
        mode: Mode = Mode.FAST,
        optional: int | None = 7,
        enabled: bool = True,
    ):
        pass

    parser.add_argument("--config")
    model = add_flags_by_app(parser, "test")
    file = tmp_path / "cfg.json"
    file.write_text('{"group__name": {"optional": 12, "enabled": true}}')
    args = parser.parse_args(
        [
            "--config",
            str(file),
            "--group__name.values",
            "2",
            "3",
            "--group__name.raw",
            "a",
            "b",
            "--group__name.pair",
            '[4,"b"]',
            "--group__name.mapping",
            '{"x":5}',
            "--group__name.number",
            "2",
            "--group__name.mode",
            "safe",
            "--group__name.optional",
            "null",
            "--no-group__name.enabled",
        ]
    )
    data = build_cfg_from_file_and_args(model, args)["group__name"]
    assert data == dict(
        values=[2, 3],
        raw=["a", "b"],
        pair=[4, "b"],
        mapping={"x": 5},
        number=2,
        mode="safe",
        optional=None,
        enabled=False,
    )


def test_defaults_validate_and_unknown_fields_are_rejected():
    @tunable(apps="test")
    def run(value: int = Field(0, gt=0)):
        pass

    model = make_config_for_app("test")
    with pytest.raises(ValidationError):
        model()
    with pytest.raises(ValidationError, match="extra_forbidden"):
        model(value=1, typo=1)


def test_equivalent_fields_merge_and_conflicts_show_both_values():
    @tunable(apps="test")
    def one(value: int = Field(1, ge=0, description="Count")):
        pass

    @tunable(apps="test")
    def two(value: int = Field(1, ge=0, description="Count")):
        pass

    assert make_config_for_app("test")().value == 1
    with pytest.raises(ValueError, match="Conflicting type.*int.*str"):

        @tunable(apps="test")
        def wrong(value: str = "1"):
            pass


@pytest.mark.parametrize("first,second", [("group", "group.value"), ("group.value", "group")])
def test_field_namespace_collisions_rejected(first, second):
    def register(path):
        ns, _, name = path.rpartition(".")
        from tunablex.registry import TunableArg

        REGISTRY.register(TunableArg(name, int, 1, {"example"}, ns, {"test"}))

    register(first)
    with pytest.raises(ValueError, match="collides"):
        register(second)


@pytest.mark.parametrize("namespace", ["bad..path", "_hidden", "model_dump", "class"])
def test_invalid_namespace_rejected(namespace):
    with pytest.raises(ValueError, match="Invalid or reserved"):
        tunable(namespace=namespace)(lambda value=1: value)


@pytest.mark.parametrize("suffix,content", [("json", "[]"), ("json", "null"), ("yaml", "false"), ("yaml", "- 1")])
def test_nonmapping_config_rejected(tmp_path, suffix, content):
    if suffix == "yaml":
        pytest.importorskip("yaml")
    file = tmp_path / f"config.{suffix}"
    file.write_text(content)
    with pytest.raises(ValueError, match="object/mapping"):
        load_structured_config(file)


def test_toml_loading_and_utf8_schema_export(tmp_path):
    file = tmp_path / "config.toml"
    file.write_text('[job]\nname = "échantillon"\n', encoding="utf-8")
    assert load_structured_config(file) == {"job": {"name": "échantillon"}}
    prefix = tmp_path / "nested" / "config"
    write_schema(prefix, {"title": "échantillon"}, {"value": 2}, yaml=False)
    assert json.loads(prefix.with_suffix(".schema.json").read_text())["title"] == "échantillon"
    assert not prefix.with_suffix(".yml").exists()


def test_json_export_does_not_require_yaml(monkeypatch, tmp_path):
    original = builtins.__import__

    def without_yaml(name, *args, **kwargs):
        if name == "yaml":
            raise ModuleNotFoundError("No module named yaml")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_yaml)
    write_schema(tmp_path / "config", {}, {"x": 1})
    assert load_structured_config(tmp_path / "config.json") == {"x": 1}
    with pytest.raises(RuntimeError, match="tunablex\\[yaml\\]"):
        write_schema(tmp_path / "explicit", {}, {}, yaml=True)


def test_data_dependent_factory_runs_after_overrides(parser):
    @tunable(apps="test")
    def run(count: int = 2, doubled: int = Field(default_factory=lambda data: data["count"] * 2)):
        pass

    model = add_flags_by_app(parser, "test")
    assert build_cfg_from_file_and_args(model, parser.parse_args(["--count", "7"])) == {"count": 7, "doubled": 14}


def test_cli_can_clear_collection_and_nullable_string(parser):
    @tunable(apps="test")
    def run(values: list[int] = Field(default_factory=lambda: [1]), name: str | None = "default"):
        pass

    model = add_flags_by_app(parser, "test")
    args = parser.parse_args(["--values", "--name", "null"])
    assert build_cfg_from_file_and_args(model, args) == {"values": [], "name": None}


def test_exported_defaults_use_field_validation_and_serialization():
    @tunable("required", "count", "encoded", "hidden", namespace="job", apps="export")
    def run(
        required: str,
        count: int = Field(default_factory=lambda: "2", ge=1),
        encoded: Annotated[int, PlainSerializer(str, return_type=str)] = "3",
        hidden: str = Field("private", exclude=True),
    ):
        pass

    _, defaults = schema_for_app("export")
    cfg = make_config_for_app("export").model_validate({"job": {"required": "provided"}})
    expected = cfg.model_dump(mode="json")
    del expected["job"]["required"]
    assert defaults == expected == {"job": {"count": 2, "encoded": "3"}}


def test_defaults_export_rejects_invalid_field_defaults():
    @tunable(apps="export")
    def run(value: int = Field(-1, ge=1)):
        pass

    with pytest.raises(ValidationError, match="greater_than_equal"):
        schema_for_app("export")
