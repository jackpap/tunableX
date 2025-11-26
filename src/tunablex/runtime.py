from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import ValidationError

from .io import load_structured_config
from .registry import REGISTRY

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic import BaseModel


def schema_for_app(app: str) -> tuple[dict, dict]:
    AppConfig = REGISTRY.build_config_for_app(app)
    return AppConfig.model_json_schema(), AppConfig().model_dump(mode="json")


def schema_for_entrypoint(entrypoint: Callable) -> tuple[dict, dict]:
    AppConfig = REGISTRY.build_config_for_entrypoint(entrypoint)
    return AppConfig.model_json_schema(), AppConfig().model_dump(mode="json")


def write_schema(prefix: str, schema: dict, defaults: dict | None = None):
    Path(f"{prefix}.schema.json").write_text(json.dumps(schema, indent=2, default=str))
    if defaults is not None:
        Path(f"{prefix}.json").write_text(json.dumps(defaults, indent=2, default=str))
        with Path(f"{prefix}.yml").open("w") as f:
            yaml.dump(defaults, f, default_flow_style=False, sort_keys=False)


def make_config_for_app(app: str) -> type[BaseModel]:
    return REGISTRY.build_config_for_app(app)


def load_config_for_app(app: str, json_path: str | Path):
    AppConfig = REGISTRY.build_config_for_app(app)
    data = load_structured_config(json_path)
    try:
        return AppConfig.model_validate(data)
    except ValidationError as e:
        msg = f"Invalid config for app '{app}':\n{e}".rstrip()
        raise SystemExit(msg) from None


def make_config_for_entry(entrypoint: Callable) -> type[BaseModel]:
    return REGISTRY.build_config_for_entrypoint(entrypoint)


def load_config_for_entry(entrypoint: Callable, json_path: str | Path):
    AppConfig = REGISTRY.build_config_for_entrypoint(entrypoint)
    data = load_structured_config(json_path)
    try:
        return AppConfig.model_validate(data)
    except ValidationError as e:
        msg = f"Invalid config for entrypoint (AST):\n{e}".rstrip()
        raise SystemExit(msg) from None
