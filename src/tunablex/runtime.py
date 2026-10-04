"""Public model composition, configuration loading and schema export."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field, create_model

from .io import load_structured_config
from .registry import REGISTRY, _ConfigModel


def _defaults_model(model: type[BaseModel]) -> type[BaseModel]:
    """Retain field schemas while omitting values that require user input."""
    fields = {}
    for name, field in model.model_fields.items():
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, _ConfigModel):
            fields[name] = (_defaults_model(annotation), Field(default_factory=dict))
        elif not field.is_required() and not field.default_factory_takes_validated_data:
            fields[name] = (annotation, field)
    return create_model(f"{model.__name__}_Defaults", __config__=model.model_config, **fields)


def _defaults(model: type[BaseModel]) -> dict:
    """Validate and serialize a partial template using each field's Pydantic schema."""
    return _defaults_model(model)().model_dump(mode="json")


def schema_for_app(app: str) -> tuple[dict, dict]:
    """Return JSON Schema and a defaults template (required values are omitted)."""
    model = make_config_for_app(app)
    return model.model_json_schema(), _defaults(model)


def schema_for_entrypoint(entrypoint: Callable) -> tuple[dict, dict]:
    """Return schema and defaults for the statically discovered call graph."""
    model = make_config_for_entry(entrypoint)
    return model.model_json_schema(), _defaults(model)


def write_schema(prefix: str | Path, schema: dict, defaults: dict | None = None, *, yaml: bool | None = None):
    """Write UTF-8 JSON files; optionally also YAML defaults.

    yaml=None retains automatic YAML output when PyYAML is installed; False
    disables it; True requires the optional dependency. Parent folders are created.
    """
    yaml_module = None
    if defaults is not None and yaml is not False:
        try:
            import yaml as yaml_module
        except ModuleNotFoundError:
            if yaml:
                raise RuntimeError("YAML support requires: pip install 'tunablex[yaml]'") from None
    path = Path(f"{prefix}.schema.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if defaults is not None:
        Path(f"{prefix}.json").write_text(json.dumps(defaults, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if yaml_module is not None:
            Path(f"{prefix}.yml").write_text(
                yaml_module.safe_dump(defaults, sort_keys=False, allow_unicode=True), encoding="utf-8"
            )


def make_config_for_app(app: str) -> type[BaseModel]:
    """Compose fields tagged with app, plus all untagged fields."""
    return REGISTRY.build_config_for_app(app)


def load_config_for_app(app: str, json_path: str | Path) -> BaseModel:
    """Load JSON/YAML/TOML and validate; raise Pydantic ValidationError on invalid input."""
    return make_config_for_app(app).model_validate(load_structured_config(json_path))


def make_config_for_entry(entrypoint: Callable) -> type[BaseModel]:
    """Compose fields from resolved calls without executing application functions."""
    return REGISTRY.build_config_for_entrypoint(entrypoint)


def load_config_for_entry(entrypoint: Callable, json_path: str | Path) -> BaseModel:
    """Load and validate an entrypoint configuration."""
    return make_config_for_entry(entrypoint).model_validate(load_structured_config(json_path))
