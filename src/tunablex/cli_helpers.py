"""CLI helpers to expose tunables as jsonargparse flags and build configs.

- Adds flags for nested namespaces using dotted paths (e.g., --model.preprocess.toto.dropna).
- Builds overrides dict matching the nested JSON structure.
"""

from __future__ import annotations

from argparse import SUPPRESS
from argparse import BooleanOptionalAction
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Literal
from typing import get_args
from typing import get_origin

from pydantic import BaseModel
from pydantic import Field

from .io import load_structured_config
from .runtime import make_config_for_app
from .runtime import make_config_for_entry

if TYPE_CHECKING:
    from jsonargparse import ArgumentParser
    from jsonargparse._core import ArgumentGroup


def _help_with_default(fld) -> str | None:
    desc = getattr(fld, "description", None)
    if fld.is_required():
        return f"{desc} (required)" if desc else "(required)"
    default_val = fld.default
    if isinstance(default_val, bool):
        default_str = str(default_val).lower()
    elif isinstance(default_val, Path):
        default_str = str(default_val)
    else:
        default_str = repr(default_val)
    if desc:
        return f"{desc} (default: {default_str})"
    return f"(default: {default_str})"


def _is_model_type(ann) -> bool:
    return isinstance(ann, type) and issubclass(ann, BaseModel)


def _add_field_flag(name: str, ann: type, field: Field, grp: ArgumentGroup | ArgumentParser):
    help_text = _help_with_default(field)
    flag = f"--{name}"
    dest = f"TX__{name.replace('.', '__')}"
    if get_origin(ann) is Literal:
        grp.add_argument(flag, choices=[*get_args(ann)], dest=dest, help=help_text, default=SUPPRESS)
    elif get_origin(ann) is Sequence or ann in (list, tuple):
        grp.add_argument(flag, nargs="+", type=get_args(ann)[0], dest=dest, help=help_text, default=SUPPRESS)
    elif ann is bool:
        grp.add_argument(flag, action=BooleanOptionalAction, dest=dest, help=help_text, default=SUPPRESS)
    elif ann in (int, float, str, Path):
        grp.add_argument(flag, type=ann, dest=dest, help=help_text, default=SUPPRESS)
    else:
        grp.add_argument(flag, type=str, dest=dest, help=help_text, default=SUPPRESS)


def _add_section_flags(section_name: str, model_type: type[BaseModel], parser: ArgumentParser) -> None:
    grp = parser.add_argument_group(section_name)
    for name, field in model_type.model_fields.items():
        ann = field.annotation
        # Recurse into nested models
        if _is_model_type(ann):
            _add_section_flags(f"{section_name}.{name}", model_type=ann, parser=parser)
        else:
            _add_field_flag(name=f"{section_name}.{name}", ann=ann, field=field, grp=grp)


def add_flags_from_model(parser: ArgumentParser, app_config_model: type[BaseModel]) -> None:
    """Create flags like --section.field for each tunable in the AppConfig model.

    Recurses into nested BaseModel fields to support compounded namespaces.
    Parser defaults are SUPPRESS so we can detect presence via hasattr(args, dest).
    Actual defaults still come from the Pydantic model instance when building the config.
    """
    for section_name, section_field in app_config_model.model_fields.items():
        ann = section_field.annotation
        if _is_model_type(ann):
            _add_section_flags(section_name, model_type=ann, parser=parser)
        else:  # Root level
            _add_field_flag(name=section_name, ann=ann, field=section_field, grp=parser)


def add_flags_by_app(parser: ArgumentParser, app: str):
    """Add flags for all tunables tagged with the given app and return the AppConfig model."""
    app_config_model = make_config_for_app(app)
    add_flags_from_model(parser, app_config_model)
    return app_config_model


def add_flags_by_entry(parser: ArgumentParser, entrypoint) -> BaseModel:  # type: ignore[override]
    """Add flags discovered via static (AST) analysis of the entrypoint call graph.

    Returns the generated AppConfig model (same as add_flags_by_app).
    """
    app_config_model = make_config_for_entry(entrypoint)
    add_flags_from_model(parser, app_config_model)
    return app_config_model


def deep_update(base: dict, extra: dict) -> dict:
    """Recursively merge extra into base (in place) and return base."""
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def collect_overrides(args, app_config_model) -> dict:
    """Collect provided CLI flags into a nested overrides dict matching AppConfig."""

    def is_model_type(ann) -> bool:
        return isinstance(ann, type) and issubclass(ann, BaseModel)

    overrides: dict = {}

    def assign_path(path: list[str], name: str, value) -> None:
        cur = overrides
        for p in path:
            cur = cur.setdefault(p, {})
        cur[name] = value

    def walk_section(dest_prefix: str, node_path: list[str], model_type: type[BaseModel]) -> None:
        for name, fld in model_type.model_fields.items():
            ann = fld.annotation
            if is_model_type(ann):
                walk_section(f"{dest_prefix}__{name}", [*node_path, name], ann)
                continue
            dest = f"TX__{dest_prefix}__{name}"
            if hasattr(args, dest):
                val = getattr(args, dest)
                if val is not None:
                    assign_path(node_path, name, val)

    for section_name, section_field in app_config_model.model_fields.items():
        section_model = section_field.annotation
        if is_model_type(section_model):
            walk_section(section_name, [section_name.replace("__", "_")], section_model)
        else:  # Root level
            dest = f"TX__{section_name}"
            if hasattr(args, dest) and (val := getattr(args, dest)) is not None:
                overrides[section_name] = val

    return overrides


def build_cfg_from_file_and_args(app_config_model, args, config_attr: str = "config") -> dict:
    """Merge defaults <- file (optional) <- CLI flags into a nested config dict."""
    cfg = app_config_model().model_dump(mode="json")
    path = getattr(args, config_attr, None)
    if path:
        cfg = deep_update(cfg, load_structured_config(path))
    overrides = collect_overrides(args, app_config_model)
    return deep_update(cfg, overrides)
