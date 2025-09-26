"""CLI helpers to expose tunables as jsonargparse flags and build configs.

- Adds flags for nested namespaces using dotted paths (e.g., --model.preprocess.toto.dropna).
- Builds overrides dict matching the nested JSON structure.
"""

from __future__ import annotations

from argparse import SUPPRESS
from argparse import BooleanOptionalAction
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Literal
from typing import get_args
from typing import get_origin

from pydantic import BaseModel

from .io import load_structured_config
from .runtime import make_app_config_for
from .runtime import make_app_config_for_entry

if TYPE_CHECKING:
    from jsonargparse import ArgumentParser


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


def add_flags_from_model(parser: ArgumentParser, app_config_model) -> None:
    """Create flags like --section.field for each tunable in the AppConfig model.

    Recurses into nested BaseModel fields to support compounded namespaces.
    Parser defaults are SUPPRESS so we can detect presence via hasattr(args, dest).
    Actual defaults still come from the Pydantic model instance when building the config.
    """

    def is_model_type(ann) -> bool:
        return isinstance(ann, type) and issubclass(ann, BaseModel)

    def add_section_flags(display_prefix: str, dest_prefix: str, model_type: type[BaseModel]) -> None:
        grp = parser.add_argument_group(display_prefix)
        for name, fld in model_type.model_fields.items():
            ann = fld.annotation
            # Recurse into nested models
            if is_model_type(ann):
                add_section_flags(f"{display_prefix}.{name}", f"{dest_prefix}__{name}", ann)
                continue
            help_text = _help_with_default(fld)
            flag = f"--{display_prefix}.{name}"
            dest = f"TX__{dest_prefix}__{name}"
            if get_origin(ann) is Literal:
                grp.add_argument(flag, choices=[*get_args(ann)], dest=dest, help=help_text, default=SUPPRESS)
            elif ann is bool:
                grp.add_argument(flag, action=BooleanOptionalAction, dest=dest, help=help_text, default=SUPPRESS)
            elif ann in (int, float, str):
                grp.add_argument(flag, type=ann, dest=dest, help=help_text, default=SUPPRESS)
            elif ann is Path:
                grp.add_argument(flag, type=str, dest=dest, help=help_text, default=SUPPRESS)
            else:
                grp.add_argument(flag, type=str, dest=dest, help=help_text, default=SUPPRESS)

    # Handle root-level fields (non-BaseModel fields)
    root_grp = None
    for field_name, field_info in app_config_model.model_fields.items():
        field_type = field_info.annotation
        if isinstance(field_type, type) and issubclass(field_type, BaseModel):
            # This is a nested section - handle it below
            continue
        
        # This is a root-level field - add it directly
        if root_grp is None:
            root_grp = parser.add_argument_group("root parameters")
        
        help_text = _help_with_default(field_info)
        flag = f"--{field_name}"
        dest = f"TX__ROOT__{field_name}"
        
        if get_origin(field_type) is Literal:
            root_grp.add_argument(flag, choices=[*get_args(field_type)], dest=dest, help=help_text, default=SUPPRESS)
        elif field_type is bool:
            root_grp.add_argument(flag, action=BooleanOptionalAction, dest=dest, help=help_text, default=SUPPRESS)
        elif field_type in (int, float, str):
            root_grp.add_argument(flag, type=field_type, dest=dest, help=help_text, default=SUPPRESS)
        elif field_type is Path:
            root_grp.add_argument(flag, type=str, dest=dest, help=help_text, default=SUPPRESS)
        else:
            root_grp.add_argument(flag, type=str, dest=dest, help=help_text, default=SUPPRESS)

    # Handle nested sections (BaseModel fields)
    for section_name, section_field in app_config_model.model_fields.items():
        section_model = section_field.annotation
        if not (isinstance(section_model, type) and issubclass(section_model, BaseModel)):
            continue
        display_section = section_name.replace("__", ".")
        add_section_flags(display_section, section_name, section_model)


def add_flags_by_app(parser: ArgumentParser, app: str):
    """Add flags for all tunables tagged with the given app and return the AppConfig model."""
    app_config_model = make_app_config_for(app)
    add_flags_from_model(parser, app_config_model)
    return app_config_model


def add_flags_by_entry(parser: ArgumentParser, entrypoint, *args, **kwargs) -> BaseModel:  # type: ignore[override]
    """Add flags discovered via static (AST) analysis of the entrypoint call graph.

    Returns the generated AppConfig model (same as add_flags_by_app).
    """
    app_config_model = make_app_config_for_entry(entrypoint, *args, **kwargs)
    add_flags_from_model(parser, app_config_model)
    return app_config_model


# Backwards compatibility alias: previously tracing; now static analysis underneath
def add_flags_by_trace(parser: ArgumentParser, entrypoint, *args, **kwargs):  # noqa: D401
    """Alias for add_flags_by_entry kept for backwards compatibility."""
    return add_flags_by_entry(parser, entrypoint, *args, **kwargs)


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

    # Handle root-level fields first
    for field_name, field_info in app_config_model.model_fields.items():
        field_type = field_info.annotation
        if isinstance(field_type, type) and issubclass(field_type, BaseModel):
            # This is a nested section - handle it below
            continue
        
        # This is a root-level field
        dest = f"TX__ROOT__{field_name}"
        if hasattr(args, dest):
            val = getattr(args, dest)
            if val is not None:
                overrides[field_name] = val

    # Handle nested sections
    for section_name, section_field in app_config_model.model_fields.items():
        section_model = section_field.annotation
        if not (isinstance(section_model, type) and issubclass(section_model, BaseModel)):
            continue
        walk_section(section_name, [section_name.replace("__", "_")], section_model)

    return overrides


def build_cfg_from_file_and_args(app_config_model, args, config_attr: str = "config") -> dict:
    """Merge defaults <- file (optional) <- CLI flags into a nested config dict."""
    cfg = app_config_model().model_dump(mode="json")
    path = getattr(args, config_attr, None)
    if path:
        cfg = deep_update(cfg, load_structured_config(path))
    overrides = collect_overrides(args, app_config_model)
    return deep_update(cfg, overrides)
