"""Dotted configuration flags for argparse and jsonargparse."""

from __future__ import annotations

from argparse import SUPPRESS, ArgumentTypeError, BooleanOptionalAction
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, get_args, get_origin

from pydantic import BaseModel, TypeAdapter, ValidationError

from .io import load_structured_config
from .runtime import make_config_for_app, make_config_for_entry


def _help_with_default(field) -> str:
    if field.is_required():
        detail = "required"
    elif field.default_factory is not None:
        detail = "default: factory"
    else:
        value = field.default
        display = (
            str(value).lower() if isinstance(value, bool) else str(value) if isinstance(value, Path) else repr(value)
        )
        detail = f"default: {display}"
    return f"{field.description} ({detail})" if field.description else f"({detail})"


def _is_model_type(annotation) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _dest(path: tuple[str, ...]) -> str:
    # Dots are valid argparse destinations and preserve names containing '__'.
    return "TX__" + ".".join(path)


def _leaves(model, path=(), ancestors=()):
    for name, field in model.model_fields.items():
        current = (*path, name)
        annotation = field.annotation
        if _is_model_type(annotation) and annotation not in ancestors:
            yield from _leaves(annotation, current, (*ancestors, model))
        else:
            yield current, field


def _convert(annotation):
    adapter = TypeAdapter(annotation)

    def parse(text):
        if text == "null" and type(None) in get_args(annotation):
            return None
        # Prefer raw strings for string fields, including string-valued Literals/Enums.
        try:
            return adapter.validate_python(text)
        except ValidationError:
            try:
                return adapter.validate_json(text)
            except ValidationError as exc:
                raise ArgumentTypeError(str(exc)) from exc

    return parse


def _add_field_flag(path, field, group):
    annotation = field.annotation
    flag = "--" + ".".join(path)
    kwargs = {"dest": _dest(path), "help": _help_with_default(field), "default": SUPPRESS}
    origin, args = get_origin(annotation), get_args(annotation)
    if annotation is bool:
        kwargs["action"] = BooleanOptionalAction
    elif (
        origin in (list, set, frozenset, Sequence)
        or annotation in (list, tuple, set, frozenset)
        or (origin is tuple and len(args) == 2 and args[1] is Ellipsis)
    ):
        kwargs.update(nargs="*", type=_convert(args[0] if args else str))
    else:
        kwargs["type"] = _convert(annotation)
        if origin is Literal:
            kwargs["choices"] = list(args)
    group.add_argument(flag, **kwargs)


def add_flags_from_model(parser, app_config_model: type[BaseModel]) -> None:
    """Add typed dotted flags. Required values may come from either file or CLI.

    Lists/Sequences use space-separated values. Mappings, fixed tuples, optional
    models and other structured values accept a JSON token. Validation is completed
    after merging, so defaults never overwrite file values.
    """
    groups = {}
    for path, field in _leaves(app_config_model):
        if len(path) > 1:
            key = ".".join(path[:-1])
            if key not in groups:
                groups[key] = parser.add_argument_group(key)
            group = groups[key]
        else:
            group = parser
        _add_field_flag(path, field, group)


def add_flags_by_app(parser, app: str) -> type[BaseModel]:
    """Add fields selected by app tag and return their config model type."""
    model = make_config_for_app(app)
    add_flags_from_model(parser, model)
    return model


def add_flags_by_entry(parser, entrypoint) -> type[BaseModel]:
    """Add fields selected through static entrypoint analysis and return their model."""
    model = make_config_for_entry(entrypoint)
    add_flags_from_model(parser, model)
    return model


def deep_update(base: dict, extra: dict) -> dict:
    """Recursively merge extra into base in place; lists and scalars replace."""
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def collect_overrides(args, app_config_model) -> dict:
    """Collect only supplied flags, preserving explicit null values."""
    overrides = {}
    # jsonargparse namespaces provide as_dict() and nest dotted destinations.
    values = args.as_dict() if hasattr(args, "as_dict") else vars(args)
    for path, _ in _leaves(app_config_model):
        destination = _dest(path)
        if destination in values:
            value = values[destination]
        else:
            current = values
            for part in destination.split("."):
                if not isinstance(current, dict) or part not in current:
                    break
                current = current[part]
            else:
                value = current
                cursor = overrides
                for part in path[:-1]:
                    cursor = cursor.setdefault(part, {})
                cursor[path[-1]] = value
                continue
            continue
        cursor = overrides
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[path[-1]] = value
    return overrides


def build_cfg_from_file_and_args(app_config_model, args, config_attr: str = "config") -> dict:
    """Merge file <- explicit flags, then apply and validate defaults once.

    Returns JSON-compatible values. Invalid/unknown fields raise ValidationError.
    """
    path = getattr(args, config_attr, None)
    data = load_structured_config(path) if path else {}
    deep_update(data, collect_overrides(args, app_config_model))
    return app_config_model.model_validate(data).model_dump(mode="json")
