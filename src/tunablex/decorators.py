"""Decorator to declare tunable function parameters and auto-inject config.

Wraps functions, registers a Pydantic model per namespace, and injects values
from the active AppConfig at call time. Supports dotted namespaces.
"""

from __future__ import annotations

import functools
import inspect
import re
import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import get_type_hints

from pydantic.fields import FieldInfo

from .context import _active_cfg
from .registry import REGISTRY
from .registry import TunableArg

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pydantic import BaseModel


def _pascalcase_to_snake_case(ns: str) -> str:
    """Convert a namespace name from PascalCase to snake_case."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", ns).lower()


class TunableParamsMeta(type):
    """A metaclass that allows to retrieve namespace and type annotation at runtime."""

    def __init__(cls, name, bases, attrs):  # noqa: D107
        super().__init__(name, bases, attrs)
        cls.namespace = None

    @staticmethod
    def _process_name(name: str) -> str:
        """Process a class name to turn it into a namespace."""
        name = _pascalcase_to_snake_case(name).replace("_params", "")
        if name == "main" or name == "root":
            name = ""
        return name

    def __getattribute__(cls, name: str) -> Any | tuple[Any, str, str, str]:
        """Override that returns additional informations when the attribute is a FieldInfo.

        Store the classes that are accessed and use them to build the final namespace.
        """
        if super().__getattribute__("namespace") is None:
            cls.namespace = TunableParamsMeta._process_name(super().__getattribute__("__name__"))

        if isinstance(value := super().__getattribute__(name), FieldInfo):
            globalsns = vars(sys.modules[cls.__module__])
            typ = get_type_hints(cls, globalns=globalsns).get(name, Any)
            return value, typ, cls.namespace, name

        # If value is a class with this metaclass, update its parent namespace
        if isinstance(value, type) and isinstance(value, TunableParamsMeta):
            value.namespace = f"{cls.namespace}.{TunableParamsMeta._process_name(name)}"

        return value


class TunableParams(metaclass=TunableParamsMeta):
    """A class containing tunable parameters.

    Inherit from this class to declare tunable parameters globally.
    If the class name contains `Params`, it will be removed from the namespace for brevity.
    If the resulting namespace is `main` or `root`, the parameters will be stored at the root level.

    When using several levels of namespaces, it is possible to declare the parameters in a class at the root level
    and to reference this class in the namespace, to avoid having too many indentations in the lower levels.

    Example:
        # This is root level
        class AdvancedParams(TunableParams):
            param1: ...
            param2: ...

        class GeneralParams(TunableParams):
            Advanced = AdvancedParams

    In this case, the namespace for param1 and param2 is `general.advanced`.
    """


def _resolve_nested_section(cfg_model: BaseModel, dotted_ns: str):
    if not dotted_ns:  # main namespace
        return cfg_model
    obj = cfg_model
    for seg in dotted_ns.split("."):
        if obj is None or not hasattr(obj, seg):
            return None
        obj = getattr(obj, seg)
    return obj


def tunable(
    *include: str,
    namespace: str = "",
    exclude: str | Iterable[str] = (),
    apps: str | Iterable[str] = (),
):
    """Mark a function's selected parameters as user-tunable.

    - include: names to include. If empty, include all params that have defaults
      (unless mode='exclude' with an explicit exclude list).
    - namespace: JSON section name; defaults to an empty namespace.
    - apps: optional tags to group functions per executable/app.
    """
    include_set = set(include or ())
    exclude_set = {exclude} if isinstance(exclude, str) else set(exclude)
    if include_set and exclude_set:
        msg = "Cannot pass both `include` and `exclude` arguments."
        raise ValueError(msg)
    apps = {apps} if isinstance(apps, str) else set(apps)

    def decorator(fn):
        sig = inspect.signature(fn)
        ns = namespace
        namespaces = set()
        ref_names = {}
        for name, p in sig.parameters.items():
            if name == "mro":
                msg = "`mro` is a protected name, please use an other name for your tunable parameters."
                raise ValueError(msg)
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            if include_set:
                selected = name in include_set
            elif exclude_set:
                selected = (p.default is not inspect._empty) and (name not in exclude_set)
            else:
                selected = p.default is not inspect._empty
            if not selected:
                continue
            default = p.default if p.default is not inspect._empty else ...
            if isinstance(default, tuple) and isinstance(default[0], FieldInfo):
                # The parameter is declared in a TunableParam class; retrieve type, namespace and reference name
                default, typ, ns, ref_name = default
                if ref_name != name:
                    # Store the reference name for later look-up
                    # This allows to have different local names for the same global parameter
                    ref_names[name] = ref_name
                    name = ref_name
            else:
                typ = inspect.get_annotations(fn, eval_str=False)[name]
                typ = eval(typ, fn.__globals__) if isinstance(typ, str) else typ
            namespaces.add(ns)
            REGISTRY.register(
                TunableArg(
                    name=name,
                    typ=typ,
                    default=default,
                    namespace=ns,
                    fn_names={fn.__qualname__, f"{fn.__module__}.{fn.__qualname__}"},
                    apps=set(apps),
                )
            )

        @functools.wraps(fn)
        def wrapper(*args, cfg: BaseModel | dict | None = None, **kwargs):
            # Handle static methods called from instances
            if isinstance(fn, staticmethod) and args[0].__class__.__name__ == fn.__qualname__.split(".")[0]:
                args = args[1:]
            if cfg is not None:
                data = cfg if isinstance(cfg, dict) else cfg.model_dump()
                filtered = {
                    # Get the tunable arguments from the config and retrieve the original name
                    k: data[ref_names.get(k, k)]
                    for k in sig.parameters
                    if ref_names.get(k, k) in data and k not in kwargs
                }
                return fn(*args, **filtered, **kwargs)

            cfg = _active_cfg.get()
            if cfg is not None:
                filtered = {}
                for ns in namespaces:
                    section = _resolve_nested_section(cfg, ns)
                    if section is not None:
                        data = section if isinstance(section, dict) else section.model_dump()
                        filtered.update({
                            # Get the tunable arguments from the config and retrieve the original name
                            k: data[ref_names.get(k, k)]
                            for k in sig.parameters
                            if ref_names.get(k, k) in data and k not in kwargs
                        })
                return fn(*args, **filtered, **kwargs)

            return fn(*args, **kwargs)

        return wrapper

    return decorator
