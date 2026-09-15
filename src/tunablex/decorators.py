"""Decorator to declare tunable function parameters and auto-inject config.

Wraps functions, registers a Pydantic model per namespace, and injects values
from the active AppConfig at call time. Supports dotted namespaces.
"""

from __future__ import annotations

import functools
import inspect
import re
import sys
from dataclasses import dataclass
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
    return re.sub(r"(?<=[a-zA-Z0-9])(?=[A-Z][a-z])", "_", ns).lower()


def _get_description(cls: type, name: str) -> str | None:
    """Get a parameter's description from its docstring.

    Args:
        cls: TunableParams class containing the parameter.
        name: Parameter's name

    Returns:
        Parameter's docstring, or None if it does not exist.
    """
    try:
        source = inspect.getsource(cls)
        i1 = source.index(name)
        s1 = source[i1:]
        i2 = s1.index('"""') + 3
        s2 = s1[i2:]
        i3 = s2.index('"""')
        return s2[:i3]
    except Exception:  # noqa: BLE001
        return None


@dataclass(frozen=True)
class TunableParamData:
    """Class containing the data of a tunable parameter."""

    value: Any
    typ: type
    namespace: str
    name: str
    raw_annotation: Any = None


class TunableParamsMeta(type):
    """Metaclass that exposes centralized fields as parameter references."""

    def __init__(cls, name, bases, attrs):  # noqa: D107
        super().__init__(name, bases, attrs)
        type.__setattr__(cls, "namespace", TunableParamsMeta._compose_namespace(name))
        type.__setattr__(cls, "__tunable_type_hints__", {})
        type.__setattr__(cls, "__tunable_fields__", {})
        type.__setattr__(cls, "__tunable_globals__", TunableParamsMeta._execution_globals(cls))

        for field_name, raw_annotation in attrs.get("__annotations__", {}).items():
            field = attrs.get(field_name)
            if not isinstance(field, FieldInfo):
                continue
            try:
                typ = TunableParamsMeta._resolve_annotation(field_name, raw_annotation, cls)
            except NameError:
                # Keep unrelated unresolved annotations lazy. If the field is
                # actually used as a tunable default, _resolve_type below
                # raises the contextual error instead.
                typ = None
            if typ is not None:
                type.__getattribute__(cls, "__tunable_type_hints__")[field_name] = typ
            if field.description is None:
                field.description = _get_description(cls, field_name)
            type.__getattribute__(cls, "__tunable_fields__")[field_name] = TunableParamData(
                field,
                typ if typ is not None else Any,
                type.__getattribute__(cls, "namespace"),
                field_name,
                raw_annotation,
            )

    @staticmethod
    def _declaring_class(cls, name: str):
        """Return the MRO class that declares ``name``'s annotation."""
        for candidate in type.__getattribute__(cls, "__mro__"):
            annotations = type.__getattribute__(candidate, "__dict__").get("__annotations__", {})
            if name in annotations:
                return candidate
        return None

    @staticmethod
    def _execution_globals(cls) -> dict[str, Any]:
        """Return globals used while executing the declaring class."""
        module_name = type.__getattribute__(cls, "__module__")
        frame = inspect.currentframe()
        try:
            while frame is not None:
                if frame.f_globals.get("__name__") == module_name:
                    return frame.f_globals
                frame = frame.f_back
        finally:
            del frame

        module = sys.modules.get(module_name)
        if module is not None:
            return vars(module)
        return {}

    @staticmethod
    def _resolve_annotation(name: str, raw_annotation: Any, declaring_cls: type) -> type:
        """Resolve one annotation using the declaring module's namespace."""
        if not isinstance(raw_annotation, str):
            return raw_annotation

        module_name = type.__getattribute__(declaring_cls, "__module__")
        module_globals = type.__getattribute__(declaring_cls, "__tunable_globals__")
        if not module_globals:
            module = sys.modules.get(module_name)
            module_globals = {} if module is None else vars(module)
        localns = dict(type.__getattribute__(declaring_cls, "__dict__"))
        localns[type.__getattribute__(declaring_cls, "__name__")] = declaring_cls
        proxy = type(
            "_TunableAnnotationProxy",
            (),
            {"__module__": module_name, "__annotations__": {name: raw_annotation}},
        )
        return get_type_hints(
            proxy,
            globalns=module_globals,
            localns=localns,
            include_extras=True,
        )[name]

    @staticmethod
    def _resolve_type(name: str, declaring_cls: type) -> type:
        """Resolve a deferred annotation when its field is actually used."""
        cache = type.__getattribute__(declaring_cls, "__tunable_type_hints__")
        if name in cache:
            return cache[name]

        field_data = type.__getattribute__(declaring_cls, "__tunable_fields__")[name]
        try:
            resolved = TunableParamsMeta._resolve_annotation(name, field_data.raw_annotation, declaring_cls)
        except NameError as exc:
            missing = exc.name or str(exc)
            msg = (
                f"Unable to resolve annotation for tunable field "
                f"'{type.__getattribute__(declaring_cls, '__module__')}."
                f"{type.__getattribute__(declaring_cls, '__qualname__')}.{name}' "
                f"(raw annotation: {field_data.raw_annotation!r}); missing name: {missing}. "
                "The name must be available at runtime, not only under TYPE_CHECKING."
            )
            raise NameError(msg) from exc

        cache[name] = resolved
        return resolved

    @staticmethod
    def _compose_namespace(name: str) -> str:
        """Turn a class name into a namespace."""
        name = _pascalcase_to_snake_case(name).replace("_params", "")
        if name == "main" or name == "root":
            name = ""
        return name

    def __getattribute__(cls, name: str) -> Any | tuple[Any, str, str, str]:
        """Return centralized fields as metadata references."""
        value = super().__getattribute__(name)
        if not isinstance(cls, TunableParamsMeta):
            return value

        # If value is a class with this metaclass, update its parent namespace
        if isinstance(value, type) and isinstance(value, TunableParamsMeta):
            parent_namespace = type.__getattribute__(cls, "namespace")
            value.namespace = (
                f"{parent_namespace}.{TunableParamsMeta._compose_namespace(name)}"
                if parent_namespace
                else TunableParamsMeta._compose_namespace(name)  # for cases like main.advanced
            )
            return value

        declaring_cls = TunableParamsMeta._declaring_class(cls, name)
        if declaring_cls is not None:
            fields = type.__getattribute__(declaring_cls, "__tunable_fields__")
            if name in fields:
                field_data = fields[name]
                typ = TunableParamsMeta._resolve_type(name, declaring_cls)
                return TunableParamData(field_data.value, typ, type.__getattribute__(cls, "namespace"), name)

        return value


class TunableParams(metaclass=TunableParamsMeta):
    """A class containing tunable parameters.

    Inherit from this class to declare tunable parameters globally.
    If the class name contains `Params`, it will be removed from the namespace for brevity.
    If the resulting namespace is `main` or `root`, the parameters will be stored at the root level.

    When using several levels of namespaces, it is possible to declare the parameters in a class at the root level
    and to reference this class in the namespace, to avoid having too many indentations in the lower levels.

    Docstrings enclosed in triple double-quotes will be used as parameter's description
    if none is provided in the Field definition.

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
        namespaces = {}
        global_names = {}
        tunable_param_defaults = set()
        for name, p in sig.parameters.items():
            global_name = name
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
            if isinstance(default, TunableParamData):
                tunable_param_defaults.add(name)
                # The parameter is declared in a TunableParam class; retrieve type, namespace and reference name
                default, typ, ns, global_name = (
                    default.value,
                    default.typ,
                    default.namespace,
                    default.name,
                )
            else:
                typ = inspect.get_annotations(fn, eval_str=False)[name]
                typ = eval(typ, fn.__globals__) if isinstance(typ, str) else typ
                ns = namespace
            namespaces.setdefault(ns, []).append(name)
            if global_name != name:
                # Store the global name for later look-up (allows different local names for the same parameter)
                global_names[name] = global_name
                name = global_name
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
        def wrapper(*args, **kwargs):
            # Handle static methods called from instances
            if isinstance(fn, staticmethod) and args[0].__class__.__name__ == fn.__qualname__.split(".")[0]:
                args = args[1:]
            injected = {}
            cfg = _active_cfg.get()
            if cfg is not None:
                for ns, ns_vars in namespaces.items():
                    section = _resolve_nested_section(cfg, ns)
                    data = section if isinstance(section, dict) else section.model_dump()
                    injected.update({
                        name: data[global_names.get(name, name)]
                        for name in ns_vars
                        if global_names.get(name, name) in data and name not in kwargs
                    })
            call_kwargs = {**injected, **kwargs}

            bound = sig.bind_partial(*args, **call_kwargs)
            for name in tunable_param_defaults:
                value = bound.arguments.get(name, inspect.Parameter.empty)
                if value is inspect.Parameter.empty or isinstance(value, TunableParamData):
                    msg = (
                        f"Function '{fn.__qualname__}' would receive TunableParamData "
                        f"for parameter '{name}'. Activate a config with use_config() "
                        "or provide an explicit value."
                    )
                    raise TypeError(msg)

            return fn(*args, **call_kwargs)

        return wrapper

    return decorator
