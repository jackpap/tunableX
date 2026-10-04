"""Decorator to declare tunable function parameters and auto-inject config.

Wraps functions, registers a Pydantic model per namespace, and injects values
from the active AppConfig at call time. Supports dotted namespaces.
"""

from __future__ import annotations

import ast
import functools
import inspect
import re
import sys
import textwrap
from copy import copy
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Annotated, Any, get_type_hints

from pydantic import TypeAdapter
from pydantic.fields import FieldInfo

from .annotations import raw_annotations, signature
from .context import _active_cfg
from .registry import REGISTRY, TunableArg

if TYPE_CHECKING:
    from collections.abc import Iterable


def _pascalcase_to_snake_case(ns: str) -> str:
    """Convert a namespace name from PascalCase to snake_case."""
    words = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", ns)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", words).lower()


def _get_description(cls: type, name: str) -> str | None:
    """Get a parameter's description from its docstring.

    Args:
        cls: TunableParams class containing the parameter.
        name: Parameter's name

    Returns:
        Parameter's docstring, or None if it does not exist.
    """
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
        body = next(node.body for node in tree.body if isinstance(node, ast.ClassDef))
        for current, following in pairwise(body):
            if (
                isinstance(current, ast.AnnAssign)
                and isinstance(current.target, ast.Name)
                and current.target.id == name
                and isinstance(following, ast.Expr)
                and isinstance(following.value, ast.Constant)
                and isinstance(following.value.value, str)
            ):
                return inspect.cleandoc(following.value.value)
    except (OSError, TypeError, SyntaxError, StopIteration):
        pass
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

        for field_name, raw_annotation in raw_annotations(cls).items():
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
            field = copy(field)
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
            fields = type.__getattribute__(candidate, "__dict__").get("__tunable_fields__", {})
            if name in fields:
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
        name = _pascalcase_to_snake_case(name).removesuffix("_params")
        if name == "main" or name == "root":
            name = ""
        return name

    def __getattribute__(cls, name: str) -> Any | tuple[Any, str, str, str]:
        """Return centralized fields as metadata references."""
        value = super().__getattribute__(name)
        if not isinstance(cls, TunableParamsMeta):
            return value

        if isinstance(value, TunableParamsMeta):
            return _ParameterNamespace(value, _join_namespace(cls.namespace, name))

        if not isinstance(value, FieldInfo):
            return value

        declaring_cls = TunableParamsMeta._declaring_class(cls, name)
        if declaring_cls is not None:
            fields = type.__getattribute__(declaring_cls, "__tunable_fields__")
            if name in fields:
                field_data = fields[name]
                typ = TunableParamsMeta._resolve_type(name, declaring_cls)
                if field_data.value.description is None:
                    field_data.value.description = _get_description(declaring_cls, name)
                return TunableParamData(field_data.value, typ, type.__getattribute__(cls, "namespace"), name)

        return value


def _join_namespace(parent: str, name: str) -> str:
    child = TunableParamsMeta._compose_namespace(name)
    return ".".join(filter(None, [parent, child]))


@dataclass(frozen=True)
class _ParameterNamespace:
    """Immutable namespace view: reusing a parameter class cannot mutate prior references."""

    cls: type
    namespace: str

    def __getattr__(self, name):
        raw = type.__getattribute__(self.cls, name)
        if isinstance(raw, TunableParamsMeta):
            return _ParameterNamespace(raw, _join_namespace(self.namespace, name))
        value = getattr(self.cls, name)
        if isinstance(value, TunableParamData):
            return TunableParamData(value.value, value.typ, self.namespace, value.name)
        return value


class TunableParams(metaclass=TunableParamsMeta):
    """A class containing tunable parameters.

    Inherit from this class to declare tunable parameters globally.
    A trailing `Params` is removed from the class namespace for brevity.
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


def _resolve_nested_section(cfg_model, dotted_ns: str):
    obj = cfg_model
    for segment in dotted_ns.split(".") if dotted_ns else ():
        obj = obj.get(segment) if isinstance(obj, dict) else getattr(obj, segment, None)
    return obj


def tunable(
    *include: str,
    namespace: str = "",
    exclude: str | Iterable[str] = (),
    apps: str | Iterable[str] = (),
):
    """Register selected function parameters and inject missing arguments from use_config.

    With no include list, select parameters with defaults, minus exclude. Explicit
    positional and keyword arguments take precedence. Without an active config,
    selected Field/reference defaults are resolved to values; required ones raise.
    """
    include_set = set(include)
    exclude_set = {exclude} if isinstance(exclude, str) else set(exclude)
    if include_set and exclude_set:
        raise ValueError("Cannot pass both `include` and `exclude` arguments.")
    app_set = {apps} if isinstance(apps, str) else set(apps)

    def decorator(fn):
        if isinstance(fn, (staticmethod, classmethod)):
            return type(fn)(decorator(fn.__func__))
        sig = signature(fn)
        original = inspect.unwrap(fn)
        unknown = (include_set | exclude_set) - sig.parameters.keys()
        if unknown:
            raise ValueError(f"Unknown tunable parameter(s) on {fn.__qualname__}: {', '.join(sorted(unknown))}")
        parameters = {}
        for name, parameter in sig.parameters.items():
            if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                if name in include_set:
                    raise ValueError(f"Variadic parameter {name!r} cannot be tunable")
                continue
            selected = (
                name in include_set
                if include_set
                else (parameter.default is not inspect.Parameter.empty and name not in exclude_set)
            )
            if not selected:
                continue
            default = parameter.default if parameter.default is not inspect.Parameter.empty else ...
            global_name, ns = name, namespace
            if isinstance(default, TunableParamData):
                default, typ, ns, global_name = default.value, default.typ, default.namespace, default.name
            else:
                annotation = raw_annotations(original).get(name, Any)
                # Resolve just the selected annotation, preserving Annotated metadata.
                proxy = type("_Annotation", (), {"__annotations__": {name: annotation}})
                typ = get_type_hints(proxy, globalns=original.__globals__, include_extras=True)[name]
            REGISTRY.register(
                TunableArg(
                    name=global_name,
                    typ=typ,
                    default=default,
                    namespace=ns,
                    fn_names={f"{fn.__module__}.{fn.__qualname__}"},
                    apps=set(app_set),
                )
            )
            parameters[name] = (ns, global_name, default, typ)

        def prepare(args, kwargs):
            bound = sig.bind_partial(*args, **kwargs)
            cfg = _active_cfg.get()
            for name, (ns, key, default, typ) in parameters.items():
                if name in bound.arguments:
                    if isinstance(bound.arguments[name], (TunableParamData, FieldInfo)):
                        raise TypeError(f"{fn.__qualname__}.{name} requires a value, not parameter metadata")
                    continue
                section = _resolve_nested_section(cfg, ns)
                if isinstance(section, dict) and key in section:
                    bound.arguments[name] = section[key]
                elif section is not None and not isinstance(section, dict) and hasattr(section, key):
                    # getattr preserves nested models, custom objects, Paths and Enums.
                    bound.arguments[name] = getattr(section, key)
                elif isinstance(default, FieldInfo):
                    if default.is_required():
                        raise TypeError(f"Missing required tunable {ns + '.' if ns else ''}{key} for {fn.__qualname__}")
                    bound.arguments[name] = TypeAdapter(Annotated[typ, default]).validate_python(
                        default.get_default(call_default_factory=True, validated_data=bound.arguments)
                    )
                elif default is not ...:
                    # Leave ordinary Python defaults alone unless binding positional-only gaps.
                    bound.arguments[name] = default
            return bound

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def wrapper(*args, **kwargs):
                bound = prepare(args, kwargs)
                return await fn(*bound.args, **bound.kwargs)
        elif inspect.isgeneratorfunction(fn):

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                bound = prepare(args, kwargs)
                yield from fn(*bound.args, **bound.kwargs)
        else:

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                bound = prepare(args, kwargs)
                return fn(*bound.args, **bound.kwargs)

        return wrapper

    return decorator
