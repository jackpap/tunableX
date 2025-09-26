"""Decorator to declare tunable function parameters and auto-inject config.

Wraps functions, registers a Pydantic model per namespace, and injects values
from the active AppConfig at call time. Supports dotted namespaces.
"""

from __future__ import annotations

import functools
import inspect
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import create_model

from .context import _active_cfg
from .registry import REGISTRY
from .registry import TunableEntry

# Add import for the new parameters module
try:
    from .parameters import TunableParameters
    from .parameters import get_parameter_class_namespace
    _PARAMETERS_AVAILABLE = True
except ImportError:
    _PARAMETERS_AVAILABLE = False

if TYPE_CHECKING:
    from collections.abc import Iterable


def _infer_namespace_from_class_attribute(default_value: Any, fn_globals: dict) -> str | None:
    """Infer namespace from a default value that's a TunableParameters class attribute.

    Detects patterns like Model.hidden_units and extracts the namespace from Model class.
    """
    if not _PARAMETERS_AVAILABLE:
        return None

    # Check if the default value is referencing a class attribute
    # This is a heuristic based on the value's type and context
    if hasattr(default_value, "__self__"):
        # This is a bound method or descriptor
        return None

    # Try to find if this default value came from a TunableParameters class
    # by scanning the function's globals for TunableParameters subclasses
    for obj in fn_globals.values():
        if (
            isinstance(obj, type)
            and issubclass(obj, TunableParameters)
            and obj is not TunableParameters
        ):

            # Check if this class has an attribute with the same value
            for attr_name in dir(obj):
                if not attr_name.startswith("_"):
                    try:
                        attr_value = getattr(obj, attr_name)
                        if attr_value is default_value:
                            # Found a match! Extract namespace from this class
                            return get_parameter_class_namespace(obj)
                    except (AttributeError, TypeError):
                        continue

    return None


def tunable(
    *include: str,
    namespace: str | None = None,
    mode: Literal["include", "exclude"] = "include",
    exclude: Iterable[str] | None = None,
    apps: Iterable[str] = (),
):
    """Mark a function's selected parameters as user-tunable.

    - include: names to include. If empty, include all params that have defaults
      (unless mode='exclude' with an explicit exclude list).
    - namespace: JSON section name; defaults to 'module.function'.
    - apps: optional tags to group functions per executable/app.
    """
    include_set = set(include or ())
    exclude_set = set(exclude or ())

    def decorator(fn):
        sig = inspect.signature(fn)
        raw_anns = inspect.get_annotations(fn, eval_str=False)

        def _eval_ann(value: Any) -> Any:
            if isinstance(value, str):  # deferred annotation (from __future__ import annotations)
                try:
                    return eval(value, fn.__globals__, {})  # noqa: S307 - controlled eval
                except (NameError, AttributeError, SyntaxError):  # pragma: no cover - fallback path
                    return Any
            return value

        fields: dict[str, tuple[type[Any], Any]] = {}
        for name, p in sig.parameters.items():
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            if include_set:
                selected = name in include_set
            elif mode == "exclude" and exclude_set:
                selected = (p.default is not inspect._empty) and (name not in exclude_set)
            else:
                selected = p.default is not inspect._empty
            if not selected:
                continue
            ann = _eval_ann(raw_anns.get(name, Any))
            default = p.default if p.default is not inspect._empty else ...
            fields[name] = (ann, default)

        # Collect fields and handle centralized parameters
        if namespace is None:
            # Group parameters by their inferred namespaces
            namespace_groups: dict[str, dict[str, tuple[type[Any], Any]]] = {}
            
            for name, p in sig.parameters.items():
                if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                    continue
                if include_set:
                    selected = name in include_set
                elif mode == "exclude" and exclude_set:
                    selected = (p.default is not inspect._empty) and (name not in exclude_set)
                else:
                    selected = p.default is not inspect._empty
                if not selected:
                    continue
                
                # Infer namespace for this specific parameter
                param_ns = _infer_namespace_from_class_attribute(p.default, fn.__globals__) or ""
                
                if param_ns not in namespace_groups:
                    namespace_groups[param_ns] = {}
                
                ann = _eval_ann(raw_anns.get(name, Any))
                default = p.default if p.default is not inspect._empty else ...
                namespace_groups[param_ns][name] = (ann, default)
            
            # Register each namespace group separately
            for ns, ns_fields in namespace_groups.items():
                if ns_fields:  # Only register if there are fields
                    model_name = f"{ns.title().replace('.', '').replace('_', '')}Config"
                    model_type = create_model(model_name, **ns_fields)  # type: ignore[assignment]
                    REGISTRY.register(TunableEntry(fn=fn, model=model_type, sig=sig, namespace=ns, apps=set(apps)))
            
            # Multi-namespace mode complete
        else:
            # Original single-namespace behavior
            for name, p in sig.parameters.items():
                if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                    continue
                if include_set:
                    selected = name in include_set
                elif mode == "exclude" and exclude_set:
                    selected = (p.default is not inspect._empty) and (name not in exclude_set)
                else:
                    selected = p.default is not inspect._empty
                if not selected:
                    continue
                ann = _eval_ann(raw_anns.get(name, Any))
                default = p.default if p.default is not inspect._empty else ...
                fields[name] = (ann, default)

            ns = namespace
            model_name = f"{ns.title().replace('.', '').replace('_', '')}Config"
            model_type = create_model(model_name, **fields)  # type: ignore[assignment]
            REGISTRY.register(TunableEntry(fn=fn, model=model_type, sig=sig, namespace=ns, apps=set(apps)))

        def _resolve_nested_section(cfg_model: BaseModel, dotted_ns: str):
            obj: Any = cfg_model
            for seg in dotted_ns.split("."):
                if obj is None or not hasattr(obj, seg):
                    return None
                obj = getattr(obj, seg)
            return obj

        @functools.wraps(fn)
        def wrapper(*args, cfg: BaseModel | dict | None = None, **kwargs):
            if cfg is not None:
                data = cfg if isinstance(cfg, dict) else cfg.model_dump()
                filtered = {k: v for k, v in data.items() if k in sig.parameters}
                return fn(*args, **filtered, **kwargs)

            app_cfg = _active_cfg.get()
            if app_cfg is not None:
                # Collect parameters from all relevant namespaces
                combined_params = {}
                
                if namespace is None:
                    # Multi-namespace mode: collect from all registered namespaces for this function
                    for entry in REGISTRY.by_namespace.values():
                        if entry.fn is fn:
                            section = _resolve_nested_section(app_cfg, entry.namespace)
                            if section is not None:
                                data = section if isinstance(section, dict) else section.model_dump()
                                for k, v in data.items():
                                    if k in sig.parameters:
                                        combined_params[k] = v
                else:
                    # Single-namespace mode: use the specified namespace
                    section = _resolve_nested_section(app_cfg, namespace)
                    if section is not None:
                        data = section if isinstance(section, dict) else section.model_dump()
                        for k, v in data.items():
                            if k in sig.parameters:
                                combined_params[k] = v
                
                if combined_params:
                    return fn(*args, **combined_params, **kwargs)

            return fn(*args, **kwargs)

        return wrapper

    return decorator
