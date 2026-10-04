"""Tunable registry and AppConfig builder.

- Supports registering tunables per namespace and app tag.
- Merges multiple function entries that target the same namespace by combining
  their Pydantic model fields and unioning app tags.
- Builds nested models for dotted namespaces so generated JSON/schema are nested.
"""

from __future__ import annotations

import keyword
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic.fields import FieldInfo

if TYPE_CHECKING:
    from collections.abc import Callable


from .analysis import called_functions


@dataclass
class TunableArg:
    """A registered tunable argument."""

    name: str
    """The argument's name."""

    typ: Any
    """The argument's type."""

    default: Any
    """The argument's default value."""

    fn_names: set[str]
    """The names of the functions that call the argument.
    Used for config generation with AST.
    Fully qualified names prevent collisions across modules.
    """

    namespace: str
    """The argument's namespace."""

    apps: set[str]
    """The apps where the argument is used."""

    def __post_init__(self):
        """If no app is provided, default to ALL."""
        if not self.apps:
            self.apps = {"ALL"}


class Node:
    """A tree node that can store the TunableArgs associated to a namespace."""

    entries: dict[str, TunableArg]
    """The entries of the namespace corresponding to the node."""

    children: dict[str, Node]
    """The node's children."""

    path: str
    """The node's full path."""

    def __init__(self, path: str):  # noqa: D107
        self.entries = {}
        self.children = {}
        self.path = path


class _ConfigModel(BaseModel):
    """Base for generated namespace models, distinct from user-supplied model fields."""

    model_config = ConfigDict(extra="forbid", validate_default=True, populate_by_name=True)


class TunableRegistry:
    """Holds all registered tunables grouped by namespace, and builds AppConfig."""

    entry_tree: Node
    """Tree containing the namespaces and the corresponding TunableEntry."""

    def __init__(self) -> None:
        """Initialize empty registry."""
        self.entry_tree = Node("")

    def register(self, entry: TunableArg) -> None:
        """Register a field, rejecting ambiguous paths and incompatible declarations."""
        segments = entry.namespace.split(".") if entry.namespace else []
        for name in [*segments, entry.name]:
            if not name.isidentifier() or keyword.iskeyword(name) or name.startswith("_") or hasattr(BaseModel, name):
                raise ValueError(f"Invalid or reserved config field name: {name!r}")
        node = self.entry_tree
        for segment in segments:
            if segment in node.entries:
                raise ValueError(f"Namespace {entry.namespace!r} collides with field {segment!r}")
            node = node.children.setdefault(segment, Node(".".join(filter(None, [node.path, segment]))))
        if entry.name in node.children:
            raise ValueError(f"Field {entry.name!r} collides with a namespace in {entry.namespace!r}")
        existing = node.entries.get(entry.name)
        if existing is None:
            node.entries[entry.name] = entry
            return
        if existing.typ != entry.typ:
            raise ValueError(
                f"Conflicting type for arg '{entry.name}' in namespace '{entry.namespace}': "
                f"{existing.typ} vs {entry.typ}"
            )
        old = existing.default.asdict() if isinstance(existing.default, FieldInfo) else existing.default
        new = entry.default.asdict() if isinstance(entry.default, FieldInfo) else entry.default
        if old != new:
            raise ValueError(
                f"Conflicting default value for arg '{entry.name}' in namespace '{entry.namespace}': "
                f"{existing.default} vs {entry.default}"
            )
        existing.apps.update(entry.apps)
        existing.fn_names.update(entry.fn_names)

    def _build(self, select, node: Node | None = None) -> type[BaseModel]:
        node = self.entry_tree if node is None else node
        fields = {}
        for name, entry in node.entries.items():
            if not select(entry):
                continue
            default = entry.default
            if isinstance(default, FieldInfo):
                info = default.asdict()
                default = Field(**info["attributes"])
                default.metadata = list(info["metadata"])
            fields[name] = (entry.typ, default)
        for name, child in node.children.items():
            child_model = self._build(select, child)
            if child_model.model_fields:
                # Required descendants make the section required in both validation and JSON Schema.
                required = any(field.is_required() for field in child_model.model_fields.values())
                fields[name] = (child_model, ... if required else Field(default_factory=dict))
        model_name = f"{node.path.title().replace('_', '').replace('.', '_')}_Config"
        return create_model(
            model_name,
            __base__=_ConfigModel,
            **fields,
        )

    def build_config_for_app(self, app: str, node: Node | None = None) -> type[BaseModel]:
        """Build a model for an app, including untagged (ALL) parameters."""
        return self._build(lambda entry: app in entry.apps or "ALL" in entry.apps, node)

    def build_config_for_entrypoint(self, entrypoint: Callable) -> type[BaseModel]:
        """Build a config from statically resolved functions, without calling them."""
        called = called_functions(entrypoint)
        return self._build(lambda entry: bool(entry.fn_names.intersection(called)))


REGISTRY = TunableRegistry()
