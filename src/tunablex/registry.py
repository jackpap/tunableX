"""Tunable registry and AppConfig builder.

- Supports registering tunables per namespace and app tag.
- Merges multiple function entries that target the same namespace by combining
  their Pydantic model fields and unioning app tags.
- Builds nested models for dotted namespaces so generated JSON/schema are nested.
"""

from __future__ import annotations

import ast
import inspect
import sys
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic import create_model

if TYPE_CHECKING:
    from collections.abc import Callable


def _gather_called_function_names(entry_fn: Callable, called: set):
    """Return set of fully qualified function names that are reachable from entry_fn's module.

    This performs a simple static AST walk starting from the entry function's
    body and collects names of function calls. It does not follow dynamic
    dispatch or conditional imports. The goal is only to approximate which
    @tunable-decorated functions may be used so we can compose an AppConfig
    without executing user code.
    """
    try:
        src = inspect.getsource(entry_fn)
    except (OSError, TypeError):  # source not available
        return

    tree = ast.parse(src)
    sub_called = set()

    class CustomVisitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call):
            """Add the name of called functions to the `sub_called` set."""
            name_parts: list[str] = []
            cur = node.func
            while isinstance(cur, ast.Attribute):
                name_parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                name_parts.append(cur.id)
            fn_fullname = ".".join(reversed(name_parts))
            if fn_fullname:
                sub_called.add(fn_fullname)
            self.generic_visit(node)

        def visit_ClassDef(self, node: ast.ClassDef):
            """Add the __init__ of called classes to the `sub_called` set."""
            sub_called.add(f"{node.name}.__init__")
            self.generic_visit(node)

    CustomVisitor().visit(tree)
    for sub_fn_fullname in sub_called.difference(called):
        called.add(sub_fn_fullname)
        sub_fn_name = sub_fn_fullname.split(".")[-1]
        with suppress(Exception):
            sub_fn = vars(sys.modules[entry_fn.__module__]).get(sub_fn_name)
            if sub_fn is not None:
                _gather_called_function_names(sub_fn, called)


@dataclass
class TunableArg:
    """A registered tunable argument."""

    name: str
    """The argument's name."""

    typ: Any
    """The argument's type."""

    default: Field | Any
    """The argument's default value."""

    fn_names: set[str]
    """The names of the functions that call the argument.
    Used for config generation with AST.
    The same function can be present twice, with and without the module name, if any.
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


class TunableRegistry:
    """Holds all registered tunables grouped by namespace, and builds AppConfig."""

    entry_tree: Node
    """Tree containing the namespaces and the corresponding TunableEntry."""

    def __init__(self) -> None:
        """Initialize empty registry."""
        self.entry_tree = Node("")

    def register(self, entry: TunableArg) -> None:
        """Register a tunable entry, merging with an existing namespace if present."""
        node = self.entry_tree
        if entry.namespace:
            fullpath = entry.namespace.split(".")
            path = []
            for p in fullpath:
                path.append(p)
                node = node.children.setdefault(p, Node(".".join(path)))
        existing_entry = node.entries.get(entry.name)
        if existing_entry is None:
            node.entries[entry.name] = entry
            return

        # Ensure the type and default are the same for already existing entries
        if existing_entry.typ != entry.typ:
            msg = f"Conflicting type for arg '{entry.name}' in namespace '{entry.namespace}': "
            f"{existing_entry.typ} vs {entry.typ}"
            raise ValueError(msg)
        if existing_entry.default != entry.default:
            msg = f"Conflicting default value for arg '{entry.name}' in namespace '{entry.namespace}': "
            f"{existing_entry.default} vs {entry.default}"
            raise ValueError(msg)

        # Merge the apps and the functions
        existing_entry.apps.update(entry.apps)
        existing_entry.fn_names.update(entry.fn_names)

    def build_config_for_app(self, app: str, node: Node | None = None) -> type[BaseModel]:
        """Recursively create an AppConfig model for a given app."""
        node = self.entry_tree if node is None else node
        fields = {}
        for name, entry in node.entries.items():
            if app in entry.apps or "ALL" in entry.apps:
                fields[name] = (entry.typ, entry.default)
        for name, child in node.children.items():
            child_model = self.build_config_for_app(app, child)
            if child_model.model_fields:
                fields[name] = (child_model, Field(default_factory=child_model))

        model_name = f"{node.path.title().replace('_', '').replace('.', '_')}_Config"
        return create_model(model_name, **fields)

    def _build_config_from_called(self, called: set[str], node: Node | None = None) -> type[BaseModel]:
        """Build a config based on a the set of functions called."""
        node = self.entry_tree if node is None else node
        fields = {}
        for name, entry in node.entries.items():
            if entry.fn_names.intersection(called):
                fields[name] = (entry.typ, entry.default)
        for name, child in node.children.items():
            child_model = self._build_config_from_called(called, child)
            if child_model.model_fields:
                fields[name] = (child_model, Field(default_factory=child_model))

        model_name = f"{node.path.title().replace('_', '').replace('.', '_')}_Config"
        return create_model(model_name, **fields)

    def build_config_for_entrypoint(self, entrypoint: Callable) -> type[BaseModel]:
        """Build a config based on all functions calls from an entry point."""
        called = {entrypoint.__qualname__}
        _gather_called_function_names(entrypoint, called)
        return self._build_config_from_called(called)


REGISTRY = TunableRegistry()
