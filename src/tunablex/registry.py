"""Tunable registry and AppConfig builder.

- Supports registering tunables per namespace and app tag.
- Merges multiple function entries that target the same namespace by combining
  their Pydantic model fields and unioning app tags.
- Builds nested models for dotted namespaces so generated JSON/schema are nested.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationError
from pydantic import create_model

if TYPE_CHECKING:  # typing-only imports
    from collections.abc import Iterable
    import inspect


@dataclass
class TunableEntry:
    """A registered tunable function and its Pydantic model."""

    fn: Any
    model: type[BaseModel]
    sig: inspect.Signature
    namespace: str
    apps: set[str] = field(default_factory=set)


class TunableRegistry:
    """Holds all registered tunables grouped by namespace, and builds AppConfig."""

    def __init__(self) -> None:
        """Initialize empty registry."""
        self.by_namespace: dict[str, TunableEntry] = {}

    def register(self, entry: TunableEntry) -> None:
        """Register a tunable entry, merging with an existing namespace if present."""
        existing = self.by_namespace.get(entry.namespace)
        if not existing:
            self.by_namespace[entry.namespace] = entry
            return

        # Merge apps
        apps_merged = set(existing.apps) | set(entry.apps)

        # Merge model fields while preserving Field metadata. Prefer first definition.
        merged_fields: dict[str, tuple[type[Any] | Any, Any]] = {}
        for name, fld in existing.model.model_fields.items():
            ann = fld.annotation
            merged_fields[name] = (ann, fld)
        for name, fld in entry.model.model_fields.items():
            ann = fld.annotation
            if name in merged_fields:
                prev_ann, _prev_def = merged_fields[name]
                if prev_ann != ann:
                    msg = f"Conflicting field '{name}' in namespace '{entry.namespace}': {prev_ann} vs {ann}"
                    raise ValueError(msg)
                continue  # keep first definition
            merged_fields[name] = (ann, fld)

        model_name = f"{entry.namespace.title().replace('.', '').replace('_', '')}Config"
        merged_model: type[BaseModel] = create_model(model_name, **merged_fields)  # type: ignore[assignment]

        # Keep original fn/sig (arbitrary); update model and apps.
        self.by_namespace[entry.namespace] = TunableEntry(
            fn=existing.fn,
            model=merged_model,
            sig=existing.sig,
            namespace=entry.namespace,
            apps=apps_merged,
        )

    def namespaces_for_apps(self, apps: Iterable[str]) -> list[str]:
        """Return namespaces that have at least one of the given app tags."""
        want = set(apps)
        return [ns for ns, e in self.by_namespace.items() if e.apps & want]

    def build_config(self, namespaces: Iterable[str]) -> BaseModel:
        """Create a top-level AppConfig model from selected namespaces.

        - Dotted namespaces are nested (e.g., "a.b.c" becomes cfg.a.b.c).
        - Sections with required fields are emitted as Optional at the point they appear
          so the top-level AppConfig can instantiate without values supplied.
        """

        # Build a namespace tree where each node may have a leaf model and children.
        class _Node:
            __slots__ = ("children", "fullpath", "leaf")

            def __init__(self, fullpath: str):
                self.leaf: type[BaseModel] | None = None
                self.children: dict[str, _Node] = {}
                self.fullpath = fullpath

        roots: dict[str, _Node] = {}
        root_fields: dict[str, tuple[type[Any] | Any, Any]] = {}  # For empty namespace (root level)

        for ns in sorted(set(namespaces)):
            entry = self.by_namespace.get(ns)
            if not entry:
                continue
            
            # Handle empty namespace (root level) specially
            if ns == "":
                # Collect root-level fields directly
                for name, fld in entry.model.model_fields.items():
                    if name in root_fields:
                        # Check for conflicts
                        prev_ann, _prev = root_fields[name]
                        if prev_ann != fld.annotation:
                            msg = f"Conflicting root field '{name}': {prev_ann} vs {fld.annotation}"
                            raise ValueError(msg)
                        continue  # keep first definition
                    root_fields[name] = (fld.annotation, fld)
                continue
            
            parts = ns.split(".")
            if not parts or (len(parts) == 1 and parts[0] == ""):
                continue
            # Traverse/create nodes
            cur_map = roots
            cur_path = []
            node: _Node | None = None
            for seg in parts:
                if seg == "":  # Skip empty segments
                    continue
                cur_path.append(seg)
                path = ".".join(cur_path)
                if seg not in cur_map:
                    cur_map[seg] = _Node(path)
                node = cur_map[seg]
                cur_map = node.children
            # Assign/merge leaf model at the final node
            if node is not None:
                if node.leaf is None:
                    node.leaf = entry.model
                else:
                    # Merge existing leaf with new entry.model (same as register merge)
                    merged_fields: dict[str, tuple[type[Any] | Any, Any]] = {}
                    for name, fld in node.leaf.model_fields.items():
                        merged_fields[name] = (fld.annotation, fld)
                    for name, fld in entry.model.model_fields.items():
                        if name in merged_fields:
                            prev_ann, _prev = merged_fields[name]
                            if prev_ann != fld.annotation:
                                msg = (
                                    f"Conflicting field '{name}' in namespace '{ns}': "
                                    f"{prev_ann} vs {fld.annotation}"
                                )
                                raise ValueError(msg)
                            continue
                        merged_fields[name] = (fld.annotation, fld)
                    model_name = f"{ns.title().replace('.', '').replace('_', '')}Config"
                    node.leaf = create_model(model_name, **merged_fields)  # type: ignore[assignment]

        # Recursively build Pydantic models from the tree.
        def build_node_model(node: _Node) -> type[BaseModel]:
            # Start from leaf fields if any
            field_defs: dict[str, tuple[type[Any] | Any, Any]] = {}
            if node.leaf is not None:
                for fname, fld in node.leaf.model_fields.items():
                    if fname in node.children:
                        msg = (
                            f"Field name '{fname}' in '{node.fullpath}' conflicts with child namespace"
                        )
                        raise ValueError(msg)
                    field_defs[fname] = (fld.annotation, fld)

            # Add children as nested models
            for child_name, child_node in node.children.items():
                child_model = build_node_model(child_node)
                # If child_model() cannot instantiate, make it optional with default None
                try:
                    child_model()
                    field_defs[child_name] = (child_model, Field(default_factory=child_model))
                except ValidationError:
                    field_defs[child_name] = (child_model | None, None)

            model_name = f"{node.fullpath.title().replace('.', '').replace('_', '')}Config"
            return create_model(model_name, **field_defs)  # type: ignore[return-value]

        # Build top-level fields for AppConfig from roots
        app_fields: dict[str, tuple[type[BaseModel] | Any, Any]] = {}
        
        # Add root-level fields first (from empty namespace)
        app_fields.update(root_fields)
        
        # Then add nested namespace models
        for root_seg, root_node in roots.items():
            seg_field_name = root_seg.replace("-", "_")  # sanitize segment for attribute
            if seg_field_name in app_fields:
                msg = f"Root field '{seg_field_name}' conflicts with namespace segment"
                raise ValueError(msg)
            model = build_node_model(root_node)
            try:
                model()
                app_fields[seg_field_name] = (model, Field(default_factory=model))
            except ValidationError:
                app_fields[seg_field_name] = (model | None, None)

        if not app_fields:
            return create_model("AppConfigEmpty")
        return create_model("AppConfig", **app_fields)


REGISTRY = TunableRegistry()
