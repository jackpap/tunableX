"""Conservative call discovery without invoking application functions or descriptors."""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable


def called_functions(entrypoint: Callable) -> set[str]:
    """Resolve source-level calls by identity, following imports, aliases and constructors.

    Both branches are visited. Dynamic dispatch, containers of callbacks and factories
    returning callables require explicit app tags; this is not a Python interpreter.
    """
    called: set[str] = set()
    visited: set[int] = set()

    def follow(target, *, required=False):
        if isinstance(target, (staticmethod, classmethod)):
            target = target.__func__
        if inspect.ismethod(target):
            target = target.__func__
        if inspect.isclass(target):
            follow(inspect.getattr_static(target, "__init__", None))
            follow(inspect.getattr_static(target, "__new__", None))
            return
        if not inspect.isfunction(target):
            if required:
                raise TypeError(
                    "Entry analysis requires a Python function or method; use app tags for callable objects."
                )
            return
        target = inspect.unwrap(target)
        if id(target) in visited:
            return
        visited.add(id(target))
        called.add(f"{target.__module__}.{target.__qualname__}")
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(target)))
        except (OSError, TypeError, SyntaxError) as exc:
            if required:
                raise ValueError(
                    "Entrypoint source is unavailable; use make_config_for_app() with explicit app tags."
                ) from exc
            return
        root = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
        if root is None:
            return
        scope = dict(target.__globals__)
        scope.update(inspect.getclosurevars(target).nonlocals)
        # Arguments shadow globals. Resolve self/cls for methods without making instances.
        for arg in [*root.args.posonlyargs, *root.args.args, *root.args.kwonlyargs]:
            scope[arg.arg] = None
        owner = target.__globals__.get(target.__qualname__.split(".")[0])
        if inspect.isclass(owner):
            scope["self"] = scope["cls"] = owner

        def resolve(node):
            if isinstance(node, ast.Name):
                return scope.get(node.id)
            if isinstance(node, ast.Attribute):
                obj = resolve(node.value)
                if obj is not None:
                    return inspect.getattr_static(obj, node.attr, None)
            # A local constructed instance is represented by its class, never executed.
            if isinstance(node, ast.Call):
                obj = resolve(node.func)
                return obj if inspect.isclass(obj) else None
            return None

        class Visitor(ast.NodeVisitor):
            def visit_Call(self, node):
                follow(resolve(node.func))
                self.generic_visit(node)

            def visit_Assign(self, node):
                self.visit(node.value)
                for item in node.targets:
                    if isinstance(item, ast.Name):
                        scope[item.id] = resolve(node.value)

            def visit_AnnAssign(self, node):
                if node.value is not None:
                    self.visit(node.value)
                    if isinstance(node.target, ast.Name):
                        scope[node.target.id] = resolve(node.value)

            def visit_FunctionDef(self, node):
                # Local definitions have no live function object until the entry runs.
                # Inspect their bodies conservatively without evaluating decorators/defaults.
                for statement in node.body:
                    self.visit(statement)

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_ClassDef(self, node):
                return

            def visit_Lambda(self, node):
                return

        visitor = Visitor()
        for statement in root.body:
            visitor.visit(statement)

    follow(entrypoint, required=True)
    return called
