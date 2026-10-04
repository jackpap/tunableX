from __future__ import annotations

import multiprocessing
import sys
import types
from typing import Annotated, Literal

import pytest
from pydantic import Field

from tunablex import TunableParams, tunable

PROBLEM_SPLITS = {"train_8": [], "valid_8": []}


class TypeResolutionParams(TunableParams):
    literal: Literal["fast", "safe"] = Field("fast")
    dynamic_literal: Literal[tuple(PROBLEM_SPLITS)] = Field("train_8")  # ruff: ignore[F821]
    annotated: Annotated[int | None, "metadata"] = Field(None)
    unresolved: MissingAtRuntime = Field(None)  # pyright: ignore[reportUndefinedVariable] # noqa: F821


@tunable("value", namespace="type_resolution")
def type_resolution_fn(value: str = TypeResolutionParams.literal):
    return value


def test_typing_annotations_are_resolved_and_cached():
    first = TypeResolutionParams.literal
    second = TypeResolutionParams.literal

    assert first.typ == Literal["fast", "safe"]
    assert second.typ is first.typ
    assert TypeResolutionParams.__tunable_type_hints__["literal"] is first.typ


def test_module_values_in_postponed_annotations_are_resolved_at_class_creation():
    assert TypeResolutionParams.dynamic_literal.typ == Literal["train_8", "valid_8"]


def test_type_resolution_works_for_unregistered_exec_namespace(monkeypatch):
    source = """
from __future__ import annotations
from typing import Literal
from pydantic import Field
from tunablex import TunableParams

PROBLEM_SPLITS = {"train_8": [], "valid_8": []}

class RunpyParams(TunableParams):
    split: Literal[tuple(PROBLEM_SPLITS)] = Field("train_8")
"""
    monkeypatch.delitem(sys.modules, "__mp_main__", raising=False)
    namespace = {"__name__": "__mp_main__", "__file__": "<runpy-test>"}
    exec(compile(source, "<runpy-test>", "exec"), namespace)

    params = namespace["RunpyParams"]
    assert params.split.typ == Literal["train_8", "valid_8"]


def test_type_resolution_prefers_exec_globals_over_placeholder_module(monkeypatch):
    source = """
from __future__ import annotations
from typing import Literal
from pydantic import Field
from tunablex import TunableParams

PROBLEM_SPLITS = {"train_8": [], "valid_8": []}

class RunpyParams(TunableParams):
    split: Literal[tuple(PROBLEM_SPLITS)] = Field("train_8")
"""
    monkeypatch.setitem(sys.modules, "__mp_main__", types.ModuleType("__mp_main__"))
    namespace = {"__name__": "__mp_main__", "__file__": "<runpy-test>"}
    exec(compile(source, "<runpy-test>", "exec"), namespace)

    assert namespace["RunpyParams"].split.typ == Literal["train_8", "valid_8"]


def test_unrelated_unresolvable_annotation_is_not_evaluated():
    assert TypeResolutionParams.literal.value.default == "fast"


def test_function_resolves_tunable_param_default_without_config():
    assert type_resolution_fn() == "fast"


def test_accessing_unresolvable_annotation_has_contextual_error():
    with pytest.raises(NameError, match=r"TypeResolutionParams.unresolved"):
        _ = TypeResolutionParams.unresolved


def _spawn_read_type(queue):
    queue.put(
        (
            TypeResolutionParams.literal.typ == Literal["fast", "safe"],
            TypeResolutionParams.dynamic_literal.typ == Literal["train_8", "valid_8"],
        )
    )


@pytest.mark.parametrize(
    "start_method", [method for method in ("spawn", "forkserver") if method in multiprocessing.get_all_start_methods()]
)
def test_type_resolution_works_in_worker(start_method):
    context = multiprocessing.get_context(start_method)
    queue = context.Queue()
    process = context.Process(target=_spawn_read_type, args=(queue,))
    try:
        try:
            process.start()
        except PermissionError as exc:
            if start_method == "forkserver":
                pytest.skip(f"Environment does not permit forkserver sockets: {exc}")
            raise
        process.join(timeout=10)
        assert process.exitcode == 0
        assert queue.get(timeout=2) == (True, True)
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        queue.close()
        queue.join_thread()
