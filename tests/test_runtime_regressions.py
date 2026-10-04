from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Annotated

import pytest
from pydantic import BaseModel, Field

from tunablex import TunableParams, make_config_for_app, tunable, use_config
from tunablex.registry import REGISTRY, Node


@pytest.fixture(autouse=True)
def isolate_registry(monkeypatch):
    monkeypatch.setattr(REGISTRY, "entry_tree", Node(""))


def test_positional_keyword_and_positional_only_values_win():
    @tunable(namespace="run", apps="test")
    def run(x: int = 1, /, y: int = 2):
        return x, y

    with use_config({"run": {"x": 10, "y": 20}}):
        assert run() == (10, 20)
        assert run(3) == (3, 20)
        assert run(3, 4) == (3, 4)
        assert run(y=7) == (10, 7)


class RuntimeOptions(BaseModel):
    path: Path = Path("data")


def test_nested_models_remain_models_on_injection():
    @tunable("options", apps="test")
    def run(options: RuntimeOptions = Field(default_factory=RuntimeOptions)):
        return options

    cfg = make_config_for_app("test")()
    with use_config(cfg):
        assert run() is cfg.options
        assert isinstance(run().path, Path)


def test_field_defaults_and_factories_resolve_without_config():
    @tunable()
    def run(x: int = Field(4, ge=1), values: list[int] = Field(default_factory=list)):
        values.append(x)
        return values

    assert run() == [4]
    assert run() == [4]
    with use_config({"unrelated": 2}):
        assert run() == [4]


def test_required_metadata_never_leaks_to_user_function():
    @tunable()
    def run(x: int = Field(...)):
        pytest.fail("Function should not run without its required value")

    with pytest.raises(TypeError, match="Missing required tunable x"):
        run()


def test_async_context_is_resolved_when_awaited_and_is_task_local():
    @tunable()
    async def run(x: int = 1):
        await asyncio.sleep(0)
        return x

    async def worker(value):
        coro = run()
        with use_config({"x": value}):
            return await coro

    async def main():
        return await asyncio.gather(worker(2), worker(3))

    assert inspect.iscoroutinefunction(run)
    assert asyncio.run(main()) == [2, 3]


def test_generator_context_and_nested_exception_restoration():
    @tunable()
    def run(x: int = 1):
        yield x

    generator = run()
    with use_config({"x": 2}):
        with pytest.raises(RuntimeError):
            with use_config({"x": 3}):
                assert list(run()) == [3]
                raise RuntimeError
        assert list(generator) == [2]
    assert list(run()) == [1]


def test_methods_both_decorator_orders():
    class Worker:
        @tunable(namespace="init")
        def __init__(self, x: int = 1):
            self.x = x

        @tunable(namespace="method")
        def method(self, x: int = 1):
            return x

        @tunable(namespace="static")
        @staticmethod
        def static(x: int = 1):
            return x

        @staticmethod
        @tunable(namespace="reverse")
        def reverse(x: int = 1):
            return x

        @tunable(namespace="class_method")
        @classmethod
        def class_method(cls, x: int = 1):
            return cls, x

    with use_config({key: {"x": 8} for key in ("init", "method", "static", "reverse", "class_method")}):
        obj = Worker()
        assert obj.x == 8
        assert obj.method(9) == 9
        assert obj.static() == Worker.static() == 8
        assert obj.reverse() == Worker.reverse() == 8
        assert obj.class_method() == (Worker, 8)


def test_unannotated_defaults_and_annotated_constraints():
    @tunable(apps="test")
    def run(value=7, positive: Annotated[int, Field(gt=0)] = 1):
        return value, positive

    model = make_config_for_app("test")
    assert model().value == 7
    with pytest.raises(ValueError):
        model(positive=0)


def test_parameter_class_can_be_reused_without_namespace_mutation():
    class SharedParams(TunableParams):
        count: int = Field(1)

    class FirstParams(TunableParams):
        Shared = SharedParams

    class SecondParams(TunableParams):
        Shared = SharedParams

    first = FirstParams.Shared
    second = SecondParams.Shared
    assert first.count.namespace == "first.shared"
    assert second.count.namespace == "second.shared"
    assert SharedParams.count.namespace == "shared"
    assert first.count.namespace == "first.shared"


@pytest.mark.parametrize("names", [{"include": ("typo",)}, {"exclude": "typo"}])
def test_unknown_parameter_names_are_rejected(names):
    include = names.pop("include", ())
    with pytest.raises(ValueError, match="Unknown tunable"):
        tunable(*include, **names)(lambda value=1: value)


def test_inherited_fields_and_plain_shadowing():
    class BaseParams(TunableParams):
        count: int = Field(2)

    class HTTPParams(BaseParams):
        pass

    class ModelXParams(BaseParams):
        count = 3

    assert HTTPParams.count.namespace == "http"
    assert HTTPParams.count.value.default == 2
    assert ModelXParams.count == 3
    assert ModelXParams.namespace == "model_x"


@pytest.mark.filterwarnings("error")
def test_renamed_factory_dependencies_use_canonical_names_per_namespace():
    class FirstParams(TunableParams):
        count: int = Field(2)
        doubled: int = Field(default_factory=lambda data: data["count"] * 2)

    class SecondParams(TunableParams):
        count: int = Field(10)
        doubled: int = Field(default_factory=lambda data: data["count"] * 2)

    @tunable(apps="factories")
    def run(n=FirstParams.count, other=SecondParams.count, d=FirstParams.doubled, twice=SecondParams.doubled):
        return n, other, d, twice

    assert run() == (2, 10, 4, 20)
    assert run(3, other=7) == (3, 7, 6, 14)
    with use_config(make_config_for_app("factories")()):
        assert run() == (2, 10, 4, 20)
