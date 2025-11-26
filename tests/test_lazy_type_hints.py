"""Tests that only tunable parameters have their annotations evaluated.

Historically the decorator called get_type_hints(fn) which eagerly resolves
all annotations. This breaks when non-tunable params reference names that only
exist under TYPE_CHECKING or simply are missing at runtime.

This test defines a function with:
  - a tunable param 'a' (selected explicitly)
  - a non-tunable param 'b' whose annotation is a forward ref to an undefined symbol
If the decorator still eagerly evaluated all annotations the decoration would
raise NameError. The lazy implementation should ignore 'b' and succeed.
"""

from __future__ import annotations

from tunablex import REGISTRY
from tunablex import tunable


# Use an explicit unique namespace to avoid interference with other tests.
@tunable("a", namespace="lazytest.func")
def sample(a: int = 1, b: MissingType = None):  # noqa: ANN001, F821 - forward ref intentional
    return a, b


def test_lazy_type_hints_ignores_nontunable_annotations():
    # Simply calling the function should work (would have failed during decoration previously)
    assert sample()[0] == 1

    # The registry should contain only field 'a' for this namespace.
    node = REGISTRY.entry_tree.children["lazytest"].children["func"]
    assert node is not None, "Namespace not registered"
    assert set(node.entries.keys()) == {"a"}

    # Building a config model for this namespace should succeed and produce defaults.
    AppConfig = REGISTRY.build_config_for_app("lazytest.func")  # noqa: N806
    inst = AppConfig()
    # For dotted namespace 'lazytest.func', root model has attribute 'lazytest'
    cfg_root = inst.lazytest  # type: ignore[attr-defined]
    assert hasattr(cfg_root, "func")
    assert cfg_root.func.a == 1
