"""Task-local configuration contexts with nested and exception-safe restoration."""

from contextlib import contextmanager
from contextvars import ContextVar

from pydantic import BaseModel

_active_cfg: ContextVar[BaseModel | dict | None] = ContextVar("tunablex_active_cfg", default=None)


@contextmanager
def use_config(cfg: BaseModel | dict):
    """Activate a validated config (or raw dict) for this thread/async task.

    Explicit function arguments win. Contexts do not automatically cross worker
    processes or new threads; activate a config in each worker.
    """
    if not isinstance(cfg, (BaseModel, dict)):
        raise TypeError("use_config expects a Pydantic model or dict")
    token = _active_cfg.set(cfg)
    try:
        yield cfg
    finally:
        _active_cfg.reset(token)
