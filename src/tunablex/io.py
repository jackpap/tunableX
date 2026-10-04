"""UTF-8 structured configuration loading with optional YAML support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_yaml(text: str):
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("YAML support requires: pip install 'tunablex[yaml]'") from exc
    value = yaml.safe_load(text)
    return {} if value is None else value


def load_structured_config(path: str | Path) -> dict[str, Any]:
    """Load a mapping from JSON, YAML or TOML; reject scalar/sequence roots."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    extension = path.suffix.lower()
    if extension in {".yaml", ".yml"}:
        data = _load_yaml(text)
    elif extension == ".toml":
        try:
            import tomllib
        except ModuleNotFoundError:  # Python 3.10
            import tomli as tomllib
        data = tomllib.loads(text)
    elif extension == ".json":
        data = json.loads(text)
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = _load_yaml(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config {str(path)!r} must contain an object/mapping, not {type(data).__name__}")
    return data
