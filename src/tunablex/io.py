from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_structured_config(path: str | Path) -> dict[str, Any]:
    """Load JSON or YAML (and TOML when '.toml') into a dict.
    Tries by extension first; falls back to JSON then YAML.
    """
    p = Path(path)
    text = p.read_text()
    ext = p.suffix.lower()

    if ext in {".yml", ".yaml"}:
        try:
            import yaml  # PyYAML
        except ModuleNotFoundError as e:
            msg = "YAML file provided but PyYAML is not installed. Install with: uv pip install '.[yaml]'"
            raise RuntimeError(msg) from e
        return yaml.safe_load(text) or {}

    if ext == ".json":
        return json.loads(text)

    if ext == ".toml":
        try:
            import tomllib  # Python 3.11+
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore[no-redef]
        return tomllib.loads(text)  # type: ignore[attr-defined]

    # Fallback: try JSON then YAML
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # PyYAML

            return yaml.safe_load(text) or {}
        except Exception as e:
            msg = f"Could not parse config file: {p}"
            raise RuntimeError(msg) from e
