"""Centralized tunable parameters via inheritance-based namespace definition.

TunableParameters classes define parameter schemas and their default values.
The class hierarchy determines the namespace structure:

- Main(TunableParameters) -> root level (no namespace prefix)
- Model(Main) -> "model" namespace
- Preprocess(Model) -> "model.preprocess" namespace
- Train(Main) -> "train" namespace

Class names are converted to lowercase for namespaces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from pydantic import Field


class TunableParameters:
    """Base class for centralized tunable parameter definitions.

    Subclasses define parameter schemas via Pydantic Field annotations.
    The class hierarchy implicitly defines the namespace structure.

    Example:
        class Main(TunableParameters):
            dropout: float = Field(0.2, description="Dropout probability")

        class Model(Main):
            hidden_units: int = Field(128, description="Hidden units")

        # Model.hidden_units gives access to the default value
        # and provides namespace inference for @tunable decorators
    """

    def __init_subclass__(cls, **kwargs):
        """Register this parameter class and setup class attributes."""
        super().__init_subclass__(**kwargs)
        _register_parameter_class(cls)

    def __class_getitem__(cls, item):
        # This enables type annotations like TunableParameters[SomeConfig]
        return cls


def _register_parameter_class(cls: type[TunableParameters]) -> None:
    """Register a TunableParameters subclass and setup attribute access."""
    # Get all fields from this class and its bases
    fields = _collect_fields_from_class(cls)

    # Calculate namespace from class hierarchy
    namespace = _calculate_namespace_from_class(cls)

    # Store metadata on the class
    cls._tunablex_namespace = namespace  # type: ignore[attr-defined]
    cls._tunablex_fields = fields  # type: ignore[attr-defined]

    # Setup class attribute access for field defaults
    for field_name, field_info in fields.items():
        _setup_class_attribute_access(cls, field_name, field_info)


def _collect_fields_from_class(cls: type[TunableParameters]) -> dict[str, Any]:
    """Collect all Pydantic Field annotations from a class and its bases."""
    fields = {}

    # Walk the MRO to collect fields (most specific first)
    for base in reversed(cls.__mro__):
        if base is TunableParameters or base is object:
            continue

        # Get annotations for this class level
        annotations = getattr(base, "__annotations__", {})

        for name in annotations:
            if name.startswith("_"):
                continue

            # Get the default value (should be a Field)
            default_value = getattr(base, name, None)
            if default_value is not None:
                fields[name] = default_value

    return fields


def _calculate_namespace_from_class(cls: type[TunableParameters]) -> str:
    """Calculate the namespace from the class hierarchy.

    Rules:
    - Main -> root level (empty namespace)
    - Other classes -> lowercase class name
    - Inheritance creates dotted paths: Model(Main) -> "model", Preprocess(Model) -> "model.preprocess"
    """
    # Get the inheritance chain
    hierarchy = []
    for base in cls.__mro__:
        if base is TunableParameters or base is object:
            break
        hierarchy.append(base.__name__)

    # Reverse to go from most general to most specific
    hierarchy.reverse()

    # Convert to namespace parts
    namespace_parts = []
    for class_name in hierarchy:
        if class_name.lower() == "main":
            # Main class represents root level - don't add to namespace path
            continue
        namespace_parts.append(class_name.lower())

    # Join with dots, or return empty string if empty (for Main class - root level)
    if not namespace_parts:
        return ""  # Root level - no namespace prefix
    return ".".join(namespace_parts)


def _setup_class_attribute_access(cls: type[TunableParameters], field_name: str, field_info: Any) -> None:
    """Setup class attribute access to return the default value of a Field."""
    # Extract the default value from the Field
    default_value = field_info.default if hasattr(field_info, "default") else field_info

    # Store the default value as a class attribute
    setattr(cls, field_name, default_value)


def get_parameter_class_namespace(cls: type[TunableParameters]) -> str:
    """Get the namespace for a TunableParameters class."""
    return getattr(cls, "_tunablex_namespace", "")


def get_parameter_class_fields(cls: type[TunableParameters]) -> dict[str, Any]:
    """Get the fields defined by a TunableParameters class."""
    return getattr(cls, "_tunablex_fields", {})
