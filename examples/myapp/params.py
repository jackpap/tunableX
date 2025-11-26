"""Centralized tunable parameters using TunableParameters inheritance.

Main -> base namespace ("main")
Model(Main) -> "model"
Preprocess(Model) -> "model.preprocess"
Train(Main) -> "train"
"""

from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003
from typing import Literal

from pydantic import Field

from tunablex import TunableParams


class MainParams(TunableParams):
    """Root for centralized tunable namespaces ("main")."""

    root_param: str = Field("default", description="A root-level parameter")


class PreprocessParams(TunableParams):
    """Preprocess options nested under "model.preprocess".

    Doing so avoids having too many indentations when using several levels of namespaces.
    """

    dropna: bool = Field(True, description="Drop rows with missing values")
    normalize: Literal["zscore", "minmax", "none"] = Field("zscore", description="Normalization strategy")
    clip_outliers: float = Field(3.0, ge=0, le=10, description="Clip values beyond k standard deviations")


class ModelParams(TunableParams):
    """Model hyper-parameters under the "model" namespace."""

    hidden_sizes: Sequence[int] = Field((128, 128), description="Sizes of hidden layers")
    dropout: float = Field(0.2, ge=0.0, le=1.0, description="Dropout probability")
    agg: Literal["sum", "concat"] = Field("sum", description="Aggregation method")

    Preprocess = PreprocessParams
    """Preprocess options nested under "model.preprocess"."""


class TrainParams(TunableParams):
    """Training loop parameters under the "train" namespace."""

    epochs: int = Field(10, ge=1, description="Number of training epochs")
    batch_size: int = Field(32, ge=1, description="Training batch size")


class ServeParams(TunableParams):
    """Serve loop parameters."""

    port: int = Field(8080, ge=1, le=65535, description="HTTP port for serving API")
    workers: int = Field(2, ge=1, description="Number of worker processes")
