"""Centralized tunable parameters using TunableParameters inheritance.

Main -> base namespace ("main")
Model(Main) -> "model"
Preprocess(Model) -> "model.preprocess"
Train(Main) -> "train"
"""
from __future__ import annotations

from typing import Literal

from pydantic import Field

from tunablex import TunableParameters


class Main(TunableParameters):
    """Root for centralized tunable namespaces ("main")."""
    root_param: str = Field("default", description="A root-level parameter")



class Model(Main):
    """Model hyper-parameters under the "model" namespace."""
    hidden_units: int = Field(128, ge=1, description="Number of hidden units")
    dropout: float = Field(0.2, ge=0.0, le=1.0, description="Dropout probability")


class Preprocess(Model):
    """Preprocess options nested under "model.preprocess"."""
    dropna: bool = Field(True, description="Drop rows with missing values")
    normalize: Literal["zscore", "minmax", "none"] = Field(
        "zscore", description="Normalization strategy"
    )
    clip_outliers: float = Field(3.0, ge=0, le=10, description="Clip values beyond k standard deviations")


class Train(Main):
    """Training loop parameters under the "train" namespace."""
    epochs: int = Field(10, ge=1, description="Number of training epochs")
    batch_size: int = Field(32, ge=1, description="Training batch size")
    optimizer: Literal["adam", "sgd"] = Field("adam", description="Optimizer choice")
