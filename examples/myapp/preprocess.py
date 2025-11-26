"""Preprocessing step with tunable parameters under a nested namespace."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from tunablex import tunable

from .pipeline_submodule import SubmoduleClass
from .pipeline_submodule import submodule_fun


@tunable(
    "dropna",
    "normalize",
    "clip_outliers",
    namespace="model.preprocess",
    apps=("train", "serve"),
)
def preprocess(
    path: str,
    dropna: bool = Field(True, description="Drop rows with missing values"),
    normalize: Literal["zscore", "minmax", "none"] = Field("zscore", description="Normalization strategy"),
    clip_outliers: float = Field(3.0, ge=0, le=10, description="Clip values beyond k standard deviations"),
):
    """Run preprocessing on the given dataset path."""
    submodule_fun()
    c = SubmoduleClass()
    c.method()
    print("preprocess", dropna, normalize, clip_outliers, "on", path)
    return "clean"
