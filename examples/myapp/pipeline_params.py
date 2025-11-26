"""Pipeline using centralized TunableParameters classes.

Functions access values via class attributes (e.g., Model.hidden_units) and
are registered by inferring namespace from default expressions like
`param=Model.hidden_units`.
"""

from __future__ import annotations

from typing import Literal

from tunablex import tunable

from .params import MainParams
from .params import ModelParams
from .params import ServeParams
from .params import TrainParams
from .pipeline_submodule import SubmoduleClass
from .pipeline_submodule import submodule_fun

Preprocess = ModelParams.Preprocess


@tunable("hidden_units", "dropout", "agg", apps="train")
@tunable("batch_norm", apps="train")
@tunable("root_param", apps="train")
def build_model(
    hidden_units=ModelParams.hidden_sizes,
    dropout=ModelParams.dropout,
    agg=ModelParams.agg,
    batch_norm: bool = True,
    root_param=MainParams.root_param,
):
    """Build the model using centralized parameters."""
    print("build_model", hidden_units, dropout, agg, batch_norm, root_param)
    return "model"


@tunable("epochs", "batch_size", apps="train")
@tunable("optimizer", namespace="train", apps="train")
def train(epochs=TrainParams.epochs, batch_size=TrainParams.batch_size, optimizer: Literal["adam", "sgd"] = "adam"):
    """Train the model using centralized parameters."""
    print("train", epochs, batch_size, optimizer)


@tunable("nb_epochs", "other_batch_size", apps="train")
@tunable("optimizer", namespace="train", apps="train")
def other_train(
    nb_epochs=TrainParams.epochs, other_batch_size=TrainParams.batch_size, optimizer: Literal["adam", "sgd"] = "adam"
):
    """Train the model using centralized parameters with other argument names than the reference ones."""
    print("other_train", nb_epochs, other_batch_size, optimizer)


@tunable("dropna", "normalize", "clip_outliers", apps=("train", "serve"))
def preprocess(
    path: str, dropna=Preprocess.dropna, normalize=Preprocess.normalize, clip_outliers=Preprocess.clip_outliers
):
    """Preprocess the dataset using centralized parameters."""
    submodule_fun()
    c = SubmoduleClass()
    c.method()
    print("preprocess", dropna, normalize, clip_outliers, "on", path)
    return "clean"


@tunable("port", "workers", apps="serve")
def serve_api(port=ServeParams.port, workers=ServeParams.workers):
    print("serve_api", port, workers)


def train_main():
    """End-to-end train entrypoint using centralized params."""
    preprocess("/data/train.csv")
    build_model()
    train()
    other_train()


def serve_main():
    preprocess("/data/live.csv")
    serve_api()
