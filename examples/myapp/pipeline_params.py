"""Pipeline using centralized TunableParameters classes.

Functions access values via class attributes (e.g., Model.hidden_units) and
are registered by inferring namespace from default expressions like
`param=Model.hidden_units`.
"""

from __future__ import annotations

from tunablex import tunable

from .params import Main, Model, Preprocess, Train


@tunable("hidden_units", "dropout", "root_param", apps=("train",))
def build_model(
    hidden_units=Model.hidden_units,
    dropout=Model.dropout,
    root_param=Main.root_param,
):
    """Build the model using centralized parameters."""
    print("build_model", hidden_units, dropout, root_param)
    return "model"


@tunable("epochs", "batch_size", "optimizer", apps=("train",))
def train(
    epochs=Train.epochs,
    batch_size=Train.batch_size,
    optimizer=Train.optimizer,
):
    """Train the model using centralized parameters."""
    print("train", epochs, batch_size, optimizer)


@tunable("dropna", "normalize", "clip_outliers", apps=("train", "serve"))
def preprocess(
    path: str,
    dropna=Preprocess.dropna,
    normalize=Preprocess.normalize,
    clip_outliers=Preprocess.clip_outliers,
):
    """Preprocess the dataset using centralized parameters."""
    print("preprocess", dropna, normalize, clip_outliers, "on", path)
    return "clean"


def train_main():
    """End-to-end train entrypoint using centralized params."""
    preprocess("/data/train.csv")
    build_model()
    train()
