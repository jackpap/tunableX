from __future__ import annotations

from typing import Literal

from pydantic import Field

from tunablex import tunable

from .preprocess import preprocess


@tunable("hidden_units", "dropout", namespace="model", apps=("train",))
@tunable("root_param", apps=("train",))
def build_model(
    hidden_units: int = Field(128, ge=1, description="Number of hidden units"),
    dropout: float = Field(0.2, ge=0.0, le=1.0, description="Dropout probability"),
    root_param: str = Field("default", description="A root-level parameter"),
):
    print("build_model", hidden_units, dropout, root_param)
    return "model"


@tunable("epochs", "batch_size", "optimizer", namespace="train", apps=("train",))
def train(
    epochs: int = Field(10, ge=1, description="Number of training epochs"),
    batch_size: int = Field(32, ge=1, description="Training batch size"),
    optimizer: Literal["adam", "sgd"] = Field("adam", description="Optimizer choice"),
):
    print("train", epochs, batch_size, optimizer)


@tunable("port", "workers", namespace="serve", apps=("serve",))
def serve_api(
    port: int = Field(8080, ge=1, le=65535, description="HTTP port for serving API"),
    workers: int = Field(2, ge=1, description="Number of worker processes"),
):
    print("serve_api", port, workers)


def train_main():
    preprocess("/data/train.csv")
    build_model()
    train()


def serve_main():
    preprocess("/data/live.csv")
    serve_api()
