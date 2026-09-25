"""Train a small MNIST classifier and save its checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torchvision.datasets import MNIST

from torchmine.pipelines import (
    ProcessorConfig,
    WorkflowConfig,
    WorkflowEarlyStoppingPipeline,
    WorkflowTrainingData,
)


def class_targets(labels: torch.Tensor) -> torch.Tensor:
    """Encode each digit label for the built-in MSE trainer."""
    return F.one_hot(labels, num_classes=10).float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("scripts/mnist.yaml"))
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as stream:
        settings = yaml.safe_load(stream)

    seed = settings["seed"]
    train_set = MNIST(root=settings["dataset_dir"], train=True, download=True)
    order = torch.randperm(len(train_set), generator=torch.Generator().manual_seed(seed))
    train_ids = order[:settings["train_samples"]]
    val_ids = order[
        settings["train_samples"]:settings["train_samples"] + settings["val_samples"]
    ]

    config = WorkflowConfig(
        model_name=settings["model_name"],
        trainer_name=settings["trainer_name"],
        model_params=settings["model_params"],
        input_processors=[ProcessorConfig(**item) for item in settings["input_processors"]],
        out_dir=Path(settings["out_dir"]),
        output=Path(settings["output"]),
        epochs=settings["epochs"],
        batch_size=settings["batch_size"],
        lr=settings["lr"],
        patience=settings["patience"],
        seed=seed,
    )
    pipeline = WorkflowEarlyStoppingPipeline(config)
    result = pipeline.train_data(WorkflowTrainingData(
        train_x=train_set.data[train_ids].unsqueeze(1),
        train_y=class_targets(train_set.targets[train_ids]),
        val_x=train_set.data[val_ids].unsqueeze(1),
        val_y=class_targets(train_set.targets[val_ids]),
    ))

    print(f"checkpoint: {result.checkpoint_path}")


if __name__ == "__main__":
    main()
