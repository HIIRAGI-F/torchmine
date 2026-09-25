"""Load an MNIST checkpoint and plot losses and five predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from torchvision.datasets import MNIST

from torchmine.pipelines import WorkflowEarlyStoppingPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=Path("scripts/mnist.yaml"))
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as stream:
        settings = yaml.safe_load(stream)

    pipeline = WorkflowEarlyStoppingPipeline().load_model(args.checkpoint, device="cpu")
    test_set = MNIST(root=settings["dataset_dir"], train=False, download=True)
    images = test_set.data[:5].unsqueeze(1)
    predictions = pipeline.predict_data(images).argmax(dim=1).tolist()
    targets = test_set.targets[:5].tolist()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    train_loss = checkpoint["train_loss_history"]
    val_loss = checkpoint["val_loss_history"]
    epochs = range(1, len(train_loss) + 1)

    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot(epochs, train_loss, label="train")
    axis.plot(epochs, val_loss, label="validation")
    axis.set(xlabel="Epoch", ylabel="MSE loss", title="MNIST training")
    axis.legend()
    fig.tight_layout()
    loss_path = args.checkpoint.parent / "loss.png"
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(10, 2.5))
    for index, axis in enumerate(axes):
        axis.imshow(test_set.data[index], cmap="gray")
        axis.set_title(f"pred: {predictions[index]}\ntrue: {targets[index]}")
        axis.axis("off")
    fig.tight_layout()
    predictions_path = args.checkpoint.parent / "predictions.png"
    fig.savefig(predictions_path, dpi=150)
    plt.close(fig)

    print(f"loss plot: {loss_path}")
    print(f"predictions plot: {predictions_path}")


if __name__ == "__main__":
    main()
