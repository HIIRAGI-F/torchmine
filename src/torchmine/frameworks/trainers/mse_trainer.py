from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ...contracts import ModelBase, TrainerBase


class MSETrainer(TrainerBase):
    """Train a tensor-prediction model from `(input, target)` batches."""

    name = "mse"

    def __init__(
        self,
        model: ModelBase,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        *,
        grad_clip: float | None = None,
    ) -> None:
        super().__init__(model=model, optimizer=optimizer, device=device)
        self.grad_clip = None if grad_clip is None else float(grad_clip)
        self._mse = nn.MSELoss()

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_samples = 0

        for xb, yb in loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            pred = self.model(xb)
            loss = self._mse(pred, yb)
            loss.backward()

            if self.grad_clip is not None and self.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            batch = xb.size(0)
            total_loss += loss.item() * batch
            n_samples += batch

        if n_samples == 0:
            raise ValueError("Training loader must contain at least one sample")
        return total_loss / n_samples

    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n_samples = 0

        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                pred = self.model(xb)
                loss = self._mse(pred, yb)
                batch = xb.size(0)
                total_loss += loss.item() * batch
                n_samples += batch

        if n_samples == 0:
            raise ValueError("Evaluation loader must contain at least one sample")
        return total_loss / n_samples
