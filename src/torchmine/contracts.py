from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Mapping, Self, TypeVar

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ----------------------------------------------------------------------------------
# frameworks contracts
# ----------------------------------------------------------------------------------
# 1. For models
class ModelBase(nn.Module, ABC):
    """Contract for a registered model that predicts one tensor from another.

    PyTorch's nn.Module supplies parameter, device, mode, and state management.
    Subclasses define their layers and implement forward() for prediction.
    """

    name: str

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a tensor prediction for a tensor input."""
        raise NotImplementedError

# 2. For trainers
class TrainerBase(ABC):
    """Base class for the training logic paired with a registered model.
    ---
    ### Subclasses must implement:
    - `train_epoch(loader)`: Train the model for one epoch and return the loss.
    - `evaluate(loader)`: Evaluate the model and return the loss.
    
    Both methods define how a batch is interpreted, how its loss is computed 
    and return a comparable scalar loss.
    """

    name: str

    def __init__(
        self,
        model: ModelBase,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device

    @abstractmethod
    def train_epoch(self, loader: DataLoader) -> float:
        """Update the model for one epoch and return a finite scalar loss."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, loader: DataLoader) -> float:
        """Return a finite scalar loss without updating model parameters."""
        raise NotImplementedError

# 3. For processors
class ProcessorBase(ABC):
    """Contract for a processor that transforms tensors.
    ---
    ### Functions provided by this base class:
    - `train()`: Set the processor to training mode.
    - `eval()`: Set the processor to evaluation or inference mode.
    - `fit(data)`: Learn any values needed for the transformation from
        training data. Stateless processors do not need to override this.
    
    ### Subclasses must implement:
    - `transform(data)`: Transform the input data and return the result.

    ### Subclasses may implement when needed:
    - `inverse_transform(data)`: Restore transformed data when the processor
      is used for targets. The default raises so a missing implementation
      cannot silently change the meaning of predictions.
    - `state_dict()` / `load_state_dict(state)`: Save and restore learned state.
    """

    name: str

    def __init__(self) -> None:
        self.training = True

    def train(self) -> Self:
        """Set the processor to training mode."""
        self.training = True
        return self

    def eval(self) -> Self:
        """Set the processor to evaluation mode."""
        self.training = False
        return self

    def fit(self, data: torch.Tensor) -> Self:
        """Learn from training data; stateless processors can use this default."""
        return self

    @abstractmethod
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        return self.fit(data).transform(data)

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Restore target data; override when this processor is used for targets."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement inverse_transform"
        )
    
    def state_dict(self) -> dict[str, Any]:
        """Return the processor state. Override for stateful processors."""
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore processor state. Override for stateful processors."""
        if state:
            raise ValueError("This processor does not support state restoration")
        
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return self.transform(data)


# ----------------------------------------------------------------------------------
# pipelines contracts
# ----------------------------------------------------------------------------------

TrainDataT = TypeVar("TrainDataT")
TrainResultT = TypeVar("TrainResultT")

class PipelineBase(
    ABC,
    Generic[TrainDataT, TrainResultT],
):
    """Contract for a pipeline that trains and predicts tensors.
    ---
    ### Subclasses must implement:
        - train_data(data): Train the pipeline and return the training result.
        - predict_data(data): Run prediction and return the result.
    """

    name: str

    @abstractmethod
    def train_data(self, data: TrainDataT) -> TrainResultT:
        raise NotImplementedError

    @abstractmethod
    def predict_data(self, data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
