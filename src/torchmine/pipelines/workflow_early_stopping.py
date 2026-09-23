from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
import math
from pathlib import Path
from typing import Any, Mapping, Self

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from ..contracts import ModelBase, PipelineBase, ProcessorBase
from ..factory import build_training_components, resolve_model, resolve_processor


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ProcessorConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WorkflowConfig:
    model_name: str
    trainer_name: str
    model_params: dict[str, Any] = field(default_factory=dict)
    trainer_params: dict[str, Any] = field(default_factory=dict)
    input_processors: list[ProcessorConfig] = field(default_factory=list)
    train_processors: list[ProcessorConfig] = field(default_factory=list)
    target_processors: list[ProcessorConfig] = field(default_factory=list)
    output_processors: list[ProcessorConfig] = field(default_factory=list)
    out_dir: Path = Path("data/checkpoints/train")
    output: Path | None = None
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    eps: float = 1e-8
    shuffle: bool = True
    seed: int = 1234
    device: str | None = None
    num_workers: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.0
    restore_best: bool = True
    save_best: bool = True
    save_last: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert paths and dataclasses to checkpoint-safe values."""
        return _plain(asdict(self))


@dataclass(frozen=True, slots=True)
class WorkflowTrainingData:
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    checkpoint_payload: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class WorkflowResults:
    run_dir: Path
    checkpoint_path: Path | None
    best_checkpoint_path: Path | None
    last_checkpoint_path: Path | None
    config_path: Path
    final_train_loss: float
    final_val_loss: float
    min_val_loss: float
    best_epoch: int
    stopped_epoch: int | None
    train_loss_history: list[float]
    val_loss_history: list[float]

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to plain Python values."""
        return _plain(asdict(self))


ProcessorEntry = tuple[ProcessorConfig, ProcessorBase]


class WorkflowEarlyStoppingPipeline(
    PipelineBase[WorkflowTrainingData, WorkflowResults],
):
    """Train one registered model with optional preprocessing and early stopping."""

    name = "workflow_early_stopping"

    def __init__(self, config: WorkflowConfig | None = None) -> None:
        self.config: WorkflowConfig | None = None
        self.device = torch.device("cpu")
        self.model: ModelBase | None = None
        self.is_trained = False
        self.input_processors: list[ProcessorEntry] = []
        self.train_processors: list[ProcessorEntry] = []
        self.target_processors: list[ProcessorEntry] = []
        self.output_processors: list[ProcessorEntry] = []
        if config is not None:
            self._configure(config)

    # Training Phase
    def train_data(self, data: WorkflowTrainingData) -> WorkflowResults:
        if self.config is None:
            raise ValueError("Workflow config is not set")
        config = self.config
        torch.manual_seed(int(config.seed))
        train_x = self._fit_transform(self.input_processors, data.train_x)
        val_x = self._transform(self.input_processors, data.val_x)
        train_x = self._fit_transform(self.train_processors, train_x, training=True)
        train_y = self._fit_transform(self.target_processors, data.train_y)
        val_y = self._transform(self.target_processors, data.val_y)
        train_loader = DataLoader(
            TensorDataset(train_x, train_y),
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers,
        )
        val_loader = DataLoader(
            TensorDataset(val_x, val_y),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
        extra_payload = data.checkpoint_payload
        model, _, trainer = build_training_components(
            model_name=config.model_name,
            model_params=config.model_params,
            trainer_name=config.trainer_name,
            trainer_params=config.trainer_params,
            optimizer_params={"lr": config.lr, "eps": config.eps},
            device=self.device,
        )
        self.model = model
        self.is_trained = False
        run_dir = Path(config.out_dir) / datetime.now().strftime("train_%Y%m%d_%H%M%S_%f")
        run_dir.mkdir(parents=True, exist_ok=False)
        config_path = run_dir / "config.yaml"
        with config_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(config.to_dict(), stream, sort_keys=False)

        train_history: list[float] = []
        val_history: list[float] = []
        best_loss = float("inf")
        best_epoch = 0
        bad_epochs = 0
        stopped_epoch: int | None = None
        best_state: dict[str, torch.Tensor] | None = None
        best_path: Path | None = None
        for epoch in range(1, config.epochs + 1):
            train_loss = float(trainer.train_epoch(train_loader))
            val_loss = float(trainer.evaluate(val_loader))
            if not math.isfinite(train_loss) or not math.isfinite(val_loss):
                raise ValueError(f"Non-finite loss at epoch {epoch}")
            train_history.append(train_loss)
            val_history.append(val_loss)
            print(
                f"Epoch {epoch}/{config.epochs} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}",
                flush=True,
            )
            if val_loss < best_loss - config.min_delta:
                best_loss = val_loss
                best_epoch = epoch
                bad_epochs = 0
                if config.restore_best:
                    best_state = copy.deepcopy(model.state_dict())
                if config.save_best:
                    best_path = run_dir / self._checkpoint_name("best")
                    self._save(best_path, epoch, best_epoch, None, train_history, val_history, extra_payload)
            else:
                bad_epochs += 1
            if config.early_stopping and bad_epochs >= config.patience:
                stopped_epoch = epoch
                print(f"Early stopping at epoch {epoch} (best: {best_epoch})", flush=True)
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self.is_trained = True
        last_path: Path | None = None
        if config.save_last:
            last_path = run_dir / self._checkpoint_name("last")
            self._save(last_path, len(train_history), best_epoch, stopped_epoch, train_history, val_history, extra_payload)
        
        return WorkflowResults(
            run_dir=run_dir,
            checkpoint_path=best_path or last_path,
            best_checkpoint_path=best_path,
            last_checkpoint_path=last_path,
            config_path=config_path,
            final_train_loss=train_history[-1],
            final_val_loss=val_history[-1],
            min_val_loss=min(val_history),
            best_epoch=best_epoch,
            stopped_epoch=stopped_epoch,
            train_loss_history=train_history,
            val_loss_history=val_history,
        )

    # Inference Phase
    def load_model(
        self, path: str | Path, strict: bool = True, eval_mode: bool = True, *, device: str | None = None,
    ) -> Self:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise TypeError("Checkpoint must be a dictionary")
        config_data = checkpoint.get("workflow_config")
        state = checkpoint.get("model_state")
        if not isinstance(config_data, Mapping) or not isinstance(state, dict):
            raise TypeError("Checkpoint is missing workflow_config or model_state")
        values = dict(config_data)
        for key in ("input_processors", "train_processors", "target_processors", "output_processors"):
            values[key] = [
                ProcessorConfig(name=str(item["name"]), params=dict(item.get("params", {})))
                for item in values.get(key, [])
            ]
        config = WorkflowConfig(**values)
        self._configure(replace(config, device=device) if device is not None else config)
        self.model = resolve_model(model_name=config.model_name, model_params=config.model_params).to(self.device)
        self.model.load_state_dict(state, strict=strict)
        self._load_processor_states(checkpoint.get("processors"))
        if eval_mode:
            self.model.eval()
        self.is_trained = True
        return self

    # Inference Phase
    @torch.inference_mode()
    def predict_data(self, data: torch.Tensor) -> torch.Tensor:
        if not self.is_trained or self.model is None:
            raise ValueError("Train or load a model before prediction")
        model = self.model
        x = self._transform(self.input_processors, data.to(self.device))
        model.eval()
        predicted = model(x)
        predicted = self._transform(
            self.output_processors,
            self._inverse(self.target_processors, predicted),
        )
        if predicted.device != x.device:
            raise ValueError("Processors must preserve the model device during prediction")
        return predicted

    # Initialization
    def _configure(self, config: WorkflowConfig) -> None:
        if not config.model_name or not config.trainer_name:
            raise ValueError("model_name and trainer_name are required")
        if config.epochs < 1 or config.batch_size < 1 or config.patience < 1:
            raise ValueError("epochs, batch_size, and patience must be positive")
        if config.min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available")
        self.device = device
        self.config = replace(config, device=str(device))
        self.input_processors = self._build_processors(config.input_processors)
        self.train_processors = self._build_processors(config.train_processors)
        self.target_processors = self._build_processors(config.target_processors)
        self.output_processors = self._build_processors(config.output_processors)
        self.model = None
        self.is_trained = False

    # Initialization of processors
    @staticmethod
    def _build_processors(configs: list[ProcessorConfig]) -> list[ProcessorEntry]:
        entries: list[ProcessorEntry] = []
        for config in configs:
            if not config.name:
                raise ValueError("Processor name must not be empty")
            entries.append((config, resolve_processor(
                processor_name=config.name, processor_params=config.params,
            )))
        return entries

    # Resolve processors
    @staticmethod
    def _fit_transform(
        entries: list[ProcessorEntry], data: torch.Tensor, *, training: bool = False,
    ) -> torch.Tensor:
        for _, processor in entries:
            processor.fit(data)
            processor.train() if training else processor.eval()
            data = processor.transform(data)
        return data

    # Resolve processors
    @staticmethod
    def _transform(entries: list[ProcessorEntry], data: torch.Tensor) -> torch.Tensor:
        for _, processor in entries:
            processor.eval()
            data = processor.transform(data)
        return data

    # Resolve processors
    @staticmethod
    def _inverse(entries: list[ProcessorEntry], data: torch.Tensor) -> torch.Tensor:
        for _, processor in reversed(entries):
            data = processor.inverse_transform(data)
        return data

    # Resolve processors
    def _load_processor_states(self, payload: Any) -> None:
        if not isinstance(payload, Mapping):
            raise TypeError("Checkpoint is missing processor states")
        for role, entries in (
            ("input_processors", self.input_processors),
            ("train_processors", self.train_processors),
            ("target_processors", self.target_processors),
            ("output_processors", self.output_processors),
        ):
            states = payload.get(role)
            if not isinstance(states, list) or len(states) != len(entries):
                raise ValueError(f"Checkpoint processor count differs for {role}")
            for (_, processor), state in zip(entries, states, strict=True):
                if not isinstance(state, Mapping):
                    raise TypeError(f"Invalid processor state in {role}")
                processor.load_state_dict(state)

    # Chore
    def _checkpoint_name(self, kind: str) -> str:
        config = self.config
        if config.output is not None:
            output = Path(config.output)
            return f"{output.stem}_{kind}{output.suffix or '.pt'}"
        return f"train_{kind}_tr{config.trainer_name}_m{config.model_name}.pt"

    # Chore
    def _save(
        self,
        path: Path,
        epoch: int,
        best_epoch: int,
        stopped_epoch: int | None,
        train_history: list[float],
        val_history: list[float],
        extra: Mapping[str, Any] | None,
    ) -> None:
        config = self.config
        model = self.model
        if model is None:
            raise RuntimeError("Model is missing")

        processor_states: dict[str, list[dict[str, Any]]] = {}
        for role, entries in (
            ("input_processors", self.input_processors),
            ("train_processors", self.train_processors),
            ("target_processors", self.target_processors),
            ("output_processors", self.output_processors),
        ):
            processor_states[role] = []
            for _, processor in entries:
                state = processor.state_dict()
                if not isinstance(state, Mapping):
                    raise TypeError(f"{type(processor).__name__}.state_dict must return a mapping")
                processor_states[role].append(dict(state))
        
        payload: dict[str, Any] = {
            "workflow_config": config.to_dict(),
            "model_state": model.state_dict(),
            "processors": processor_states,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "train_loss_history": list(train_history),
            "val_loss_history": list(val_history),
        }
        if extra:
            overlap = payload.keys() & extra.keys()
            if overlap:
                raise ValueError(f"checkpoint_payload cannot replace reserved keys: {sorted(overlap)}")
            payload.update(extra)
        torch.save(payload, path)
