from __future__ import annotations

from importlib import import_module
from typing import Any, Mapping

import torch

from .contracts import ModelBase, ProcessorBase, TrainerBase

ModelType = type[ModelBase]
ProcessorType = type[ProcessorBase]
TrainerType = type[TrainerBase]


# Internal class to resolve components from the registry and build the training pipeline
class ComponentRegistry:
    def __init__(self) -> None:
        self._model_types: dict[str, ModelType] = {}
        self._processor_types: dict[str, ProcessorType] = {}
        self._trainer_types: dict[str, TrainerType] = {}
        self._loaded = False

    def register_model(self, model_name: str, model_type: ModelType) -> None:
        self._model_types[model_name] = model_type

    def register_processor(self, processor_name: str, processor_type: ProcessorType) -> None:
        self._processor_types[processor_name] = processor_type

    def register_trainer(self, trainer_name: str, trainer_type: TrainerType) -> None:
        self._trainer_types[trainer_name] = trainer_type

    def available_models(self) -> list[str]:
        self._ensure_loaded()
        return sorted(self._model_types.keys())

    def available_processors(self) -> list[str]:
        self._ensure_loaded()
        return sorted(self._processor_types.keys())

    def available_trainers(self) -> list[str]:
        self._ensure_loaded()
        return sorted(self._trainer_types.keys())

    def resolve_model(
        self,
        *,
        model_name: str,
        model_params: Mapping[str, Any] | None = None,
    ) -> ModelBase:
        self._ensure_loaded()
        if model_name not in self._model_types:
            raise ValueError(
                f"Unsupported model '{model_name}'. Available models: {self.available_models()}"
            )

        model_kwargs = dict(model_params or {})
        model_type = self._model_types[model_name]
        try:
            return model_type(**model_kwargs)
        except TypeError as e:
            raise ValueError(f"Invalid model_params for model '{model_name}': {model_kwargs}") from e

    def resolve_trainer(
        self,
        *,
        trainer_name: str,
        model: ModelBase,
        optimizer: torch.optim.Optimizer,
        device: str | torch.device,
        trainer_params: Mapping[str, Any] | None = None,
    ) -> TrainerBase:
        self._ensure_loaded()
        if trainer_name not in self._trainer_types:
            raise ValueError(
                f"Unsupported trainer '{trainer_name}'. Available trainers: {self.available_trainers()}"
            )

        trainer_type = self._trainer_types[trainer_name]
        target_device = torch.device(device)
        trainer_kwargs = dict(trainer_params or {})
        try:
            return trainer_type(
                model=model,
                optimizer=optimizer,
                device=target_device,
                **trainer_kwargs,
            )
        except TypeError as e:
            raise ValueError(
                f"Invalid trainer_params for trainer '{trainer_type.__name__}': {trainer_kwargs}"
            ) from e

    def resolve_processor(
        self,
        *,
        processor_name: str,
        processor_params: Mapping[str, Any] | None = None,
    ) -> ProcessorBase:
        self._ensure_loaded()
        if processor_name not in self._processor_types:
            raise ValueError(
                f"Unsupported processor '{processor_name}'. Available processors: {self.available_processors()}"
            )

        processor_kwargs = dict(processor_params or {})
        processor_type = self._processor_types[processor_name]
        try:
            return processor_type(**processor_kwargs)
        except TypeError as e:
            raise ValueError(
                f"Invalid params for processor '{processor_name}': {processor_kwargs}"
            ) from e

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        module = import_module("torchmine.frameworks.registry")
        register = getattr(module, "register_components", None)
        if not callable(register):
            raise ValueError("torchmine.frameworks.registry must expose register_components(registry)")
        register(self)
        self._loaded = True

REGISTRY = ComponentRegistry()


def resolve_model(
    *,
    model_name: str,
    model_params: Mapping[str, Any] | None = None,
) -> ModelBase:
    """Resolve and build a model from the registry."""
    return REGISTRY.resolve_model(
        model_name=model_name,
        model_params=model_params,
    )

def resolve_trainer(
    *,
    trainer_name: str,
    model: ModelBase,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    trainer_params: Mapping[str, Any] | None = None,
) -> TrainerBase:
    """Resolve and build a trainer from the registry."""
    return REGISTRY.resolve_trainer(
        trainer_name=trainer_name,
        model=model,
        optimizer=optimizer,
        device=device,
        trainer_params=trainer_params,
    )

def resolve_processor(
    *,
    processor_name: str,
    processor_params: Mapping[str, Any] | None = None,
) -> ProcessorBase:
    """Resolve and build a processor from the registry."""
    return REGISTRY.resolve_processor(
        processor_name=processor_name,
        processor_params=processor_params,
    )


def available_models() -> list[str]:
    """Show the names of all registered models."""
    return REGISTRY.available_models()

def available_trainers() -> list[str]:
    """Show the names of all registered trainers."""
    return REGISTRY.available_trainers()

def available_processors() -> list[str]:
    """Show the names of all registered processors."""
    return REGISTRY.available_processors()
