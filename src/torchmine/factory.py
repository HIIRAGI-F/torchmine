from __future__ import annotations

from importlib import import_module
from typing import Any, Mapping

import torch

from .contracts import ModelBase, ProcessorBase, TrainerBase

ModelType = type[ModelBase]
ProcessorType = type[ProcessorBase]
TrainerType = type[TrainerBase]
OptimizerType = type[torch.optim.Optimizer]


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

    def resolve_trainer_type(self, *, trainer_name: str) -> TrainerType:
        self._ensure_loaded()
        if trainer_name not in self._trainer_types:
            raise ValueError(
                f"Unsupported trainer '{trainer_name}'. Available trainers: {self.available_trainers()}"
            )
        return self._trainer_types[trainer_name]

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

    def resolve_components(
        self,
        *,
        model_name: str,
        trainer_name: str,
        model_params: Mapping[str, Any] | None = None,
    ) -> tuple[ModelBase, TrainerType]:
        model = self.resolve_model(model_name=model_name, model_params=model_params)
        trainer_type = self.resolve_trainer_type(trainer_name=trainer_name)
        return model, trainer_type

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


def resolve_components(
    *,
    model_name: str,
    trainer_name: str,
    model_params: Mapping[str, Any] | None = None,
) -> tuple[ModelBase, TrainerType]:
    return REGISTRY.resolve_components(
        model_name=model_name,
        model_params=model_params,
        trainer_name=trainer_name,
    )


def resolve_model(
    *,
    model_name: str,
    model_params: Mapping[str, Any] | None = None,
) -> ModelBase:
    return REGISTRY.resolve_model(
        model_name=model_name,
        model_params=model_params,
    )


def resolve_trainer_type(*, trainer_name: str) -> TrainerType:
    return REGISTRY.resolve_trainer_type(trainer_name=trainer_name)


def resolve_processor(
    *,
    processor_name: str,
    processor_params: Mapping[str, Any] | None = None,
) -> ProcessorBase:
    return REGISTRY.resolve_processor(
        processor_name=processor_name,
        processor_params=processor_params,
    )


def build_trainer(
    *,
    trainer_type: TrainerType,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    trainer_params: Mapping[str, Any] | None = None,
) -> TrainerBase:
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


def build_training_components(
    *,
    model_name: str,
    trainer_name: str,
    model_params: Mapping[str, Any] | None = None,
    trainer_params: Mapping[str, Any] | None = None,
    optimizer_cls: OptimizerType = torch.optim.Adam,
    optimizer_params: Mapping[str, Any] | None = None,
    device: str | torch.device | None = None,
) -> tuple[ModelBase, torch.optim.Optimizer, TrainerBase]:
    target_device = torch.device(device) if device is not None else torch.device("cpu")
    model, trainer_type = resolve_components(
        model_name=model_name,
        model_params=model_params,
        trainer_name=trainer_name,
    )
    model = model.to(target_device)

    optimizer_kwargs = dict(optimizer_params or {})
    try:
        optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
    except TypeError as e:
        raise ValueError(
            f"Invalid optimizer_params for optimizer '{optimizer_cls.__name__}': {optimizer_kwargs}"
        ) from e

    trainer = build_trainer(
        trainer_type=trainer_type,
        model=model,
        optimizer=optimizer,
        device=target_device,
        trainer_params=trainer_params,
    )
    return model, optimizer, trainer


def available_models() -> list[str]:
    return REGISTRY.available_models()


def available_processors() -> list[str]:
    return REGISTRY.available_processors()


def available_trainers() -> list[str]:
    return REGISTRY.available_trainers()
