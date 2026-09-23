from __future__ import annotations

from typing import Any, Mapping, Self

import torch

from ...contracts import ProcessorBase


class NormalizerProcessor(ProcessorBase):
    """Normalize tensors using statistics fitted on training data."""

    name = "normalizer"

    def __init__(
        self,
        dims: int | list[int] | tuple[int, ...] = 0,
        *,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.dims = self._normalize_dims(dims)
        self.eps = float(eps)
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        self.fitted = False
        self._tensor_states: dict[tuple[torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}

    def fit(self, data: torch.Tensor) -> Self:
        tensor = data.float()
        dims = self._resolve_dims(tensor.ndim)
        self.mean = tensor.mean(dim=dims, keepdim=True).detach().cpu()
        std = tensor.std(dim=dims, unbiased=False, keepdim=True).detach().cpu()
        self.std = std.clamp_min(self.eps)
        self.fitted = True
        self._tensor_states.clear()
        return self

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        self._validate_fitted()
        tensor = data.float()
        mean, std = self._state_for_tensor(tensor)
        return (tensor - mean) / std

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        self._validate_fitted()
        tensor = data.float()
        mean, std = self._state_for_tensor(tensor)
        return tensor * std + mean

    def state_dict(self) -> dict[str, Any]:
        return {
            "dims": list(self.dims),
            "eps": self.eps,
            "mean": self.mean,
            "std": self.std,
            "fitted": self.fitted,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._tensor_states.clear()
        dims = state.get("dims", self.dims)
        self.dims = self._normalize_dims(dims)
        self.eps = float(state.get("eps", self.eps))
        self.mean = self._optional_tensor(state.get("mean"))
        self.std = self._optional_tensor(state.get("std"))
        self.fitted = bool(state.get("fitted", self.mean is not None and self.std is not None))

    @staticmethod
    def _normalize_dims(dims: int | list[int] | tuple[int, ...]) -> tuple[int, ...]:
        if isinstance(dims, int):
            return (dims,)
        if not dims:
            raise ValueError("dims must contain at least one dimension")
        return tuple(int(dim) for dim in dims)

    def _resolve_dims(self, ndim: int) -> tuple[int, ...]:
        resolved: list[int] = []
        for dim in self.dims:
            normalized = dim if dim >= 0 else ndim + dim
            if normalized < 0 or normalized >= ndim:
                raise ValueError(f"dims contains invalid dimension {dim} for ndim={ndim}")
            resolved.append(normalized)
        return tuple(sorted(set(resolved)))

    @staticmethod
    def _optional_tensor(value: Any) -> torch.Tensor | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        return torch.as_tensor(value, dtype=torch.float32).detach().cpu()

    def _state_for_tensor(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mean is None or self.std is None:
            raise ValueError("NormalizerProcessor is missing fitted state")
        # Avoid transferring fixed statistics on every step of an autoregressive run.
        key = (tensor.device, tensor.dtype)
        if key not in self._tensor_states:
            # Cached statistics may later be reused outside inference_mode.
            with torch.inference_mode(False):
                self._tensor_states[key] = (
                    self.mean.to(device=tensor.device, dtype=tensor.dtype),
                    self.std.to(device=tensor.device, dtype=tensor.dtype),
                )
        return self._tensor_states[key]

    def _validate_fitted(self) -> None:
        if not self.fitted or self.mean is None or self.std is None:
            raise ValueError("NormalizerProcessor must be fitted before transform")
