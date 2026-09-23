from __future__ import annotations

import torch

from ...contracts import ProcessorBase


class IdentityProcessor(ProcessorBase):
    """Return tensor data unchanged in either direction."""

    name = "identity"

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        if not isinstance(data, torch.Tensor):
            raise TypeError("IdentityProcessor expects a torch.Tensor")
        return data

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Leave target data unchanged when no inverse operation is needed."""
        return self.transform(data)
