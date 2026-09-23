from __future__ import annotations

from .models.conv2d_classifier import Conv2DClassifier
from .trainers.mse_trainer import MSETrainer
from .processors.identity import IdentityProcessor
from .processors.normalizer import NormalizerProcessor


# If you add new framework components like model, trainer or processor, you must register that framework to the following function.
def register_components(registry) -> None:
    # model
    registry.register_model("conv2d_classifier", Conv2DClassifier)

    # trainer
    registry.register_trainer("mse", MSETrainer)

    # processor
    registry.register_processor("identity", IdentityProcessor)
    registry.register_processor("normalizer", NormalizerProcessor)
