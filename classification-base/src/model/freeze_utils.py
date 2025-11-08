from enum import Enum
from typing import Optional, Tuple

import torch.nn as nn


class FreezeStrategy(Enum):
    """Enum describing backbone freezing strategies."""
    NO = "no"
    LAST = "last"          # freeze entire backbone, train classifier head only
    PCT70 = "freeze_70"    # freeze first 70% of backbone params


def _parse_freeze_strategy(freeze: Optional[FreezeStrategy]) -> FreezeStrategy:
    return freeze if isinstance(freeze, FreezeStrategy) else FreezeStrategy.NO


def apply_freeze(model: nn.Module, classifier_prefixes: Tuple[str, ...], strategy: Optional[FreezeStrategy]) -> None:
    """Freeze backbone parameters according to the provided strategy.

    Classifier parameters (whose names start with any of classifier_prefixes)
    are always kept trainable.
    """

    strategy: FreezeStrategy = _parse_freeze_strategy(freeze=strategy)

    # Ensure classifier is trainable
    for name, p in model.named_parameters():
        if name.startswith(classifier_prefixes):
            p.requires_grad = True

    if strategy == FreezeStrategy.NO:
        return

    backbone_params = [(n, p) for n, p in model.named_parameters() if not n.startswith(classifier_prefixes)]

    if strategy == FreezeStrategy.LAST:
        for _, p in backbone_params:
            p.requires_grad = False
        return

    if strategy == FreezeStrategy.PCT70:
        total = len(backbone_params)
        cutoff = int(total * 0.7)
        for idx, (_, p) in enumerate(backbone_params):
            p.requires_grad = False if idx < cutoff else True
        return
