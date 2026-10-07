"""Seed all random sources."""
from __future__ import annotations

import random


def set_seed(seed: int) -> None:
    # numpy and torch are imported here, not at module level, so that importing
    # src.utils (and with it the benchmark loaders and the unit tests) does not
    # require them.
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
