"""Shared device selection: MPS > CUDA > CPU."""

import torch

_DEVICE = None


def get_device() -> torch.device:
    global _DEVICE
    if _DEVICE is None:
        if torch.backends.mps.is_available():
            _DEVICE = torch.device("mps")
        elif torch.cuda.is_available():
            _DEVICE = torch.device("cuda")
        else:
            _DEVICE = torch.device("cpu")
    return _DEVICE
