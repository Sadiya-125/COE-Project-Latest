"""Compact CNN used by the local CPU experiments.

Design constraints:
* small enough for 30 federated rounds x 5 clients on a laptop CPU;
* GroupNorm instead of BatchNorm - BN's running statistics are computed on
  local data and leak information across the federation while also degrading
  FedAvg under non-IID splits (Hsieh et al., 2020). GroupNorm is the standard
  fix and is also required for correct per-example gradients in DP-SGD.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import CLASSES


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(min(groups, cout), cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(min(groups, cout), cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.relu(self.gn2(self.conv2(x)))
        return F.max_pool2d(x, 2)


class DermCNN(nn.Module):
    def __init__(self, num_classes: int = len(CLASSES), width: int = 32, in_ch: int = 3):
        super().__init__()
        self.block1 = ConvBlock(in_ch, width)
        self.block2 = ConvBlock(width, width * 2)
        self.block3 = ConvBlock(width * 2, width * 4)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block3(self.block2(self.block1(x)))
        x = F.adaptive_avg_pool2d(x, 1)
        return self.head(x)


def build_model(width: int = 32, num_classes: int = len(CLASSES), seed: int | None = None) -> DermCNN:
    if seed is not None:
        torch.manual_seed(seed)
    return DermCNN(num_classes=num_classes, width=width)


def state_to_numpy(model: nn.Module) -> dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}


def numpy_to_state(model: nn.Module, arrays: dict[str, np.ndarray]) -> None:
    sd = model.state_dict()
    model.load_state_dict(
        {k: torch.as_tensor(arrays[k], dtype=sd[k].dtype) for k in sd}, strict=True
    )


def get_flat_params(model: nn.Module) -> np.ndarray:
    return np.concatenate([v.detach().cpu().numpy().ravel() for v in model.state_dict().values()])


def set_flat_params(model: nn.Module, flat: np.ndarray) -> None:
    sd = model.state_dict()
    new, off = {}, 0
    for k, v in sd.items():
        n = v.numel()
        new[k] = torch.as_tensor(flat[off : off + n].reshape(v.shape), dtype=v.dtype)
        off += n
    model.load_state_dict(new, strict=True)


def param_spec(model: nn.Module) -> list[tuple[str, tuple[int, ...]]]:
    return [(k, tuple(v.shape)) for k, v in model.state_dict().items()]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
