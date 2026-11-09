"""Scratch tuner: local DP-SGD clip/lr sweep on a single client (dev aid)."""

import torch.nn as nn

from fedderm.config import DataConfig, FedConfig, PrivacyConfig
from fedderm.torchimpl.data import build_federation
from fedderm.torchimpl.federated import LocalClient
from fedderm.torchimpl.metrics import evaluate
from fedderm.torchimpl.models import build_model

GRID = [(2.0, 0.01, 0.0), (2.0, 0.04, 0.0), (8.0, 0.005, 0.0), (2.0, 0.01, 0.3)]

if __name__ == "__main__":
    fed = build_federation(DataConfig())
    vl = fed.loader(fed.splits["val"], 256)
    w = fed.class_weights()
    for clip, lr, sigma in GRID:
        fc = FedConfig(local_batch_size=128, local_epochs=20, local_lr=lr)
        pv = PrivacyConfig(
            enabled=True, mode="example", l2_clip=clip, noise_multiplier=sigma,
            dp_local_momentum=0.0,
        )
        c = LocalClient(3, fed.parts[3], fed, fc, pv, w)
        m = build_model(width=16, seed=1337)
        loss = c._train_dpsgd(m, nn.CrossEntropyLoss(weight=w))
        bacc = evaluate(m, vl)["balanced_accuracy"]
        print(
            f"clip={clip} lr={lr} sigma={sigma} maxq={c.sampling_rate:.3f} "
            f"steps={c.local_steps} loss={loss:.3f} bacc={bacc:.4f}",
            flush=True,
        )
