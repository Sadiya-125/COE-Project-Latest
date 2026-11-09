"""Centralized baseline: the (privacy-violating) upper bound to compare against.

This is what you would get if all hospitals shipped their images to one server.
It is the reference point for the federated runs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..config import RESULTS, DataConfig, TrainConfig
from .data import Federation, build_federation
from .metrics import evaluate
from .models import build_model, count_params


def train_centralized(
    fed: Federation,
    cfg: TrainConfig,
    device: str = "cpu",
    log_every: int = 1,
    verbose: bool = True,
) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = build_model(width=cfg.width, seed=cfg.seed).to(device)
    train_loader = fed.loader(fed.splits["train"], cfg.batch_size, train=True)
    val_loader = fed.loader(fed.splits["val"], 256)
    test_loader = fed.loader(fed.splits["test"], 256)

    weights = fed.class_weights().to(device) if cfg.class_balanced_loss else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    opt = (
        torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        if cfg.optimizer == "adamw"
        else torch.optim.SGD(
            model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
        )
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    history, best = [], {"balanced_accuracy": -1.0}
    best_state = None
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += float(loss.detach()) * xb.size(0)
            seen += xb.size(0)
        sched.step()
        val = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": running / max(seen, 1),
            "val_accuracy": val["accuracy"],
            "val_balanced_accuracy": val["balanced_accuracy"],
            "val_macro_f1": val["macro_f1"],
            "val_malignant_sensitivity": val["malignant_sensitivity"],
            "elapsed_s": time.time() - t0,
        }
        history.append(row)
        if val["balanced_accuracy"] > best["balanced_accuracy"]:
            best = val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and epoch % log_every == 0:
            print(
                f"[central] epoch {epoch:>3}/{cfg.epochs} loss={row['train_loss']:.4f} "
                f"val_bacc={val['balanced_accuracy']:.4f} val_f1={val['macro_f1']:.4f}"
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate(model, test_loader, device)
    if verbose:
        print(f"[central] TEST bacc={test['balanced_accuracy']:.4f} f1={test['macro_f1']:.4f}")
    return {
        "mode": "centralized",
        "config": {"train": cfg.__dict__},
        "num_params": count_params(model),
        "history": history,
        "val": best,
        "test": test,
        "wall_time_s": time.time() - t0,
        "privacy": {"epsilon": None, "note": "no formal guarantee: raw images pooled centrally"},
        "model": model,
    }


def run(out: Path | None = None, epochs: int | None = None) -> dict:
    data_cfg, train_cfg = DataConfig(), TrainConfig()
    if epochs:
        train_cfg.epochs = epochs
    fed = build_federation(data_cfg)
    res = train_centralized(fed, train_cfg)
    model = res.pop("model")
    out = out or RESULTS / "centralized.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), out.with_suffix(".pt"))
    print(f"[central] saved {out}")
    return res


if __name__ == "__main__":
    run()
