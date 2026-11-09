"""In-memory tensor datasets + the federation builder for the local CPU path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import DataConfig
from ..data.metadata import dirichlet_partition, load_metadata, make_splits, partition_report
from ..data.preprocess import build_cache, load_cache

# Channel statistics of HAM10000 (computed once over the training split).
MEAN = np.array([0.7635, 0.5461, 0.5705], dtype=np.float32)
STD = np.array([0.1409, 0.1526, 0.1697], dtype=np.float32)


class LesionDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, train: bool = False):
        # NHWC uint8 -> NCHW float normalized
        x = images.astype(np.float32) / 255.0
        x = (x - MEAN) / STD
        self.x = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))
        self.y = torch.from_numpy(labels.astype(np.int64))
        self.train = train

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int):
        img = self.x[i]
        if self.train:  # cheap, label-preserving dermatoscopy augmentations
            if torch.rand(()) < 0.5:
                img = torch.flip(img, dims=[2])
            if torch.rand(()) < 0.5:
                img = torch.flip(img, dims=[1])
            k = int(torch.randint(0, 4, ()))
            if k:
                img = torch.rot90(img, k, dims=[1, 2])
        return img, self.y[i]


@dataclass
class Federation:
    """Everything the trainers need: global splits + per-client index sets."""

    df: object
    images: np.ndarray
    labels: np.ndarray
    splits: dict[str, np.ndarray]
    parts: list[np.ndarray]
    report: dict

    def dataset(self, idx: np.ndarray, train: bool = False) -> LesionDataset:
        return LesionDataset(self.images[idx], self.labels[idx], train=train)

    def loader(self, idx: np.ndarray, batch_size: int, train: bool = False) -> DataLoader:
        return DataLoader(
            self.dataset(idx, train),
            batch_size=batch_size,
            shuffle=train,
            drop_last=False,
            num_workers=0,
        )

    def client_loader(self, cid: int, batch_size: int) -> DataLoader:
        return self.loader(self.parts[cid], batch_size, train=True)

    def class_weights(self) -> torch.Tensor:
        y = self.labels[self.splits["train"]]
        counts = np.bincount(y, minlength=7).astype(np.float64)
        w = counts.sum() / (len(counts) * np.clip(counts, 1, None))
        return torch.tensor(w, dtype=torch.float32)


def build_federation(cfg: DataConfig, ensure_cache: bool = True) -> Federation:
    if ensure_cache:
        build_cache(cfg.image_size)
    images, labels, image_ids = load_cache(cfg.image_size)
    df = load_metadata(resolve_paths=False)
    assert list(df["image_id"]) == list(image_ids), "cache/metadata row order mismatch"
    splits = make_splits(df, cfg.test_fraction, cfg.val_fraction, cfg.group_column, cfg.seed)
    parts = dirichlet_partition(
        df, splits["train"], cfg.num_clients, cfg.dirichlet_alpha, cfg.group_column, cfg.seed
    )
    return Federation(df, images, labels, splits, parts, partition_report(df, parts))
