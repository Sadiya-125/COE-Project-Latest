"""Metadata handling: lesion-grouped splits and non-IID client partitioning.

Everything here is framework-agnostic (pandas/numpy only) so the exact same
partition can be reproduced by the local PyTorch runs and by the TensorFlow
TPU job on Kaggle: same seed => same hospitals => comparable numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import CLASSES
from ..config import IMAGE_DIRS, METADATA_CSV


def _resolve_image_path(image_id: str, image_dirs=IMAGE_DIRS) -> str:
    for d in image_dirs:
        p = Path(d) / f"{image_id}.jpg"
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"{image_id}.jpg not found in {[str(d) for d in image_dirs]}")


def load_metadata(csv_path: Path = METADATA_CSV, resolve_paths: bool = True) -> pd.DataFrame:
    """Load HAM10000_metadata.csv, add integer labels and (optionally) file paths."""
    df = pd.read_csv(csv_path)
    df["label"] = df["dx"].map({c: i for i, c in enumerate(CLASSES)}).astype(int)
    # age is the only column with missing values (57 rows); median-impute so the
    # tabular metadata stays usable for stratification/reporting.
    df["age"] = df["age"].fillna(df["age"].median())
    df = df.sort_values("image_id").reset_index(drop=True)
    if resolve_paths:
        df["path"] = df["image_id"].map(_resolve_image_path)
    return df


def make_splits(
    df: pd.DataFrame,
    test_fraction: float = 0.15,
    val_fraction: float = 0.10,
    group_column: str = "lesion_id",
    seed: int = 1337,
) -> dict[str, np.ndarray]:
    """Split into train/val/test by *lesion group*, stratified by diagnosis.

    HAM10000 contains 7470 lesions across 10015 images. Splitting per-image
    leaks near-duplicate views of the same lesion across splits and inflates
    accuracy by several points, so we assign whole lesions.
    """
    rng = np.random.default_rng(seed)
    # one representative label per group (groups are single-diagnosis in HAM10000)
    groups = df.groupby(group_column)["label"].first()
    test_groups: list[str] = []
    val_groups: list[str] = []
    for label in sorted(groups.unique()):
        g = groups[groups == label].index.to_numpy()
        rng.shuffle(g)
        n_test = max(1, int(round(len(g) * test_fraction)))
        n_val = max(1, int(round(len(g) * val_fraction)))
        test_groups.extend(g[:n_test])
        val_groups.extend(g[n_test : n_test + n_val])
    test_set, val_set = set(test_groups), set(val_groups)
    in_test = df[group_column].isin(test_set).to_numpy()
    in_val = df[group_column].isin(val_set).to_numpy()
    return {
        "train": np.where(~in_test & ~in_val)[0],
        "val": np.where(in_val)[0],
        "test": np.where(in_test)[0],
    }


def dirichlet_partition(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    num_clients: int = 10,
    alpha: float = 0.5,
    group_column: str = "lesion_id",
    seed: int = 1337,
    min_samples: int = 32,
) -> list[np.ndarray]:
    """Partition the training pool into `num_clients` non-IID hospitals.

    Label proportions per hospital are drawn from Dir(alpha); a whole lesion
    always lands on one hospital, mirroring reality where all images of a
    patient's lesion live in one institution's PACS.
    """
    rng = np.random.default_rng(seed)
    sub = df.loc[train_idx]
    group_label = sub.groupby(group_column)["label"].first()

    for _ in range(100):  # retry until no hospital is degenerately small
        client_groups: list[list[str]] = [[] for _ in range(num_clients)]
        for label in range(len(CLASSES)):
            g = group_label[group_label == label].index.to_numpy()
            rng.shuffle(g)
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            cuts = (np.cumsum(proportions) * len(g)).astype(int)[:-1]
            for cid, chunk in enumerate(np.split(g, cuts)):
                client_groups[cid].extend(chunk)
        group_to_idx = sub.groupby(group_column).indices  # positional within `sub`
        pos = sub.index.to_numpy()
        parts = [
            np.sort(np.concatenate([pos[group_to_idx[g]] for g in gs])) if gs else np.array([], int)
            for gs in client_groups
        ]
        if min(len(p) for p in parts) >= min_samples:
            return parts
    raise RuntimeError(
        f"Could not build {num_clients} clients with >= {min_samples} samples "
        f"at alpha={alpha}; raise alpha or lower num_clients."
    )


def partition_report(df: pd.DataFrame, parts: list[np.ndarray]) -> dict:
    """Human/JSON friendly description of the federation (for the dashboard)."""
    clients = []
    for cid, idx in enumerate(parts):
        counts = df.loc[idx, "dx"].value_counts()
        clients.append(
            {
                "client_id": cid,
                "name": f"Hospital-{cid + 1:02d}",
                "num_samples": int(len(idx)),
                "num_lesions": int(df.loc[idx, "lesion_id"].nunique()),
                "label_counts": {c: int(counts.get(c, 0)) for c in CLASSES},
                "top_localizations": {
                    str(k): int(v)
                    for k, v in df.loc[idx, "localization"].value_counts().head(3).items()
                },
            }
        )
    sizes = np.array([c["num_samples"] for c in clients], dtype=float)
    dists = np.array(
        [[c["label_counts"][k] for k in CLASSES] for c in clients], dtype=float
    )
    dists = dists / np.clip(dists.sum(1, keepdims=True), 1, None)
    global_dist = dists.mean(0)
    # mean total-variation distance from the federation-average label mix:
    # a compact scalar summary of how non-IID the split is.
    tv = float(np.mean(0.5 * np.abs(dists - global_dist).sum(1)))
    return {
        "num_clients": len(clients),
        "clients": clients,
        "size_min": int(sizes.min()),
        "size_max": int(sizes.max()),
        "mean_total_variation": tv,
    }
