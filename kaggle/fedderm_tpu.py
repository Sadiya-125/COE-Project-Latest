"""FedDerm on Kaggle: federated + centralized HAM10000 at 224x224.

Self-contained TensorFlow/Keras implementation (identical splits, identical
Dirichlet partition seed, identical RDP accountant to the local PyTorch
package), so the numbers are directly comparable to the CPU numbers and to
each other. The default backbone is a pretrained EfficientNet-B0; the original
GroupNorm CNN is kept as architecture="dermcnn".

Why TensorFlow here: Kaggle's TPU VM ships TF with a working `TPUStrategy`, and
federated averaging maps cleanly onto it - the 8 TPU cores execute each
client's local epochs as one replicated data-parallel job, and only the
aggregation step touches numpy.

Usage inside a Kaggle notebook (see kaggle/README.md for the full recipe):

    !cp -r /kaggle/input/<your-utility-dataset>/fedderm_tpu.py .
    import fedderm_tpu as F
    F.main(mode="all")

Or as a script:
    !python fedderm_tpu.py --mode all --rounds 20
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

# --------------------------------------------------------------------------- #
# Paths (Kaggle layout, with a local fallback)
# --------------------------------------------------------------------------- #
KAGGLE_INPUT = Path("/kaggle/input")
LOCAL_INPUT = Path(__file__).resolve().parent.parent / "archive"


def discover_data_dir(input_root: Path = KAGGLE_INPUT, local_input: Path = LOCAL_INPUT) -> Path:
    legacy = input_root / "skin-cancer-mnist-ham10000"
    if (legacy / "HAM10000_metadata.csv").exists():
        return legacy
    if input_root.exists():
        matches = sorted(input_root.rglob("HAM10000_metadata.csv"))
        if matches:
            return matches[0].parent
    return local_input


DATA_DIR = discover_data_dir()
OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("results")

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MALIGNANT_IDX = [0, 1, 4]  # akiec, bcc, mel
AUTOTUNE = tf.data.AUTOTUNE


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class Cfg:
    image_size: int = 224
    architecture: str = "efficientnetb0"  # "efficientnetb0" | "dermcnn"
    pretrained: bool = True
    fine_tune_layers: int = 20
    federated_fine_tune_layers: int = 0
    num_clients: int = 10
    dirichlet_alpha: float = 0.5
    test_fraction: float = 0.15
    val_fraction: float = 0.10
    seed: int = 1337
    # centralized
    epochs: int = 12
    batch_size: int = 32
    lr: float = 2e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    class_weight_power: float = 1.0
    checkpoint_metric: str = "macro_f1"
    test_time_augmentation: bool = True
    # federated
    rounds: int = 20
    clients_per_round: int = 5
    local_epochs: int = 1
    local_batch_size: int = 32
    local_lr: float = 2e-4
    server_lr: float = 1.0
    federated_algorithm: str = "fedprox"  # "fedavg" | "fedprox"
    fedprox_mu: float = 0.01
    balanced_federated_sampling: bool = False
    # privacy
    dp_mode: str = "client"        # "client" (DP-FedAvg) | "example" (DP-SGD) | "none"
    l2_clip: float = 1.0
    noise_multiplier: float = 0.8
    delta: float = 1e-5
    secure_aggregation: bool = True
    microbatches: int = 32         # for dp_mode="example" on TPU


TUNING_PLAN = (
    (
        "federated",
        "fedavg_balanced_e4",
        {
            "dp_mode": "none",
            "rounds": 30,
            "local_epochs": 4,
            "local_lr": 1e-3,
            "federated_algorithm": "fedavg",
            "fedprox_mu": 0.0,
            "balanced_federated_sampling": True,
        },
    ),
    (
        "federated",
        "fedprox_balanced_e4",
        {
            "dp_mode": "none",
            "rounds": 30,
            "local_epochs": 4,
            "local_lr": 1e-3,
            "federated_algorithm": "fedprox",
            "fedprox_mu": 1e-3,
            "balanced_federated_sampling": True,
        },
    ),
    (
        "centralized",
        "centralized_ft40",
        {"fine_tune_layers": 40, "epochs": 18, "lr": 1e-4},
    ),
    (
        "centralized",
        "centralized_ft80",
        {"fine_tune_layers": 80, "epochs": 18, "lr": 5e-5},
    ),
)

TUNING_PLAN_V2 = (
    (
        "federated",
        "fedavg_full10_balanced_e4",
        {
            "dp_mode": "none",
            "rounds": 25,
            "clients_per_round": 10,
            "local_epochs": 4,
            "local_lr": 5e-4,
            "federated_algorithm": "fedavg",
            "fedprox_mu": 0.0,
            "balanced_federated_sampling": True,
        },
    ),
    (
        "centralized",
        "centralized_b0_ft120_w075",
        {
            "architecture": "efficientnetb0",
            "image_size": 224,
            "fine_tune_layers": 120,
            "epochs": 20,
            "lr": 5e-5,
            "class_weight_power": 0.75,
        },
    ),
    (
        "centralized",
        "centralized_b1_ft100_w075",
        {
            "architecture": "efficientnetb1",
            "image_size": 240,
            "fine_tune_layers": 100,
            "epochs": 20,
            "lr": 5e-5,
            "class_weight_power": 0.75,
        },
    ),
    (
        "centralized",
        "centralized_b2_ft120_w075",
        {
            "architecture": "efficientnetb2",
            "image_size": 260,
            "fine_tune_layers": 120,
            "epochs": 20,
            "lr": 4e-5,
            "class_weight_power": 0.75,
        },
    ),
)

TUNING_PLAN_V3 = (
    (
        "federated",
        "fedavg_ft20_balanced_e2",
        {
            "dp_mode": "none",
            "rounds": 30,
            "clients_per_round": 5,
            "local_epochs": 2,
            "local_lr": 1e-4,
            "federated_algorithm": "fedavg",
            "fedprox_mu": 0.0,
            "balanced_federated_sampling": True,
            "federated_fine_tune_layers": 20,
        },
    ),
    (
        "centralized",
        "centralized_b0_ft80_w085",
        {
            "architecture": "efficientnetb0",
            "image_size": 224,
            "fine_tune_layers": 80,
            "epochs": 20,
            "lr": 5e-5,
            "class_weight_power": 0.85,
        },
    ),
    (
        "centralized",
        "centralized_b0_ft120_w085",
        {
            "architecture": "efficientnetb0",
            "image_size": 224,
            "fine_tune_layers": 120,
            "epochs": 20,
            "lr": 5e-5,
            "class_weight_power": 0.85,
        },
    ),
    (
        "centralized",
        "centralized_b0_ft120_w090",
        {
            "architecture": "efficientnetb0",
            "image_size": 224,
            "fine_tune_layers": 120,
            "epochs": 20,
            "lr": 5e-5,
            "class_weight_power": 0.90,
        },
    ),
    (
        "centralized",
        "centralized_b0_ft120_w085_s000",
        {
            "architecture": "efficientnetb0",
            "image_size": 224,
            "fine_tune_layers": 120,
            "epochs": 20,
            "lr": 5e-5,
            "class_weight_power": 0.85,
            "label_smoothing": 0.0,
        },
    ),
    (
        "centralized",
        "centralized_b0_256_ft120_w085",
        {
            "architecture": "efficientnetb0",
            "image_size": 256,
            "fine_tune_layers": 120,
            "epochs": 20,
            "lr": 4e-5,
            "class_weight_power": 0.85,
        },
    ),
)

TUNING_PLAN_V4 = (
    (
        "centralized",
        "centralized_b0_256_ft120_w090",
        {
            "architecture": "efficientnetb0",
            "image_size": 256,
            "fine_tune_layers": 120,
            "epochs": 20,
            "lr": 4e-5,
            "class_weight_power": 0.90,
        },
    ),
    (
        "centralized",
        "centralized_b0_256_ft160_w085",
        {
            "architecture": "efficientnetb0",
            "image_size": 256,
            "fine_tune_layers": 160,
            "epochs": 20,
            "lr": 3e-5,
            "class_weight_power": 0.85,
        },
    ),
    (
        "centralized",
        "centralized_b0_288_ft120_w085",
        {
            "architecture": "efficientnetb0",
            "image_size": 288,
            "fine_tune_layers": 120,
            "epochs": 20,
            "lr": 4e-5,
            "class_weight_power": 0.85,
        },
    ),
    (
        "centralized",
        "centralized_v2b0_ft100_w085",
        {
            "architecture": "efficientnetv2b0",
            "image_size": 224,
            "fine_tune_layers": 100,
            "epochs": 20,
            "lr": 5e-5,
            "class_weight_power": 0.85,
        },
    ),
    (
        "centralized",
        "centralized_v2b0_ft160_w085",
        {
            "architecture": "efficientnetv2b0",
            "image_size": 224,
            "fine_tune_layers": 160,
            "epochs": 20,
            "lr": 3e-5,
            "class_weight_power": 0.85,
        },
    ),
    (
        "centralized",
        "centralized_v2b1_ft120_w085",
        {
            "architecture": "efficientnetv2b1",
            "image_size": 240,
            "fine_tune_layers": 120,
            "epochs": 20,
            "lr": 4e-5,
            "class_weight_power": 0.85,
        },
    ),
)

TUNING_PLAN_V5 = (
    (
        "centralized",
        "centralized_b0_288_ft120_w0875",
        {
            "architecture": "efficientnetb0",
            "image_size": 288,
            "fine_tune_layers": 120,
            "epochs": 20,
            "lr": 4e-5,
            "class_weight_power": 0.875,
        },
    ),
    (
        "centralized",
        "centralized_b0_288_ft120_w090",
        {
            "architecture": "efficientnetb0",
            "image_size": 288,
            "fine_tune_layers": 120,
            "epochs": 20,
            "lr": 4e-5,
            "class_weight_power": 0.90,
        },
    ),
    (
        "centralized",
        "centralized_b0_288_ft160_w085",
        {
            "architecture": "efficientnetb0",
            "image_size": 288,
            "fine_tune_layers": 160,
            "epochs": 20,
            "lr": 3e-5,
            "class_weight_power": 0.85,
        },
    ),
    (
        "centralized",
        "centralized_b0_320_ft120_w085",
        {
            "architecture": "efficientnetb0",
            "image_size": 320,
            "fine_tune_layers": 120,
            "epochs": 20,
            "lr": 4e-5,
            "class_weight_power": 0.85,
        },
    ),
    (
        "centralized",
        "centralized_b0_288_ft120_w085_b16",
        {
            "architecture": "efficientnetb0",
            "image_size": 288,
            "fine_tune_layers": 120,
            "epochs": 20,
            "batch_size": 16,
            "lr": 4e-5,
            "class_weight_power": 0.85,
        },
    ),
    (
        "centralized",
        "centralized_b0_288_ft120_w085_lr5",
        {
            "architecture": "efficientnetb0",
            "image_size": 288,
            "fine_tune_layers": 120,
            "epochs": 20,
            "lr": 5e-5,
            "class_weight_power": 0.85,
        },
    ),
)

TUNING_PLAN_V6 = (
    (
        "centralized",
        "centralized_b0_288_ft120_w0875_b16",
        {
            "architecture": "efficientnetb0",
            "image_size": 288,
            "fine_tune_layers": 120,
            "epochs": 20,
            "batch_size": 16,
            "lr": 4e-5,
            "class_weight_power": 0.875,
        },
    ),
    (
        "centralized",
        "centralized_b0_320_ft120_w0875_b16",
        {
            "architecture": "efficientnetb0",
            "image_size": 320,
            "fine_tune_layers": 120,
            "epochs": 20,
            "batch_size": 16,
            "lr": 4e-5,
            "class_weight_power": 0.875,
        },
    ),
    (
        "federated",
        "fedavg_ft20_balanced_e3",
        {
            "dp_mode": "none",
            "rounds": 35,
            "clients_per_round": 5,
            "local_epochs": 3,
            "local_batch_size": 32,
            "local_lr": 7.5e-5,
            "server_lr": 1.0,
            "federated_algorithm": "fedavg",
            "fedprox_mu": 0.0,
            "balanced_federated_sampling": True,
            "federated_fine_tune_layers": 20,
        },
    ),
    (
        "federated",
        "fedavg_ft40_balanced_e2",
        {
            "dp_mode": "none",
            "rounds": 35,
            "clients_per_round": 5,
            "local_epochs": 2,
            "local_batch_size": 32,
            "local_lr": 5e-5,
            "server_lr": 1.0,
            "federated_algorithm": "fedavg",
            "fedprox_mu": 0.0,
            "balanced_federated_sampling": True,
            "federated_fine_tune_layers": 40,
        },
    ),
)


# --------------------------------------------------------------------------- #
# Privacy: RDP accountant (mirrors fedderm/privacy/accountant.py)
# --------------------------------------------------------------------------- #
ORDERS = tuple([1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64)) + [128, 256, 512])


def _log_add(a, b):
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    return max(a, b) + math.log1p(math.exp(-abs(a - b)))


def _log_sub(a, b):
    if b == -np.inf:
        return a
    if a == b:
        return -np.inf
    return a + math.log1p(-math.exp(b - a))


def _log_comb(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _rdp_int(q, sigma, alpha):
    if q == 0:
        return 0.0
    if q == 1:
        return alpha / (2 * sigma**2)
    log_a = -np.inf
    for k in range(alpha + 1):
        log_a = _log_add(
            log_a,
            _log_comb(alpha, k)
            + (alpha - k) * math.log1p(-q)
            + k * math.log(q)
            + (k * k - k) / (2 * sigma**2),
        )
    return log_a / (alpha - 1)


def _rdp_frac(q, sigma, alpha):
    from scipy.special import binom, log_ndtr

    log_a0 = log_a1 = -np.inf
    z0 = sigma**2 * math.log(1 / q - 1) + 0.5
    i = 0
    while True:
        coef = binom(alpha, i)
        lc, j = math.log(abs(coef)), alpha - i
        le0 = math.log(0.5) + math.log(2) + log_ndtr(-((i - z0) / (math.sqrt(2) * sigma)) * 2**0.5)
        le1 = math.log(0.5) + math.log(2) + log_ndtr(-((z0 - j) / (math.sqrt(2) * sigma)) * 2**0.5)
        ls0 = lc + i * math.log(q) + j * math.log1p(-q) + (i * i - i) / (2 * sigma**2) + le0
        ls1 = lc + j * math.log(q) + i * math.log1p(-q) + (j * j - j) / (2 * sigma**2) + le1
        if coef > 0:
            log_a0, log_a1 = _log_add(log_a0, ls0), _log_add(log_a1, ls1)
        else:
            log_a0, log_a1 = _log_sub(log_a0, ls0), _log_sub(log_a1, ls1)
        i += 1
        if max(ls0, ls1) < -30 or i > 1000:
            break
    return _log_add(log_a0, log_a1) / (alpha - 1)


def rdp_curve(q, sigma):
    out = []
    for a in ORDERS:
        if q == 0:
            out.append(0.0)
        elif float(a).is_integer():
            out.append(_rdp_int(q, sigma, int(a)))
        else:
            out.append(_rdp_frac(q, sigma, float(a)))
    return np.asarray(out)


def rdp_to_dp(rdp, delta):
    o = np.asarray(ORDERS, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        eps = rdp + np.log1p(-1 / o) - (np.log(delta) + np.log(o)) / (o - 1)
    eps = np.where(np.isfinite(eps), eps, np.inf)
    i = int(np.argmin(eps))
    return float(max(eps[i], 0.0)), float(o[i])


class Accountant:
    def __init__(self, delta=1e-5):
        self.delta, self._rdp, self.n = delta, np.zeros(len(ORDERS)), 0

    def step(self, q, sigma, steps=1):
        if sigma <= 0:
            self._rdp = np.full_like(self._rdp, np.inf)
        else:
            self._rdp = self._rdp + rdp_curve(q, sigma) * steps
        self.n += 1

    @property
    def epsilon(self):
        return rdp_to_dp(self._rdp, self.delta)[0]

    def summary(self):
        e, a = rdp_to_dp(self._rdp, self.delta)
        return {"epsilon": e, "delta": self.delta, "optimal_order": a, "num_compositions": self.n}


# --------------------------------------------------------------------------- #
# Privacy: secure aggregation (numpy; mirrors fedderm/privacy/secure_agg.py)
# --------------------------------------------------------------------------- #
class SecAgg:
    """Pairwise-mask secure aggregation over Z_M with fixed-point encoding.

    On TPU the model updates come back as numpy arrays, so masking happens on
    the host - exactly where a real deployment would do it (inside the client's
    trusted boundary, before anything is sent over the wire).
    """

    def __init__(self, bits=20, modulus_bits=62, clip_range=8.0, seed=0):
        self.scale = float(1 << bits)
        self.M = np.uint64(1) << np.uint64(modulus_bits)
        self.clip_range = clip_range
        self.master = np.random.default_rng(seed)
        self._pair_seeds: dict[tuple[int, int], int] = {}

    def pair_seed(self, i, j, rnd):
        key = (min(i, j), max(i, j))
        if key not in self._pair_seeds:
            # stands in for the X25519 ECDH handshake of the local implementation
            self._pair_seeds[key] = int(self.master.integers(0, 2**62))
        return (self._pair_seeds[key] * 1_000_003 + rnd) % (2**63 - 1)

    def encode(self, x):
        m = np.int64(self.M)
        return (np.rint(np.clip(x, -self.clip_range, self.clip_range) * self.scale).astype(np.int64) % m).astype(np.uint64)

    def decode(self, y):
        m = np.int64(self.M)
        v = y.astype(np.int64) % m
        return np.where(v > m // 2, v - m, v).astype(np.float64) / self.scale

    def mask(self, i, peers, rnd, size):
        out = np.zeros(size, dtype=np.uint64)
        for j in peers:
            if j == i:
                continue
            m = np.random.default_rng(self.pair_seed(i, j, rnd)).integers(
                0, int(self.M), size=size, dtype=np.uint64
            )
            out = (out + m) % self.M if j > i else (out + (self.M - m)) % self.M
        return out

    def mask_update(self, i, peers, rnd, vec):
        return (self.encode(vec) + self.mask(i, peers, rnd, vec.size)) % self.M

    def aggregate(self, payloads):
        total = np.zeros_like(next(iter(payloads.values())))
        for y in payloads.values():
            total = (total + y) % self.M
        return self.decode(total)


# --------------------------------------------------------------------------- #
# Data: splits + Dirichlet partition (identical logic/seed to the local package)
# --------------------------------------------------------------------------- #
def load_metadata(cfg: Cfg) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "HAM10000_metadata.csv")
    df["label"] = df["dx"].map({c: i for i, c in enumerate(CLASSES)}).astype(int)
    df["age"] = df["age"].fillna(df["age"].median())
    df = df.sort_values("image_id").reset_index(drop=True)
    part1 = DATA_DIR / "HAM10000_images_part_1"
    part2 = DATA_DIR / "HAM10000_images_part_2"
    have1 = {p.stem for p in part1.glob("*.jpg")} if part1.exists() else set()
    df["path"] = [
        str((part1 if i in have1 else part2) / f"{i}.jpg") for i in df["image_id"]
    ]
    return df


def make_splits(df, cfg: Cfg):
    rng = np.random.default_rng(cfg.seed)
    groups = df.groupby("lesion_id")["label"].first()
    test_g, val_g = [], []
    for label in sorted(groups.unique()):
        g = groups[groups == label].index.to_numpy()
        rng.shuffle(g)
        n_test = max(1, int(round(len(g) * cfg.test_fraction)))
        n_val = max(1, int(round(len(g) * cfg.val_fraction)))
        test_g.extend(g[:n_test])
        val_g.extend(g[n_test : n_test + n_val])
    in_test = df["lesion_id"].isin(set(test_g)).to_numpy()
    in_val = df["lesion_id"].isin(set(val_g)).to_numpy()
    return {
        "train": np.where(~in_test & ~in_val)[0],
        "val": np.where(in_val)[0],
        "test": np.where(in_test)[0],
    }


def dirichlet_partition(df, train_idx, cfg: Cfg, min_samples=32):
    rng = np.random.default_rng(cfg.seed)
    sub = df.loc[train_idx]
    group_label = sub.groupby("lesion_id")["label"].first()
    for _ in range(100):
        buckets = [[] for _ in range(cfg.num_clients)]
        for label in range(len(CLASSES)):
            g = group_label[group_label == label].index.to_numpy()
            rng.shuffle(g)
            p = rng.dirichlet(np.repeat(cfg.dirichlet_alpha, cfg.num_clients))
            cuts = (np.cumsum(p) * len(g)).astype(int)[:-1]
            for cid, chunk in enumerate(np.split(g, cuts)):
                buckets[cid].extend(chunk)
        gidx, pos = sub.groupby("lesion_id").indices, sub.index.to_numpy()
        parts = [
            np.sort(np.concatenate([pos[gidx[g]] for g in gs])) if gs else np.array([], int)
            for gs in buckets
        ]
        if min(len(p) for p in parts) >= min_samples:
            return parts
    raise RuntimeError("partitioning failed; raise alpha")


# --------------------------------------------------------------------------- #
# tf.data pipelines
# --------------------------------------------------------------------------- #
MEAN = tf.constant([0.7635, 0.5461, 0.5705])
STD = tf.constant([0.1409, 0.1526, 0.1697])


def _decode(path, label, size):
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.image.resize_with_crop_or_pad(img, 450, 450)   # centre square
    img = tf.image.resize(img, (size, size), method="bilinear")
    return tf.cast(img, tf.float32) / 255.0, label


def _augment(img, label):
    img = tf.image.random_flip_left_right(tf.image.random_flip_up_down(img))
    img = tf.image.rot90(img, tf.random.uniform([], 0, 4, tf.int32))
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    return tf.clip_by_value(img, 0.0, 1.0), label


def _normalize(img, label):
    return (img - MEAN) / STD, label


def _class_balanced(df, idx, size, architecture):
    """Uniform-over-classes stream.

    Needed for example-level DP: per-example clipping pins every gradient at
    `l2_clip`, which cancels class weighting entirely and collapses the model
    onto `nv` (67% of HAM10000). Under DP the only way to rebalance is to change
    how often each class is *sampled*, so we interleave one dataset per class
    with equal probability.
    """
    d = df.loc[idx]
    per_class = []
    for label in sorted(d["label"].unique()):
        sub = d[d["label"] == label]
        c = tf.data.Dataset.from_tensor_slices((sub["path"].to_numpy(), sub["label"].to_numpy()))
        c = c.shuffle(max(len(sub), 2), seed=1337, reshuffle_each_iteration=True).repeat()
        c = c.map(lambda p, y: _decode(p, y, size), num_parallel_calls=AUTOTUNE)
        per_class.append(c)
    return tf.data.Dataset.sample_from_datasets(
        per_class, weights=[1.0 / len(per_class)] * len(per_class), seed=1337
    )


def make_ds(df, idx, size, batch, train, architecture="efficientnetb0",
            cache_path=None, drop_remainder=False, balanced=False):
    # EfficientNet does its own rescaling inside the model, so the pipeline
    # leaves images in [0,1]; the HAM10000 mean/std normalization is only for
    # the GroupNorm dermcnn.
    normalize = architecture == "dermcnn"
    if balanced:
        # already repeating + shuffled; caching a repeating stream would pin it
        ds = _class_balanced(df, idx, size, architecture)
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)
        if normalize:
            ds = ds.map(_normalize, num_parallel_calls=AUTOTUNE)
        return ds.batch(batch, drop_remainder=True).prefetch(1)
    d = df.loc[idx]
    ds = tf.data.Dataset.from_tensor_slices((d["path"].to_numpy(), d["label"].to_numpy()))
    ds = ds.map(lambda p, y: _decode(p, y, size), num_parallel_calls=AUTOTUNE)
    # cache the decoded tensors: 10k 224x224 images ~= 1.5 GB, still trivial
    # for the host RAM, and it removes JPEG decoding from the critical path.
    ds = ds.cache(cache_path) if cache_path else ds.cache()
    if train:
        ds = ds.shuffle(min(len(idx), 4096), seed=1337, reshuffle_each_iteration=True)
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)
    if normalize:
        ds = ds.map(_normalize, num_parallel_calls=AUTOTUNE)
    return ds.batch(batch, drop_remainder=drop_remainder).prefetch(AUTOTUNE)


# --------------------------------------------------------------------------- #
# Model: same topology as the PyTorch DermCNN, plus an optional wider variant
# --------------------------------------------------------------------------- #
def conv_block(x, filters, groups=8):
    for _ in range(2):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.GroupNormalization(groups=min(groups, filters))(x)
        x = tf.keras.layers.ReLU()(x)
    return tf.keras.layers.MaxPooling2D()(x)


def build_model(
    image_size=224,
    architecture="efficientnetb0",
    pretrained=True,
    fine_tune_layers=0,
    num_classes=7,
):
    """Pretrained EfficientNet-B0/B1/B2 or the from-scratch GroupNorm CNN.

    GroupNorm is used in `dermcnn` because BatchNorm's running statistics are
    computed on local client data and both leak information and destabilise
    FedAvg under non-IID partitions. For the pretrained backbone the same
    concern is handled by keeping every BatchNormalization layer frozen and
    calling the backbone with `training=False`.
    """
    inp = tf.keras.Input((image_size, image_size, 3))
    if architecture in ("efficientnetb0", "efficientnetb1", "efficientnetb2",
                        "efficientnetv2b0", "efficientnetv2b1"):
        builders = {
            "efficientnetb0": tf.keras.applications.EfficientNetB0,
            "efficientnetb1": tf.keras.applications.EfficientNetB1,
            "efficientnetb2": tf.keras.applications.EfficientNetB2,
            "efficientnetv2b0": tf.keras.applications.EfficientNetV2B0,
            "efficientnetv2b1": tf.keras.applications.EfficientNetV2B1,
        }
        backbone = builders[architecture](
            include_top=False,
            weights="imagenet" if pretrained else None,
            input_shape=(image_size, image_size, 3),
        )
        if fine_tune_layers == 0:
            backbone.trainable = False
        else:
            for layer in backbone.layers[:-fine_tune_layers]:
                layer.trainable = False
            for layer in backbone.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False
        x = tf.keras.layers.Rescaling(255.0)(inp)  # dataset emits [0,1]
        x = backbone(x, training=False)
    elif architecture == "dermcnn":
        x = inp
        for mult in (1, 2, 4, 8):
            x = conv_block(x, 32 * mult)
    else:
        raise ValueError(f"unknown architecture: {architecture!r}")
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(num_classes)(x)
    return tf.keras.Model(inp, out, name=architecture)


def get_tpu_strategy():
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        print(f"[tpu] connected: {strategy.num_replicas_in_sync} cores")
        return strategy
    except Exception as exc:  # GPU/CPU fallback keeps the script runnable anywhere
        print(f"[tpu] unavailable ({type(exc).__name__}: {exc}); falling back")
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print(f"[gpu] using one device for sound DP accounting: {gpus[0].name}")
            return tf.distribute.OneDeviceStrategy("/GPU:0")
        return tf.distribute.get_strategy()


def class_weights(df, train_idx, power=1.0):
    y = df.loc[train_idx, "label"].to_numpy()
    counts = np.bincount(y, minlength=len(CLASSES)).astype(np.float64)
    w = (counts.sum() / (len(CLASSES) * np.clip(counts, 1, None))) ** power
    return tf.constant(w, dtype=tf.float32)


def weighted_loss_fn(weights, num_replicas, label_smoothing=0.0):
    ce = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )

    def loss(y, logits):
        y = tf.cast(y, tf.int32)
        y_onehot = tf.one_hot(y, len(CLASSES))
        if label_smoothing:
            y_onehot = y_onehot * (1.0 - label_smoothing) + label_smoothing / len(CLASSES)
        per_example = ce(y_onehot, logits) * tf.gather(weights, y)
        return tf.nn.compute_average_loss(per_example)

    return loss


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def evaluate(model, ds, tta=False) -> dict:
    logits_all, y_all = [], []
    for xb, yb in ds:
        y_all.append(yb.numpy())
        logits = model(xb, training=False)
        if tta:
            # fixed 4 views: original, hflip, vflip, rot90 — labels never used
            logits = (
                logits
                + model(tf.image.flip_left_right(xb), training=False)
                + model(tf.image.flip_up_down(xb), training=False)
                + model(tf.image.rot90(xb, 1), training=False)
            ) / 4.0
        logits_all.append(logits.numpy())
    probs = tf.nn.softmax(np.concatenate(logits_all)).numpy()
    y = np.concatenate(y_all)[: len(probs)]
    pred = probs.argmax(1)
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    recall = np.diag(cm) / np.clip(cm.sum(1), 1, None)
    precision = np.diag(cm) / np.clip(cm.sum(0), 1, None)
    f1 = []
    for k in range(len(CLASSES)):
        tp, fp, fn = cm[k, k], cm[:, k].sum() - cm[k, k], cm[k].sum() - cm[k, k]
        f1.append(2 * tp / max(2 * tp + fp + fn, 1))
    mal_t, mal_p = np.isin(y, MALIGNANT_IDX), np.isin(pred, MALIGNANT_IDX)
    return {
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(recall.mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(np.mean(f1)),
        "malignant_sensitivity": float(mal_p[mal_t].mean()) if mal_t.any() else float("nan"),
        "malignant_specificity": float((~mal_p[~mal_t]).mean()),
        "per_class_recall": {c: float(recall[i]) for i, c in enumerate(CLASSES)},
        "confusion_matrix": cm.tolist(),
        "num_samples": int(len(y)),
    }


# --------------------------------------------------------------------------- #
# Centralized training (all data pooled - the privacy-violating upper bound)
# --------------------------------------------------------------------------- #
class _MetricCheckpoint(tf.keras.callbacks.Callback):
    """Evaluate on the val split after each epoch, keep the best weights by
    `cfg.checkpoint_metric`, restore them at the end and early-stop after
    `patience` non-improving epochs."""

    def __init__(self, val_ds, metric, t0, patience=4):
        super().__init__()
        self.val_ds, self.metric, self.t0 = val_ds, metric, t0
        self.patience = patience
        self.best, self.best_weights, self.rows, self._wait = -1.0, None, [], 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val = evaluate(self.model, self.val_ds)
        row = {
            "epoch": epoch + 1,
            "train_loss": float(logs.get("loss", float("nan"))),
            "val_accuracy": val["accuracy"],
            "val_balanced_accuracy": val["balanced_accuracy"],
            "val_macro_f1": val["macro_f1"],
            "val_malignant_sensitivity": val["malignant_sensitivity"],
            "elapsed_s": time.time() - self.t0,
        }
        self.rows.append(row)
        score = val[self.metric]
        if score > self.best:
            self.best, self._wait = score, 0
            self.best_weights = self.model.get_weights()
        else:
            self._wait += 1
            if self._wait >= self.patience:
                self.model.stop_training = True
        print(
            f"[centralized] epoch {epoch + 1} loss={row['train_loss']:.4f} "
            f"val_bacc={val['balanced_accuracy']:.4f} f1={val['macro_f1']:.4f}"
        )

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def train_centralized(cfg: Cfg, df, splits, strategy, label="centralized") -> dict:
    t0 = time.time()
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(cfg.seed)
    with strategy.scope():
        model = build_model(
            image_size=cfg.image_size,
            architecture=cfg.architecture,
            pretrained=cfg.pretrained,
            fine_tune_layers=cfg.fine_tune_layers,
        )
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=cfg.lr, weight_decay=cfg.weight_decay
            ),
            loss=lambda y, logits: tf.keras.losses.categorical_crossentropy(
                tf.one_hot(tf.cast(y, tf.int32), len(CLASSES)),
                logits,
                from_logits=True,
                label_smoothing=cfg.label_smoothing,
            ),
            metrics=["accuracy"],
        )
    train_ds = make_ds(df, splits["train"], cfg.image_size, cfg.batch_size, True,
                       architecture=cfg.architecture, drop_remainder=True)
    val_ds = make_ds(df, splits["val"], cfg.image_size, cfg.batch_size, False,
                     architecture=cfg.architecture)
    test_ds = make_ds(df, splits["test"], cfg.image_size, cfg.batch_size, False,
                      architecture=cfg.architecture)

    cw = class_weights(df, splits["train"], power=cfg.class_weight_power).numpy()
    ckpt = _MetricCheckpoint(val_ds, cfg.checkpoint_metric, t0)
    model.fit(
        train_ds,
        epochs=cfg.epochs,
        class_weight={i: float(w) for i, w in enumerate(cw)},
        callbacks=[
            tf.keras.callbacks.LearningRateScheduler(
                lambda e: cfg.lr * 0.5 * (1 + math.cos(math.pi * e / cfg.epochs))
            ),
            ckpt,
        ],
        verbose=2,
    )
    res = {
        "mode": label,
        "history": ckpt.rows,
        "val": evaluate(model, val_ds),
        "test": evaluate(model, test_ds, tta=cfg.test_time_augmentation),
        "wall_time_s": time.time() - t0,
        "num_params": int(model.count_params()),
        "privacy": {"epsilon": None, "note": "raw images pooled centrally; no formal guarantee"},
        "config": asdict(cfg),
    }
    model.save_weights(OUT_DIR / f"{label}_tpu.weights.h5")
    return res


# --------------------------------------------------------------------------- #
# Federated training
# --------------------------------------------------------------------------- #
def _federation_epsilon(cfg: Cfg, dp_on: bool, glob: Accountant, per_client: dict):
    """max over sites for record-level DP, single value for client-level DP."""
    if not dp_on:
        return None
    if cfg.dp_mode == "example":
        eps = [a.epsilon for a in per_client.values() if a.n]
        return max(eps) if eps else 0.0
    return glob.epsilon


def _flat(weights):
    return np.concatenate([w.ravel() for w in weights])


def _unflat(flat, template):
    out, off = [], 0
    for w in template:
        n = w.size
        out.append(flat[off : off + n].reshape(w.shape).astype(w.dtype))
        off += n
    return out


def _trainable_values(model):
    return [v.numpy() for v in model.trainable_variables]


def _set_trainable_values(model, values):
    for variable, value in zip(model.trainable_variables, values):
        variable.assign(value)


def train_federated(cfg: Cfg, df, splits, strategy, label=None) -> dict:
    """FedAvg/FedProx over `cfg.num_clients` simulated hospitals.

    One Keras model lives inside the TPU scope and is re-seeded with each
    client's starting weights; this avoids re-tracing the training graph for
    every client (the expensive part on TPU) while keeping client updates
    strictly independent. Only the trainable variables (a small head by
    default, since the pretrained backbone is frozen) are aggregated and
    communicated.
    """
    t0 = time.time()
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(cfg.seed)
    if cfg.federated_algorithm not in ("fedavg", "fedprox"):
        raise ValueError(f"unsupported federated_algorithm: {cfg.federated_algorithm!r}")
    if cfg.dp_mode == "example" and strategy.num_replicas_in_sync != 1:
        raise ValueError(
            "record-level DP-SGD (dp_mode='example') must run on one GPU/CPU: "
            "replicated noise accounting across replicas is not implemented"
        )
    rng = np.random.default_rng(cfg.seed)
    parts = dirichlet_partition(df, splits["train"], cfg)
    dp_on = cfg.dp_mode != "none"
    tag = label or (
        f"dp-{cfg.federated_algorithm}-{cfg.dp_mode}" if dp_on else cfg.federated_algorithm
    )

    dp_example_mode = cfg.dp_mode == "example"
    balanced_sampling = dp_example_mode or cfg.balanced_federated_sampling
    cw = class_weights(df, splits["train"], power=cfg.class_weight_power)
    with strategy.scope():
        model = build_model(
            image_size=cfg.image_size,
            architecture=cfg.architecture,
            pretrained=cfg.pretrained,
            fine_tune_layers=cfg.federated_fine_tune_layers,
        )
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=cfg.local_lr, weight_decay=cfg.weight_decay
        )
        # Class rebalancing under DP comes from balanced sampling, not weights.
        loss_fn = weighted_loss_fn(
            tf.ones_like(cw) if balanced_sampling else cw,
            strategy.num_replicas_in_sync,
            label_smoothing=cfg.label_smoothing,
        )
        # FedProx proximal centre: snapshot of the global trainable weights,
        # refreshed before every client update.
        prox_center = [
            tf.Variable(v.numpy(), trainable=False)
            for v in model.trainable_variables
        ]
        optimizer.build(model.trainable_variables)

    # True initial optimizer state (iterations, Adam slots, any lr variable),
    # restored before each client instead of zeroing.
    optimizer_initial_state = [v.numpy() for v in optimizer.variables]

    dp_example = cfg.dp_mode == "example"
    # For example-level DP the batch is split into `microbatches` groups whose
    # gradients are clipped independently before noise is added. microbatches ==
    # batch size gives textbook per-example DP-SGD; larger microbatches trade a
    # weaker (group-level) guarantee for speed. map_fn is used instead of
    # vectorized_map because the latter silently falls back to a while_loop for
    # models with normalization layers, which is slower on TPU.
    n_micro = min(cfg.microbatches, cfg.local_batch_size) if dp_example else 0

    def _per_microbatch_grads(x, y):
        mb = tf.shape(x)[0] // n_micro
        xs = tf.reshape(x, tf.concat([[n_micro, mb], tf.shape(x)[1:]], 0))
        ys = tf.reshape(y, [n_micro, mb])

        def one(args):
            xi, yi = args
            with tf.GradientTape() as tape:
                loss = loss_fn(yi, model(xi, training=True))
                if cfg.federated_algorithm == "fedprox":
                    loss = loss + _proximal_term()
            g = tape.gradient(loss, model.trainable_variables)
            flat = tf.concat([tf.reshape(t, [-1]) for t in g], 0)
            return flat * tf.minimum(1.0, cfg.l2_clip / (tf.norm(flat) + 1e-6))

        flats = tf.map_fn(one, (xs, ys), fn_output_signature=tf.float32)
        summed = tf.reduce_sum(flats, axis=0)
        summed += tf.random.normal(tf.shape(summed), stddev=cfg.noise_multiplier * cfg.l2_clip)
        summed /= tf.cast(n_micro, tf.float32)
        out, off = [], 0
        for v in model.trainable_variables:
            n = tf.size(v)
            out.append(tf.reshape(summed[off : off + n], tf.shape(v)))
            off += n
        return out

    def _proximal_term():
        return 0.5 * cfg.fedprox_mu * tf.add_n(
            [tf.reduce_sum(tf.square(w - w_g))
             for w, w_g in zip(model.trainable_variables, prox_center)]
        )

    @tf.function
    def train_step(iterator, steps):
        def step(batch):
            x, y = batch
            if dp_example:
                grads = _per_microbatch_grads(x, y)
                with tf.GradientTape() as tape:  # loss value for logging only
                    loss = loss_fn(y, model(x, training=False))
                    if cfg.federated_algorithm == "fedprox":
                        loss = loss + _proximal_term()
                del tape
            else:
                with tf.GradientTape() as tape:
                    loss = loss_fn(y, model(x, training=True))
                    if cfg.federated_algorithm == "fedprox":
                        loss = loss + _proximal_term()
                grads = tape.gradient(loss, model.trainable_variables)
                grads, _ = tf.clip_by_global_norm(grads, 5.0)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss

        total = 0.0
        for _ in tf.range(steps):
            total += strategy.run(step, args=(next(iterator),))
        return total / tf.cast(steps, tf.float32)

    # per-client distributed datasets, built once and reused across rounds
    client_ds, client_iters, client_steps = {}, {}, {}
    for cid, idx in enumerate(parts):
        gbatch = max(cfg.local_batch_size, strategy.num_replicas_in_sync)
        ds = make_ds(
            df, idx, cfg.image_size, gbatch, True, architecture=cfg.architecture,
            drop_remainder=True, balanced=balanced_sampling
        )
        if not dp_example_mode:
            ds = ds.repeat()
        client_ds[cid] = strategy.experimental_distribute_dataset(ds)
        client_iters[cid] = iter(client_ds[cid])
        client_steps[cid] = max(1, (len(idx) // gbatch)) * cfg.local_epochs

    val_ds = make_ds(df, splits["val"], cfg.image_size, cfg.batch_size, False,
                     architecture=cfg.architecture)
    test_ds = make_ds(df, splits["test"], cfg.image_size, cfg.batch_size, False,
                      architecture=cfg.architecture)

    # Only the trainable variables are aggregated/communicated; the frozen
    # pretrained backbone stays resident and identical on every client.
    template = _trainable_values(model)
    global_flat = _flat(template)
    secagg = SecAgg(seed=cfg.seed) if cfg.secure_aggregation else None
    accountant = Accountant(cfg.delta)
    per_client_acct = {cid: Accountant(cfg.delta) for cid in range(cfg.num_clients)}
    cohort_rate = cfg.clients_per_round / cfg.num_clients
    # Worst-case per-example inclusion probability at each site. With balanced
    # sampling a rare-class record is drawn far more often than 1/n, and the
    # accountant must use that larger rate.
    client_q = {}
    for cid, idx in enumerate(parts):
        counts = np.bincount(df.loc[idx, "label"].to_numpy(), minlength=len(CLASSES))
        present = max(int((counts > 0).sum()), 1)
        if dp_example_mode:
            rates = np.where(counts > 0, (cfg.local_batch_size / present) / np.clip(counts, 1, None), 0)
            client_q[cid] = float(min(np.max(rates), 0.5))
        else:
            client_q[cid] = float(min(1.0, cfg.local_batch_size / max(len(idx), 1)))
    metric = cfg.checkpoint_metric
    history, best, best_flat = [], {metric: -1.0}, global_flat.copy()

    for rnd in range(1, cfg.rounds + 1):
        cohort = sorted(rng.choice(cfg.num_clients, cfg.clients_per_round, replace=False))
        deltas, losses, sizes, norms = {}, {}, {}, {}
        for cid in cohort:
            start = _unflat(global_flat, template)
            _set_trainable_values(model, start)
            for variable, value in zip(prox_center, start):
                variable.assign(value)
            for variable, value in zip(optimizer.variables, optimizer_initial_state):
                variable.assign(value)  # fresh local optimizer state each round
            loss = train_step(client_iters[cid], tf.constant(client_steps[cid]))
            d = _flat(_trainable_values(model)) - global_flat
            norms[cid] = float(np.linalg.norm(d))
            if dp_on and cfg.dp_mode == "client":
                # clip the whole client update (DP-FedAvg). In "example" mode the
                # clipping already happened per microbatch inside the client.
                d = d * min(1.0, cfg.l2_clip / (norms[cid] + 1e-12))
            deltas[cid] = d
            losses[cid] = float(loss.numpy() if hasattr(loss, "numpy") else loss)
            sizes[cid] = len(parts[cid])

        # Client-level DP requires uniform weights: the aggregate's L2
        # sensitivity to one hospital is then exactly clip/m, which is what the
        # noise below is calibrated to. Size-weighting is used only without DP.
        m = len(cohort)
        n_total = sum(sizes.values())
        uniform = dp_on and cfg.dp_mode == "client"
        contribs = {}
        for cid in cohort:
            w = 1.0 / m if uniform else sizes[cid] / n_total
            c = deltas[cid] * w
            if dp_on and cfg.dp_mode == "client":
                # Distributed Gaussian noise: each client adds
                # sigma*clip/m/sqrt(m) so the cohort's independent noises sum in
                # variance to sigma*(clip/m) - no trusted aggregator needed.
                c = c + rng.normal(0, cfg.noise_multiplier * cfg.l2_clip / m / math.sqrt(m), c.shape)
            contribs[cid] = c

        if secagg is not None:
            payloads = {
                cid: secagg.mask_update(cid, cohort, rnd, contribs[cid]) for cid in cohort
            }
            aggregate = secagg.aggregate(payloads)
        else:
            aggregate = np.sum(list(contribs.values()), axis=0)
        global_flat = global_flat + cfg.server_lr * aggregate

        if dp_on:
            if cfg.dp_mode == "example":
                # Record-level guarantees compose in PARALLEL across sites (a
                # patient's images live at exactly one hospital), so the
                # federation's epsilon is the max over sites, not the sum.
                for cid in cohort:
                    per_client_acct[cid].step(
                        client_q[cid] * cohort_rate, cfg.noise_multiplier, client_steps[cid]
                    )
            else:
                accountant.step(cohort_rate, cfg.noise_multiplier, 1)

        _set_trainable_values(model, _unflat(global_flat, template))
        val = evaluate(model, val_ds)
        row = {
            "round": rnd,
            "cohort": [int(c) for c in cohort],
            "train_loss": float(np.mean(list(losses.values()))),
            "mean_update_norm": float(np.mean(list(norms.values()))),
            "epsilon": _federation_epsilon(cfg, dp_on, accountant, per_client_acct),
            "val_accuracy": val["accuracy"],
            "val_balanced_accuracy": val["balanced_accuracy"],
            "val_macro_f1": val["macro_f1"],
            "val_malignant_sensitivity": val["malignant_sensitivity"],
            "elapsed_s": time.time() - t0,
        }
        history.append(row)
        if val[metric] > best[metric]:
            best, best_flat = val, global_flat.copy()
        eps = f" eps={row['epsilon']:.2f}" if row["epsilon"] else ""
        print(
            f"[{tag}] round {rnd:>3}/{cfg.rounds} loss={row['train_loss']:.4f} "
            f"val_bacc={val['balanced_accuracy']:.4f} f1={val['macro_f1']:.4f}{eps}"
        )

    _set_trainable_values(model, _unflat(best_flat, template))
    test = evaluate(model, test_ds, tta=cfg.test_time_augmentation)
    print(f"[{tag}] TEST bacc={test['balanced_accuracy']:.4f} f1={test['macro_f1']:.4f}")
    model.save_weights(OUT_DIR / f"{tag}_tpu.weights.h5")
    return {
        "mode": tag,
        "history": history,
        "val": best,
        "test": test,
        "wall_time_s": time.time() - t0,
        "num_params": int(model.count_params()),
        "federation": {
            "num_clients": cfg.num_clients,
            "client_sizes": [int(len(p)) for p in parts],
            "clients": [
                {
                    "client_id": cid,
                    "name": f"Hospital-{cid + 1:02d}",
                    "num_samples": int(len(idx)),
                    "num_lesions": int(df.loc[idx, "lesion_id"].nunique()),
                    "label_counts": {
                        c: int((df.loc[idx, "label"] == i).sum()) for i, c in enumerate(CLASSES)
                    },
                }
                for cid, idx in enumerate(parts)
            ],
        },
        "privacy": (
            {**(
                {
                    **accountant.summary(),
                    "epsilon": _federation_epsilon(cfg, dp_on, accountant, per_client_acct),
                    "unit": "patient record" if dp_example_mode else "hospital participation",
                    "composition": "parallel across sites, sequential within a site"
                    if dp_example_mode
                    else "sequential over rounds",
                    "per_client_epsilon": {
                        str(c): a.epsilon for c, a in per_client_acct.items() if a.n
                    }
                    if dp_example_mode
                    else None,
                }
             ), "mode": cfg.dp_mode, "l2_clip": cfg.l2_clip,
             "noise_multiplier": cfg.noise_multiplier,
             "secure_aggregation": cfg.secure_aggregation}
            if dp_on
            else {"epsilon": None, "secure_aggregation": cfg.secure_aggregation,
                  "note": "FedAvg without DP: data stays local, no formal guarantee"}
        ),
        "communication": {
            "params_per_message": int(global_flat.size),
            "bytes_per_client_round": int(global_flat.size * 4),
            "total_uplink_mb": float(global_flat.size * 4 * cfg.clients_per_round * cfg.rounds / 1e6),
        },
        "config": asdict(cfg),
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def _write_results(results: dict) -> Path:
    out = OUT_DIR / "tpu_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[save] wrote {out}")
    return out


def main(mode="all", **overrides) -> dict:
    cfg = Cfg(**{k: v for k, v in overrides.items() if v is not None})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[data] {DATA_DIR}")
    strategy = get_tpu_strategy()
    df = load_metadata(cfg)
    splits = make_splits(df, cfg)
    print(f"[data] train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    results = {}
    if mode in ("tune", "tune2", "tune3", "tune4", "tune5", "tune6"):
        plans = {
            "tune": TUNING_PLAN,
            "tune2": TUNING_PLAN_V2,
            "tune3": TUNING_PLAN_V3,
            "tune4": TUNING_PLAN_V4,
            "tune5": TUNING_PLAN_V5,
            "tune6": TUNING_PLAN_V6,
        }
        plan = plans[mode]
        for kind, key, overrides_for_run in plan:
            run_cfg = Cfg(**{**asdict(cfg), **overrides_for_run})
            if kind == "federated":
                results[key] = train_federated(
                    run_cfg, df, splits, strategy, label=key
                )
            else:
                results[key] = train_centralized(
                    run_cfg, df, splits, strategy, label=key
                )
            _write_results(results)
    else:
        if mode in ("all", "centralized"):
            results["centralized"] = train_centralized(cfg, df, splits, strategy)
            _write_results(results)
        if mode in ("all", "federated"):
            c = Cfg(**{**asdict(cfg), "dp_mode": "none"})
            results["federated"] = train_federated(
                c, df, splits, strategy, label=cfg.federated_algorithm
            )
            _write_results(results)
        if mode in ("all", "federated_dp"):
            results["federated_dp"] = train_federated(cfg, df, splits, strategy)
            _write_results(results)

    out = _write_results(results)
    print(f"[done] wrote {out}")
    for k, v in results.items():
        t = v["test"]
        print(
            f"  {k:<16} acc={t['accuracy']:.4f} bacc={t['balanced_accuracy']:.4f} "
            f"f1={t['macro_f1']:.4f} eps={v['privacy'].get('epsilon')}"
        )
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="tune6",
                    choices=["all", "centralized", "federated", "federated_dp",
                             "tune", "tune2", "tune3", "tune4", "tune5", "tune6"])
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--image-size", dest="image_size", type=int, default=None)
    ap.add_argument("--architecture", default=None,
                    choices=["efficientnetb0", "efficientnetb1", "efficientnetb2",
                             "efficientnetv2b0", "efficientnetv2b1", "dermcnn"])
    ap.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    ap.add_argument("--num-clients", dest="num_clients", type=int, default=None)
    ap.add_argument("--federated-algorithm", dest="federated_algorithm", default=None,
                    choices=["fedavg", "fedprox"])
    ap.add_argument("--fedprox-mu", dest="fedprox_mu", type=float, default=None)
    ap.add_argument("--local-batch-size", dest="local_batch_size", type=int, default=None)
    ap.add_argument("--local-epochs", dest="local_epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--local-lr", dest="local_lr", type=float, default=None)
    ap.add_argument("--dp-mode", dest="dp_mode", default=None,
                    choices=["none", "client", "example"])
    ap.add_argument("--noise-multiplier", dest="noise_multiplier", type=float, default=None)
    ap.add_argument("--l2-clip", dest="l2_clip", type=float, default=None)
    ap.add_argument("--balanced-federated-sampling", dest="balanced_federated_sampling",
                    action="store_true", default=None)
    ap.add_argument("--no-test-time-augmentation", dest="test_time_augmentation",
                    action="store_false", default=None)
    a = vars(ap.parse_args())
    main(a.pop("mode"), **a)
