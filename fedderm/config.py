"""Central configuration for every stage of the pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "archive"
METADATA_CSV = ARCHIVE / "HAM10000_metadata.csv"
HMNIST_CSV = ARCHIVE / "hmnist_28_28_RGB.csv"
IMAGE_DIRS = (ARCHIVE / "HAM10000_images_part_1", ARCHIVE / "HAM10000_images_part_2")
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"


@dataclass
class DataConfig:
    """How the single physical dataset is carved into virtual hospitals."""

    num_clients: int = 10
    # Dirichlet concentration over the 7 diagnoses. Lower => more non-IID.
    dirichlet_alpha: float = 0.5
    test_fraction: float = 0.15
    val_fraction: float = 0.10
    # Splits are grouped by lesion_id so multiple shots of one lesion never
    # straddle a split boundary (prevents optimistic leakage).
    group_column: str = "lesion_id"
    seed: int = 1337
    image_size: int = 32  # local CPU path; the Kaggle TPU job uses 128x128


@dataclass
class TrainConfig:
    # Model width is shared by every run so centralized/federated/DP numbers are
    # comparable; 16 keeps the parameter count (~80k) low enough that DP noise,
    # which scales with sqrt(d), does not swamp the gradient signal.
    width: int = 16
    optimizer: str = "adamw"  # "adamw" | "sgd"
    lr: float = 1e-3          # AdamW + cosine decay
    momentum: float = 0.9
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 30  # centralized baseline
    class_balanced_loss: bool = True
    seed: int = 1337


@dataclass
class FedConfig:
    width: int = 16
    rounds: int = 30
    clients_per_round: int = 5     # cohort size m (of num_clients)
    local_epochs: int = 4
    local_batch_size: int = 32
    local_lr: float = 2e-2
    # Server optimizer. "sgd" + server_lr=1.0 is textbook FedAvg and, measured on
    # this partition, beats "adam" (FedAdam, Reddi et al. ICLR 2021) by a wide
    # margin: with only ~80k parameters and 4 local epochs the aggregate is
    # already well-scaled, and Adam's tau floor slows the first ~10 rounds.
    # FedAdam is kept because it helps when DP noise dominates the aggregate.
    server_optimizer: str = "sgd"
    server_lr: float = 1.0
    server_momentum: float = 0.0   # beta1 when server_optimizer="adam"
    server_beta2: float = 0.99
    server_tau: float = 1e-3       # adaptivity floor
    seed: int = 1337


@dataclass
class PrivacyConfig:
    enabled: bool = True
    # "example" = true DP-SGD inside each client (per-example clip + noise).
    # "client" = DP-FedAvg (clip the whole client delta, noise it at the server).
    mode: str = "example"
    l2_clip: float = 1.0
    noise_multiplier: float = 0.8
    # DP-SGD works on gradients whose norm is pinned to `l2_clip`, so the
    # effective step is lr/(1-momentum)*clip regardless of the loss landscape.
    # Heavy momentum therefore overshoots badly; 0.0 is the stable choice here.
    dp_local_momentum: float = 0.0
    # Class rebalancing for DP-SGD is done by sampling, because per-example
    # clipping nullifies loss weights (see federated.LocalClient._sampling_rates).
    balanced_sampling: bool = True
    dp_balanced_q_cap: float = 0.5
    delta: float = 1e-5
    secure_aggregation: bool = True
    # fixed-point scale for integer-domain masking
    quantization_bits: int = 20
    modulus_bits: int = 62


@dataclass
class ExperimentConfig:
    name: str = "local-cpu"
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    fed: FedConfig = field(default_factory=FedConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
