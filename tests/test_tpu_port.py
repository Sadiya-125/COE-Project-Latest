"""The Kaggle/TPU port must agree with the local package, or the comparison
between CPU and TPU numbers is meaningless.

TensorFlow is not installable on this machine (no cp314 wheels), so the module
is imported with a stubbed `tensorflow`. That is enough to exercise everything
that decides *what* is trained on and *what privacy is claimed*: the splits, the
Dirichlet partition, the RDP accountant and the secure-aggregation arithmetic.
The Keras/TPU code paths are exercised on Kaggle itself.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from fedderm.config import METADATA_CSV, DataConfig
from fedderm.data.metadata import dirichlet_partition, load_metadata, make_splits
from fedderm.privacy.accountant import compute_epsilon

TPU_SRC = Path(__file__).resolve().parent.parent / "kaggle" / "fedderm_tpu.py"

pytestmark = pytest.mark.skipif(not METADATA_CSV.exists(), reason="archive/ not present")


@pytest.fixture(scope="module")
def tpu():
    """Import kaggle/fedderm_tpu.py with tensorflow stubbed out."""
    for name in ("tensorflow", "tensorflow.keras"):
        sys.modules.setdefault(name, MagicMock())
    spec = importlib.util.spec_from_file_location("fedderm_tpu", TPU_SRC)
    mod = importlib.util.module_from_spec(spec)
    # must be registered before exec: @dataclass resolves annotations through
    # sys.modules[cls.__module__]
    sys.modules["fedderm_tpu"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_same_splits_as_local_package(tpu):
    cfg = tpu.Cfg()
    df_tpu = tpu.load_metadata(cfg)
    splits_tpu = tpu.make_splits(df_tpu, cfg)

    d = DataConfig()
    df = load_metadata(resolve_paths=False)
    splits = make_splits(df, d.test_fraction, d.val_fraction, d.group_column, d.seed)

    for key in ("train", "val", "test"):
        assert np.array_equal(splits_tpu[key], splits[key]), f"{key} split diverged"


def test_same_dirichlet_partition_as_local_package(tpu):
    cfg = tpu.Cfg()
    df_tpu = tpu.load_metadata(cfg)
    parts_tpu = tpu.dirichlet_partition(df_tpu, tpu.make_splits(df_tpu, cfg)["train"], cfg)

    d = DataConfig()
    df = load_metadata(resolve_paths=False)
    splits = make_splits(df, d.test_fraction, d.val_fraction, d.group_column, d.seed)
    parts = dirichlet_partition(
        df, splits["train"], d.num_clients, d.dirichlet_alpha, d.group_column, d.seed
    )

    assert len(parts_tpu) == len(parts)
    for a, b in zip(parts_tpu, parts):
        assert np.array_equal(a, b)


def test_accountant_matches_local_accountant(tpu):
    for q, sigma, steps in [(0.5, 0.8, 45), (0.05, 1.2, 480), (256 / 60000, 1.1, 3510)]:
        acc = tpu.Accountant(1e-5)
        acc.step(q, sigma, steps)
        local, _ = compute_epsilon(q, sigma, steps, 1e-5)
        assert acc.epsilon == pytest.approx(local, rel=1e-6)


def test_secure_aggregation_masks_cancel(tpu):
    sec = tpu.SecAgg(seed=7)
    rng = np.random.default_rng(0)
    updates = {cid: rng.normal(0, 0.05, 800) for cid in range(5)}
    cohort = list(updates)
    payloads = {cid: sec.mask_update(cid, cohort, 3, updates[cid]) for cid in cohort}
    recovered = sec.aggregate(payloads)
    assert np.allclose(recovered, sum(updates.values()), atol=1e-4)


def test_single_masked_payload_reveals_nothing(tpu):
    sec = tpu.SecAgg(seed=11)
    cohort = [0, 1, 2]
    updates = {cid: np.full(1000, 0.02) for cid in cohort}
    payloads = {cid: sec.mask_update(cid, cohort, 1, updates[cid]) for cid in cohort}
    seen = sec.decode(payloads[0])
    assert abs(seen.mean() - 0.02) > 1.0
    assert seen.std() > 100.0


def test_problem_statement_defaults(tpu):
    """The PDF-aligned training recipe must be the default configuration."""
    cfg = tpu.Cfg()
    assert cfg.architecture == "efficientnetb0"
    assert cfg.pretrained is True
    assert cfg.image_size == 224
    assert cfg.epochs == 12
    assert cfg.batch_size == 32
    assert cfg.lr == pytest.approx(2e-4)
    assert cfg.weight_decay == pytest.approx(1e-4)
    assert cfg.label_smoothing == pytest.approx(0.05)
    assert cfg.checkpoint_metric == "macro_f1"
    assert cfg.federated_algorithm == "fedprox"
    assert cfg.fedprox_mu == pytest.approx(0.01)
    assert cfg.federated_fine_tune_layers == 0  # backbone frozen in federation
    assert cfg.rounds == 20
    assert cfg.test_time_augmentation is True
    assert cfg.balanced_federated_sampling is False
    assert cfg.class_weight_power == pytest.approx(1.0)


def test_tuning_plan_is_targeted_and_non_private(tpu):
    assert [entry[:2] for entry in tpu.TUNING_PLAN] == [
        ("federated", "fedavg_balanced_e4"),
        ("federated", "fedprox_balanced_e4"),
        ("centralized", "centralized_ft40"),
        ("centralized", "centralized_ft80"),
    ]
    for kind, _, overrides in tpu.TUNING_PLAN:
        if kind == "federated":
            assert overrides["dp_mode"] == "none"
            assert overrides["balanced_federated_sampling"] is True
            assert overrides["local_epochs"] == 4
            assert overrides["rounds"] == 30


def test_tuning_plan_v2_targets_capacity_and_full_participation(tpu):
    assert [entry[:2] for entry in tpu.TUNING_PLAN_V2] == [
        ("federated", "fedavg_full10_balanced_e4"),
        ("centralized", "centralized_b0_ft120_w075"),
        ("centralized", "centralized_b1_ft100_w075"),
        ("centralized", "centralized_b2_ft120_w075"),
    ]
    fed = tpu.TUNING_PLAN_V2[0][2]
    assert fed["dp_mode"] == "none"
    assert fed["clients_per_round"] == 10
    assert fed["balanced_federated_sampling"] is True
    assert fed["local_epochs"] == 4
    for kind, _, overrides in tpu.TUNING_PLAN_V2[1:]:
        assert kind == "centralized"
        assert overrides["class_weight_power"] == pytest.approx(0.75)


def test_tuning_plan_v3_targets_b0_balance_and_federated_finetuning(tpu):
    assert [entry[:2] for entry in tpu.TUNING_PLAN_V3] == [
        ("federated", "fedavg_ft20_balanced_e2"),
        ("centralized", "centralized_b0_ft80_w085"),
        ("centralized", "centralized_b0_ft120_w085"),
        ("centralized", "centralized_b0_ft120_w090"),
        ("centralized", "centralized_b0_ft120_w085_s000"),
        ("centralized", "centralized_b0_256_ft120_w085"),
    ]
    fed = tpu.TUNING_PLAN_V3[0][2]
    assert fed["dp_mode"] == "none"
    assert fed["federated_fine_tune_layers"] == 20
    assert fed["balanced_federated_sampling"] is True
    assert fed["local_epochs"] == 2
    for kind, _, overrides in tpu.TUNING_PLAN_V3[1:]:
        assert kind == "centralized"
        assert overrides["architecture"] == "efficientnetb0"
        assert 0.0 < overrides["class_weight_power"] <= 1.0


def test_tuning_plan_v4_is_validation_guided_and_non_private(tpu):
    assert [entry[:2] for entry in tpu.TUNING_PLAN_V4] == [
        ("centralized", "centralized_b0_256_ft120_w090"),
        ("centralized", "centralized_b0_256_ft160_w085"),
        ("centralized", "centralized_b0_288_ft120_w085"),
        ("centralized", "centralized_v2b0_ft100_w085"),
        ("centralized", "centralized_v2b0_ft160_w085"),
        ("centralized", "centralized_v2b1_ft120_w085"),
    ]
    for kind, _, overrides in tpu.TUNING_PLAN_V4:
        assert kind == "centralized"
        assert overrides["architecture"] in {
            "efficientnetb0", "efficientnetv2b0", "efficientnetv2b1"
        }
        assert overrides["class_weight_power"] in {0.85, 0.90}
        assert overrides["epochs"] == 20


def test_tuning_plan_v5_refines_validation_winner(tpu):
    assert [entry[:2] for entry in tpu.TUNING_PLAN_V5] == [
        ("centralized", "centralized_b0_288_ft120_w0875"),
        ("centralized", "centralized_b0_288_ft120_w090"),
        ("centralized", "centralized_b0_288_ft160_w085"),
        ("centralized", "centralized_b0_320_ft120_w085"),
        ("centralized", "centralized_b0_288_ft120_w085_b16"),
        ("centralized", "centralized_b0_288_ft120_w085_lr5"),
    ]
    for kind, _, overrides in tpu.TUNING_PLAN_V5:
        assert kind == "centralized"
        assert overrides["architecture"] == "efficientnetb0"
        assert overrides["image_size"] in {288, 320}
        assert 0.0 < overrides["class_weight_power"] <= 1.0
        assert overrides["epochs"] == 20


def test_tuning_plan_v6_combines_validation_supported_factors(tpu):
    assert [entry[:2] for entry in tpu.TUNING_PLAN_V6] == [
        ("centralized", "centralized_b0_288_ft120_w0875_b16"),
        ("centralized", "centralized_b0_320_ft120_w0875_b16"),
        ("federated", "fedavg_ft20_balanced_e3"),
        ("federated", "fedavg_ft40_balanced_e2"),
    ]
    for kind, _, overrides in tpu.TUNING_PLAN_V6[:2]:
        assert kind == "centralized"
        assert overrides["architecture"] == "efficientnetb0"
        assert overrides["batch_size"] == 16
        assert overrides["class_weight_power"] == pytest.approx(0.875)
        assert overrides["epochs"] == 20
    for kind, _, overrides in tpu.TUNING_PLAN_V6[2:]:
        assert kind == "federated"
        assert overrides["dp_mode"] == "none"
        assert overrides["federated_algorithm"] == "fedavg"
        assert overrides["balanced_federated_sampling"] is True
        assert overrides["clients_per_round"] == 5
        assert overrides["rounds"] == 35


def test_flat_unflat_round_trip(tpu):
    template = [
        np.arange(6, dtype=np.float32).reshape(2, 3),
        np.full((4,), -0.5, dtype=np.float64),
        np.array(2.5, dtype=np.float32),
    ]
    flat = tpu._flat(template)
    assert flat.size == sum(w.size for w in template)
    restored = tpu._unflat(flat, template)
    for original, got in zip(template, restored):
        assert got.shape == original.shape
        assert got.dtype == original.dtype
        np.testing.assert_array_equal(got, original)


def test_discovers_versioned_kaggle_mount(tpu, tmp_path):
    mounted = tmp_path / "datasets" / "kmader" / "skin-cancer-mnist-ham10000" / "versions" / "2"
    mounted.mkdir(parents=True)
    (mounted / "HAM10000_metadata.csv").touch()
    fallback = tmp_path / "archive"
    assert tpu.discover_data_dir(tmp_path, fallback) == mounted


def test_federation_epsilon_uses_parallel_composition(tpu):
    """Record-level epsilon must be the max over sites, not the sum."""
    cfg = tpu.Cfg(dp_mode="example")
    per_client = {}
    for cid in range(4):
        a = tpu.Accountant(1e-5)
        a.step(0.05, 1.0, 100 * (cid + 1))
        per_client[cid] = a
    glob = tpu.Accountant(1e-5)
    eps = tpu._federation_epsilon(cfg, True, glob, per_client)
    assert eps == pytest.approx(max(a.epsilon for a in per_client.values()))
    assert eps < sum(a.epsilon for a in per_client.values())
