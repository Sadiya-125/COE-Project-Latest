import numpy as np
import pytest

from fedderm import CLASSES
from fedderm.config import METADATA_CSV
from fedderm.data.metadata import (
    dirichlet_partition,
    load_metadata,
    make_splits,
    partition_report,
)

pytestmark = pytest.mark.skipif(not METADATA_CSV.exists(), reason="archive/ not present")


@pytest.fixture(scope="module")
def df():
    return load_metadata(resolve_paths=False)


def test_metadata_shape(df):
    assert len(df) == 10015
    assert df["label"].between(0, 6).all()
    assert df["age"].isna().sum() == 0
    assert sorted(df["dx"].unique()) == sorted(CLASSES)


def test_splits_are_disjoint_and_lesion_grouped(df):
    s = make_splits(df, seed=7)
    idx = np.concatenate([s["train"], s["val"], s["test"]])
    assert len(idx) == len(df) and len(set(idx.tolist())) == len(df)
    lesions = {k: set(df.loc[v, "lesion_id"]) for k, v in s.items()}
    assert not lesions["train"] & lesions["test"]
    assert not lesions["train"] & lesions["val"]
    assert not lesions["val"] & lesions["test"]


def test_every_class_present_in_test_split(df):
    s = make_splits(df, seed=7)
    assert set(df.loc[s["test"], "label"]) == set(range(len(CLASSES)))


def test_partition_covers_train_pool_exactly(df):
    s = make_splits(df, seed=7)
    parts = dirichlet_partition(df, s["train"], num_clients=10, alpha=0.5, seed=7)
    allp = np.concatenate(parts)
    assert len(allp) == len(s["train"])
    assert set(allp.tolist()) == set(s["train"].tolist())


def test_partition_is_non_iid(df):
    s = make_splits(df, seed=7)
    iid = partition_report(df, dirichlet_partition(df, s["train"], 10, alpha=100.0, seed=7))
    skewed = partition_report(df, dirichlet_partition(df, s["train"], 10, alpha=0.3, seed=7))
    assert skewed["mean_total_variation"] > iid["mean_total_variation"]


def test_clients_do_not_share_lesions(df):
    s = make_splits(df, seed=7)
    parts = dirichlet_partition(df, s["train"], 8, 0.5, seed=7)
    seen: set[str] = set()
    for p in parts:
        les = set(df.loc[p, "lesion_id"])
        assert not les & seen
        seen |= les
