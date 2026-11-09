import numpy as np
import pytest

from fedderm.privacy.secure_agg import (
    Quantizer,
    SecureAggregationClient,
    SecureAggregator,
    flatten,
    masked_stats,
    simulate_round,
    unflatten,
)


def test_masks_cancel_exactly():
    rng = np.random.default_rng(0)
    updates = {i: rng.normal(0, 0.05, size=500) for i in range(6)}
    out = simulate_round(updates, round_no=3)
    expected = sum(updates.values())
    # error is bounded by quantization only (2^-20 per element)
    assert np.max(np.abs(out["aggregate"] - expected)) < 1e-4


def test_weighted_mean_matches_fedavg():
    rng = np.random.default_rng(1)
    updates = {i: rng.normal(0, 0.1, size=300) for i in range(4)}
    weights = {0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1}
    out = simulate_round(updates, round_no=1, weights=weights)
    expected = sum(weights[i] * u for i, u in updates.items())
    assert np.allclose(out["aggregate"], expected, atol=1e-4)


def test_individual_payload_hides_update():
    """A single masked payload must look like uniform noise, not like the update."""
    rng = np.random.default_rng(2)
    updates = {i: np.full(2000, 0.01 * (i + 1)) for i in range(5)}
    out = simulate_round(updates, round_no=7)
    stats = masked_stats(out["payloads"][0])
    assert abs(stats["mean"] - 0.01) > 1.0       # nothing like the true 0.01 mean
    assert stats["std"] > 100.0                  # spread across the whole group


def test_signature_tampering_detected():
    from cryptography.exceptions import InvalidSignature

    q = Quantizer()
    clients = {i: SecureAggregationClient(i, q) for i in range(3)}
    pub = {i: c.dh_public_bytes for i, c in clients.items()}
    updates = {i: np.ones(50) * 0.1 for i in range(3)}
    payloads, sigs = {}, {}
    for i, u in updates.items():
        payloads[i], sigs[i] = clients[i].mask_update(u, pub, 0)
    payloads[1] = payloads[1] + np.uint64(12345)  # man-in-the-middle edit
    with pytest.raises(InvalidSignature):
        SecureAggregator(q).aggregate(
            payloads, sigs, {i: c.sig_public_bytes for i, c in clients.items()}, round_no=0
        )


def test_dropout_recovery():
    """Server can still decode when a client drops, using disclosed pair masks."""
    q = Quantizer()
    ids = [0, 1, 2, 3]
    clients = {i: SecureAggregationClient(i, q) for i in ids}
    pub = {i: c.dh_public_bytes for i, c in clients.items()}
    rng = np.random.default_rng(5)
    updates = {i: rng.normal(0, 0.05, 400) for i in ids}
    payloads, sigs = {}, {}
    for i in ids:
        payloads[i], sigs[i] = clients[i].mask_update(updates[i], pub, 2)
    dropped = 3
    del payloads[dropped], sigs[dropped]
    # survivors reveal only the masks they share with the dropped client
    recovery = []
    for i in ids:
        if i == dropped:
            continue
        m = clients[i].pair_mask(pub[dropped], dropped, 2, 400)
        recovery.append(m if dropped > i else q.sub(np.zeros_like(m), m))
    agg = SecureAggregator(q).aggregate(
        payloads,
        sigs,
        {i: clients[i].sig_public_bytes for i in payloads},
        round_no=2,
        dropout_masks=recovery,
    )
    expected = sum(updates[i] for i in ids if i != dropped)
    assert np.allclose(agg, expected, atol=1e-4)


def test_overflow_is_reported_not_silent():
    q = Quantizer(clip_range=1.0)
    with pytest.warns(RuntimeWarning, match="exceed the fixed-point range"):
        q.encode(np.array([0.5, 2.0, -3.0]))


def test_quantization_roundtrip_precision():
    q = Quantizer(bits=20)
    x = np.linspace(-4, 4, 1001)
    assert np.max(np.abs(q.decode(q.encode(x)) - x)) < 1e-6


def test_flatten_roundtrip():
    state = {"a": np.arange(6.0).reshape(2, 3), "b": np.ones((4,))}
    vec, spec = flatten(state)
    back = unflatten(vec, spec)
    assert np.allclose(back["a"], state["a"]) and np.allclose(back["b"], state["b"])
