import math

import numpy as np

from fedderm.privacy.accountant import (
    RDPAccountant,
    compute_epsilon,
    noise_for_target_epsilon,
    rdp_sampled_gaussian,
)


def test_epsilon_decreases_with_noise():
    eps = [compute_epsilon(0.01, s, 1000)[0] for s in (0.5, 0.8, 1.0, 2.0, 4.0)]
    assert all(a > b for a, b in zip(eps, eps[1:]))


def test_epsilon_increases_with_steps():
    e1 = compute_epsilon(0.01, 1.0, 100)[0]
    e2 = compute_epsilon(0.01, 1.0, 1000)[0]
    assert e2 > e1 > 0


def test_no_noise_is_infinite():
    assert compute_epsilon(0.01, 0.0, 10)[0] == float("inf")


def test_full_batch_matches_gaussian_mechanism():
    """At q=1 the sampled Gaussian degenerates to the Gaussian mechanism:
    RDP(alpha) = alpha / (2 sigma^2)."""
    sigma, orders = 2.0, (2.0, 4.0, 8.0)
    got = rdp_sampled_gaussian(1.0, sigma, orders)
    want = np.array([a / (2 * sigma**2) for a in orders])
    assert np.allclose(got, want, rtol=1e-6)


def test_reference_value_against_published_accountants():
    """Canonical TF-Privacy MNIST DP-SGD setting: N=60000, batch=256,
    sigma=1.1, 15 epochs, delta=1e-5. Published accountants report
    eps ~= 1.19-1.20. We must land close to that and never *below* it
    (a lower epsilon would mean an unsound, over-optimistic guarantee).
    """
    eps, _ = compute_epsilon(256 / 60000, 1.1, 15 * 234, 1e-5)
    assert 1.19 <= eps < 1.40, eps


def test_binary_search_hits_target():
    sigma = noise_for_target_epsilon(0.02, 500, target_epsilon=3.0)
    eps, _ = compute_epsilon(0.02, sigma, 500)
    assert eps <= 3.0 + 1e-2


def test_accountant_composes_heterogeneous_steps():
    acc = RDPAccountant(delta=1e-5)
    acc.step(0.02, 1.0, 10)
    e1 = acc.epsilon
    acc.step(0.05, 1.0, 10)
    assert acc.epsilon > e1
    assert acc.summary()["num_compositions"] == 2
    assert math.isfinite(acc.epsilon)
