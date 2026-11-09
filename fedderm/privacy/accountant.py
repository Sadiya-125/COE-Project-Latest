"""Renyi differential privacy accountant for the sampled Gaussian mechanism.

Implements the moments/RDP accounting of Mironov, Talwar & Zhang (2019),
"Renyi Differential Privacy of the Sampled Gaussian Mechanism", for integer
orders, plus the standard RDP -> (eps, delta) conversion of Balle et al. /
Canonne-Kamath-Steinke (the improved bound used by Opacus & TF-Privacy).

Two accounting granularities are supported, matching the two DP modes:

* example-level: one step = one Poisson-subsampled minibatch inside a client,
  q = batch_size / client_dataset_size, so epsilon is the guarantee given to a
  single patient record held by that client.
* client-level:  one step = one federated round, q = cohort_size / num_clients,
  so epsilon is the guarantee given to an entire hospital's participation.

Everything is pure NumPy: no TF-Privacy/Opacus dependency, so the identical
accountant runs locally and inside the Kaggle TPU notebook.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.special import binom, log_ndtr

# Orders used for the RDP curve, matching the grid used by TF-Privacy/Opacus.
# Fractional orders below 11 matter: the epsilon-minimizing order for realistic
# DP-SGD settings usually sits there.
DEFAULT_ORDERS: tuple[float, ...] = tuple(
    [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64)) + [128, 256, 512]
)


def _log_add(a: float, b: float) -> float:
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    return max(a, b) + math.log1p(math.exp(-abs(a - b)))


def _log_sub(a: float, b: float) -> float:
    """log(exp(a) - exp(b)), requires a >= b."""
    if b == -np.inf:
        return a
    if a == b:
        return -np.inf
    return a + math.log1p(-math.exp(b - a))


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _rdp_sampled_gaussian_int(q: float, sigma: float, alpha: int) -> float:
    """RDP epsilon of the sampled Gaussian mechanism at *integer* order alpha.

    Uses the closed-form binomial expansion (Mironov et al. 2019, Sec. 3.3):
        eps(alpha) = 1/(alpha-1) * log( sum_{k=0}^{alpha} C(alpha,k)
                        (1-q)^{alpha-k} q^k exp(k(k-1)/(2 sigma^2)) )
    computed in log space for numerical stability.
    """
    if q == 0.0:
        return 0.0
    if q == 1.0:
        return alpha / (2.0 * sigma**2)
    log_a = -np.inf
    for k in range(alpha + 1):
        term = (
            _log_comb(alpha, k)
            + (alpha - k) * math.log1p(-q)
            + k * math.log(q)
            + (k * k - k) / (2.0 * sigma**2)
        )
        log_a = _log_add(log_a, term)
    return float(log_a / (alpha - 1))


def _log_erfc(x: float) -> float:
    """log(erfc(x)), stable in the far tail (needed for large alpha)."""
    return math.log(2) + log_ndtr(-x * 2**0.5)


def _rdp_sampled_gaussian_frac(q: float, sigma: float, alpha: float) -> float:
    """RDP epsilon at *fractional* order alpha.

    The Renyi divergence of the sampled Gaussian has no binomial closed form
    for non-integer alpha; we use the two-sided integral decomposition of
    Mironov et al. (2019, Sec. 3.3) split at the crossing point z0, with the
    alternating generalized-binomial series evaluated in log space. Fractional
    orders matter in practice: the epsilon-minimizing order for typical DP-SGD
    settings frequently lies between 1 and 3.
    """
    log_a0, log_a1 = -np.inf, -np.inf
    z0 = sigma**2 * math.log(1 / q - 1) + 0.5
    i = 0
    while True:
        coef = binom(alpha, i)
        log_coef = math.log(abs(coef))
        j = alpha - i
        log_t0 = log_coef + i * math.log(q) + j * math.log1p(-q)
        log_t1 = log_coef + j * math.log(q) + i * math.log1p(-q)
        log_e0 = math.log(0.5) + _log_erfc((i - z0) / (math.sqrt(2) * sigma))
        log_e1 = math.log(0.5) + _log_erfc((z0 - j) / (math.sqrt(2) * sigma))
        log_s0 = log_t0 + (i * i - i) / (2 * sigma**2) + log_e0
        log_s1 = log_t1 + (j * j - j) / (2 * sigma**2) + log_e1
        if coef > 0:
            log_a0, log_a1 = _log_add(log_a0, log_s0), _log_add(log_a1, log_s1)
        else:
            log_a0, log_a1 = _log_sub(log_a0, log_s0), _log_sub(log_a1, log_s1)
        i += 1
        if max(log_s0, log_s1) < -30 or i > 1000:
            break
    return float(_log_add(log_a0, log_a1) / (alpha - 1))


def rdp_sampled_gaussian(q: float, sigma: float, orders=DEFAULT_ORDERS) -> np.ndarray:
    """RDP curve of the sampled Gaussian mechanism over the given orders."""
    if q == 0.0:
        return np.zeros(len(orders))
    out = []
    for a in orders:
        if a <= 1.0:
            out.append(np.inf)
        elif q == 1.0:
            out.append(a / (2.0 * sigma**2))
        elif float(a).is_integer():
            out.append(_rdp_sampled_gaussian_int(q, sigma, int(a)))
        else:
            out.append(_rdp_sampled_gaussian_frac(q, sigma, float(a)))
    return np.asarray(out, dtype=float)


def rdp_to_dp(rdp: np.ndarray, delta: float, orders=DEFAULT_ORDERS) -> tuple[float, float]:
    """Convert an RDP curve to (eps, best order) at a target delta.

    Uses the tightened conversion of Canonne, Kamath & Steinke (2020):
        eps = rdp + log((alpha-1)/alpha) - (log delta + log alpha)/(alpha - 1)
    """
    orders = np.asarray(orders, dtype=float)
    rdp = np.asarray(rdp, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        eps = (
            rdp
            + np.log1p(-1.0 / orders)
            - (np.log(delta) + np.log(orders)) / (orders - 1.0)
        )
    eps = np.where(np.isfinite(eps), eps, np.inf)
    eps = np.maximum(eps, 0.0)
    i = int(np.argmin(eps))
    return float(eps[i]), float(orders[i])


def compute_epsilon(
    q: float, noise_multiplier: float, steps: int, delta: float = 1e-5, orders=DEFAULT_ORDERS
) -> tuple[float, float]:
    """(eps, optimal order) after `steps` sampled-Gaussian applications."""
    if noise_multiplier <= 0 or steps <= 0:
        return float("inf"), 0.0
    rdp = rdp_sampled_gaussian(q, noise_multiplier, orders) * steps
    return rdp_to_dp(rdp, delta, orders)


def noise_for_target_epsilon(
    q: float,
    steps: int,
    target_epsilon: float,
    delta: float = 1e-5,
    lo: float = 0.3,
    hi: float = 64.0,
    tol: float = 1e-3,
) -> float:
    """Smallest noise multiplier meeting a target epsilon (binary search)."""
    if compute_epsilon(q, hi, steps, delta)[0] > target_epsilon:
        raise ValueError("target epsilon unreachable within sigma <= 64")
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if compute_epsilon(q, mid, steps, delta)[0] > target_epsilon:
            lo = mid
        else:
            hi = mid
    return hi


@dataclass
class RDPAccountant:
    """Accumulates RDP across heterogeneous steps and reports (eps, delta).

    Heterogeneity matters in federated learning: clients have different dataset
    sizes, so each contributes a different sampling rate q.
    """

    delta: float = 1e-5
    orders: tuple[float, ...] = DEFAULT_ORDERS
    _rdp: np.ndarray = field(init=False)
    history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._rdp = np.zeros(len(self.orders), dtype=float)

    def step(self, q: float, noise_multiplier: float, steps: int = 1, tag: str = "") -> None:
        if steps <= 0:
            return
        if noise_multiplier <= 0:
            self._rdp = np.full_like(self._rdp, np.inf)
        else:
            self._rdp = self._rdp + rdp_sampled_gaussian(q, noise_multiplier, self.orders) * steps
        self.history.append(
            {"q": q, "sigma": noise_multiplier, "steps": steps, "tag": tag, "epsilon": self.epsilon}
        )

    @property
    def epsilon(self) -> float:
        return rdp_to_dp(self._rdp, self.delta, self.orders)[0]

    @property
    def best_order(self) -> float:
        return rdp_to_dp(self._rdp, self.delta, self.orders)[1]

    def summary(self) -> dict:
        eps, order = rdp_to_dp(self._rdp, self.delta, self.orders)
        return {
            "epsilon": eps,
            "delta": self.delta,
            "optimal_order": order,
            "num_compositions": len(self.history),
        }


if __name__ == "__main__":  # quick sanity check / reference numbers
    for sigma in (0.5, 0.8, 1.0, 1.5):
        e, a = compute_epsilon(q=0.01, noise_multiplier=sigma, steps=1000, delta=1e-5)
        print(f"sigma={sigma:>4}  q=0.01  steps=1000 -> eps={e:8.3f} (alpha*={a})")
