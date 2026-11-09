"""Secure aggregation: the server learns only the SUM of client updates.

Protocol (a practical subset of Bonawitz et al., CCS 2017 "Practical Secure
Aggregation for Privacy-Preserving Machine Learning"):

  Round setup
    1. Every client publishes an X25519 public key and an Ed25519 signing key.
    2. Each pair (i, j) derives a shared secret via X25519 ECDH and expands it
       with HKDF-SHA256 (salt = round number) into a 32-byte PRG seed. The
       server never sees these secrets.

  Masking
    3. Client i quantizes its model delta into Z_M fixed point, adds its local
       DP noise, then adds sum_{j>i} PRG(s_ij) and subtracts sum_{j<i} PRG(s_ij),
       all mod M. The result is uniformly random in Z_M: individually
       indistinguishable from noise, so the server (or a network eavesdropper)
       learns nothing about a single hospital's update.
    4. Client i signs the masked payload with Ed25519 -> integrity/authenticity.

  Aggregation
    5. Server verifies signatures and sums the masked vectors mod M. Pairwise
       masks cancel exactly (integer arithmetic, not float), leaving the true
       sum, which is then dequantized.
    6. Dropouts: survivors disclose only the pairwise masks they shared with
       the dropped clients (never their own update), letting the server strip
       those terms. Full Shamir-based recovery is a drop-in extension.

Only the *sum* is ever revealed; per-client DP noise on top of that gives the
formal guarantee (see accountant.py). Together they defend both against a
curious server and against membership inference on the released model.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

ShapeSpec = list[tuple[str, tuple[int, ...]]]


def flatten(state: dict[str, np.ndarray]) -> tuple[np.ndarray, ShapeSpec]:
    spec: ShapeSpec = [(k, tuple(v.shape)) for k, v in state.items()]
    if not spec:
        return np.zeros(0, dtype=np.float64), spec
    return np.concatenate([np.asarray(v, dtype=np.float64).ravel() for v in state.values()]), spec


def unflatten(vec: np.ndarray, spec: ShapeSpec) -> dict[str, np.ndarray]:
    out, off = {}, 0
    for name, shape in spec:
        n = int(np.prod(shape)) if shape else 1
        out[name] = vec[off : off + n].reshape(shape)
        off += n
    return out


class Quantizer:
    """Signed fixed-point <-> Z_M encoding with exact modular arithmetic.

    Values are clipped to +-clip_range before scaling; with client updates
    already L2-clipped this is a safe bound and keeps the modulus small.
    """

    def __init__(self, bits: int = 20, modulus_bits: int = 62, clip_range: float = 8.0):
        self.scale = float(1 << bits)
        self.modulus = np.uint64(1) << np.uint64(modulus_bits)
        self.clip_range = clip_range

    def encode(self, x: np.ndarray) -> np.ndarray:
        # Silent saturation here would corrupt the aggregate in a way that is
        # very hard to debug (masks still cancel, the *sum* is just wrong), so
        # surface it loudly instead.
        overflow = int(np.count_nonzero(np.abs(x) > self.clip_range))
        if overflow:
            warnings.warn(
                f"{overflow}/{x.size} values exceed the fixed-point range "
                f"+-{self.clip_range} and will saturate; raise clip_range or "
                f"clip client updates harder.",
                RuntimeWarning,
                stacklevel=2,
            )
        q = np.rint(np.clip(x, -self.clip_range, self.clip_range) * self.scale).astype(np.int64)
        # two's complement into [0, M)
        m = np.int64(self.modulus)
        return (q % m).astype(np.uint64)

    def decode(self, x: np.ndarray) -> np.ndarray:
        m = np.int64(self.modulus)
        v = x.astype(np.int64) % m
        v = np.where(v > m // 2, v - m, v)
        return v.astype(np.float64) / self.scale

    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # inputs are < 2^62 so the sum fits in uint64 before reduction
        return (a + b) % self.modulus

    def sub(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a + (self.modulus - b)) % self.modulus


def _prg(seed: bytes, size: int, modulus: np.uint64) -> np.ndarray:
    """Deterministic uniform mask vector in Z_M from a 32-byte seed."""
    rng = np.random.default_rng(list(np.frombuffer(seed, dtype=np.uint32)))
    return rng.integers(0, int(modulus), size=size, dtype=np.uint64, endpoint=False)


def _derive_seed(shared: bytes, round_no: int, lo: int, hi: int) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=round_no.to_bytes(8, "big"),
        info=f"fedderm-secagg-pair-{lo}-{hi}".encode(),
    ).derive(shared)


@dataclass
class SecureAggregationClient:
    """Client-side of the protocol; holds the long-lived keys of one hospital."""

    client_id: int
    quantizer: Quantizer = field(default_factory=Quantizer)

    def __post_init__(self) -> None:
        self._dh = X25519PrivateKey.generate()
        self._sig = Ed25519PrivateKey.generate()

    @property
    def dh_public_bytes(self) -> bytes:
        return self._dh.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )

    @property
    def sig_public_bytes(self) -> bytes:
        return self._sig.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )

    def _pair_seed(self, peer_dh: bytes, peer_id: int, round_no: int) -> bytes:
        shared = self._dh.exchange(X25519PublicKey.from_public_bytes(peer_dh))
        lo, hi = sorted((self.client_id, peer_id))
        return _derive_seed(shared, round_no, lo, hi)

    def pair_mask(self, peer_dh: bytes, peer_id: int, round_no: int, size: int) -> np.ndarray:
        return _prg(self._pair_seed(peer_dh, peer_id, round_no), size, self.quantizer.modulus)

    def mask_update(
        self,
        update: np.ndarray,
        peers: dict[int, bytes],
        round_no: int,
    ) -> tuple[np.ndarray, bytes]:
        """Quantize + pairwise-mask `update`, returning (payload, signature)."""
        q = self.quantizer
        y = q.encode(update)
        for pid, pk in sorted(peers.items()):
            if pid == self.client_id:
                continue
            mask = self.pair_mask(pk, pid, round_no, update.size)
            y = q.add(y, mask) if pid > self.client_id else q.sub(y, mask)
        sig = self._sig.sign(y.tobytes() + round_no.to_bytes(8, "big"))
        return y, sig


@dataclass
class SecureAggregator:
    """Server-side aggregator. Sees only masked payloads."""

    quantizer: Quantizer = field(default_factory=Quantizer)
    verify_signatures: bool = True

    def aggregate(
        self,
        payloads: dict[int, np.ndarray],
        signatures: dict[int, bytes] | None = None,
        sig_public_keys: dict[int, bytes] | None = None,
        round_no: int = 0,
        weights: dict[int, float] | None = None,
        dropout_masks: Iterable[np.ndarray] = (),
    ) -> np.ndarray:
        """Return the dequantized *sum* (weighted mean if `weights` given).

        Note masks cancel only for unweighted sums, so weighting is applied by
        the clients themselves (they upload w_i * delta_i) and the server just
        divides by sum(w_i) - exactly what FedAvg needs.
        """
        if not payloads:
            raise ValueError("no payloads to aggregate")
        q = self.quantizer
        if self.verify_signatures:
            if signatures is None or sig_public_keys is None:
                raise ValueError("signatures required (set verify_signatures=False to skip)")
            for cid, y in payloads.items():
                Ed25519PublicKey.from_public_bytes(sig_public_keys[cid]).verify(
                    signatures[cid], y.tobytes() + round_no.to_bytes(8, "big")
                )
        total = np.zeros_like(next(iter(payloads.values())))
        for y in payloads.values():
            total = q.add(total, y)
        for m in dropout_masks:  # strip masks shared with clients that dropped
            total = q.sub(total, m)
        summed = q.decode(total)
        if weights:
            denom = sum(weights.values())
            if denom > 0:
                summed = summed / denom
        return summed


def simulate_round(
    updates: dict[int, np.ndarray],
    round_no: int = 0,
    weights: dict[int, float] | None = None,
    clients: dict[int, SecureAggregationClient] | None = None,
    quantizer: Quantizer | None = None,
) -> dict:
    """Run one full masked round end-to-end. Used by tests and the web demo."""
    quantizer = quantizer or Quantizer()
    clients = clients or {cid: SecureAggregationClient(cid, quantizer) for cid in updates}
    pub = {cid: c.dh_public_bytes for cid, c in clients.items()}
    payloads, sigs = {}, {}
    for cid, upd in updates.items():
        w = (weights or {}).get(cid, 1.0)
        payloads[cid], sigs[cid] = clients[cid].mask_update(upd * w, pub, round_no)
    agg = SecureAggregator(quantizer).aggregate(
        payloads,
        sigs,
        {cid: c.sig_public_bytes for cid, c in clients.items()},
        round_no=round_no,
        weights=weights,
    )
    return {"aggregate": agg, "payloads": payloads, "clients": clients}


def masked_stats(payload: np.ndarray, quantizer: Quantizer | None = None) -> dict:
    """What the server actually observes for one client (for the UI)."""
    q = quantizer or Quantizer()
    dec = q.decode(payload)
    return {
        "mean": float(dec.mean()),
        "std": float(dec.std()),
        "min": float(dec.min()),
        "max": float(dec.max()),
        "sample": [float(v) for v in dec[:8]],
    }


def clip_by_l2(vec: np.ndarray, clip: float) -> tuple[np.ndarray, float]:
    norm = float(np.linalg.norm(vec))
    factor = min(1.0, clip / (norm + 1e-12))
    return vec * factor, norm


def gaussian_noise(size: int, sigma: float, clip: float, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, sigma * clip, size=size)


def split_noise(size: int, sigma: float, clip: float, num_clients: int, rng) -> np.ndarray:
    """Distributed DP: each of n clients adds sigma*clip/sqrt(n) noise so the
    aggregate carries exactly sigma*clip - no trusted server needed."""
    return rng.normal(0.0, sigma * clip / np.sqrt(num_clients), size=size)


def as_vector(state: dict[str, np.ndarray]) -> tuple[np.ndarray, ShapeSpec]:
    return flatten(state)


def as_state(vec: np.ndarray, spec: Sequence[tuple[str, tuple[int, ...]]]) -> dict[str, np.ndarray]:
    return unflatten(vec, list(spec))
