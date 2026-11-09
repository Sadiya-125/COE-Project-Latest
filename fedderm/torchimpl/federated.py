"""Federated training architecture (FedAvg) with DP-SGD + secure aggregation.

Flow of one round r:

    server  --- global weights theta_r --->  cohort S_r (sampled hospitals)
    client i: theta_i <- LocalTrain(theta_r, D_i)          (never uploads data)
              delta_i <- theta_i - theta_r
              DP:  per-example clipping + Gaussian noise (mode="example")
                   or clip(delta_i) + distributed noise    (mode="client")
              SecAgg: quantize, pairwise-mask, sign
    server  <-- masked payloads ---  cohort
    server: verify -> sum (masks cancel) -> theta_{r+1} = theta_r + eta_s * sum

The server only ever holds masked bytes and the final aggregate, so no single
hospital's update is observable, and the aggregate itself is differentially
private. Data never leaves the client process boundary in this simulation, and
the same code path runs against real remote clients by swapping `LocalClient`
for an RPC stub (see server/README notes).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, grad, vmap

from .. import CLASSES
from ..config import RESULTS, DataConfig, FedConfig, PrivacyConfig
from ..privacy.accountant import RDPAccountant
from ..privacy.secure_agg import (
    Quantizer,
    SecureAggregationClient,
    SecureAggregator,
    clip_by_l2,
)
from .data import Federation, build_federation
from .metrics import evaluate
from .models import build_model, count_params, get_flat_params, set_flat_params


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
@dataclass
class LocalClient:
    """One hospital: owns its data indices and its cryptographic identity."""

    client_id: int
    indices: np.ndarray
    fed: Federation
    fed_cfg: FedConfig
    priv: PrivacyConfig
    class_weights: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.crypto = SecureAggregationClient(
            self.client_id, Quantizer(self.priv.quantization_bits, self.priv.modulus_bits)
        )
        self._data = self.fed.dataset(self.indices, train=True)
        self.rng = np.random.default_rng(self.fed_cfg.seed * 1000 + self.client_id)
        self._q = self._sampling_rates()

    def _sampling_rates(self) -> np.ndarray:
        """Per-example Poisson inclusion probability.

        Plain training rebalances the 67%-`nv` dataset with a class-weighted
        loss. That trick is incompatible with DP-SGD: per-example clipping caps
        every gradient at `l2_clip`, which erases the up-weighting of rare
        classes and collapses the model onto `nv`. Under DP the only way to
        rebalance is to change how *often* a class is sampled, so for the
        example-level DP path we draw each class at ~batch/7 per step.

        The per-example rate is capped at `dp_balanced_q_cap`: an example that
        appears in every batch would otherwise carry a much weaker privacy
        guarantee (and be memorised) than the rest of the cohort.
        """
        n = self.num_samples
        base = min(1.0, self.fed_cfg.local_batch_size / max(n, 1))
        if not (self.priv.enabled and self.priv.mode == "example" and self.priv.balanced_sampling):
            return np.full(n, base)
        labels = self.fed.labels[self.indices]
        counts = np.bincount(labels, minlength=len(CLASSES)).astype(float)
        per_class_budget = self.fed_cfg.local_batch_size / max((counts > 0).sum(), 1)
        with np.errstate(divide="ignore"):
            rates = np.where(counts > 0, per_class_budget / np.clip(counts, 1, None), 0.0)
        rates = np.clip(rates, 0.0, self.priv.dp_balanced_q_cap)
        return rates[labels]

    @property
    def num_samples(self) -> int:
        return len(self.indices)

    @property
    def sampling_rate(self) -> float:
        """The rate the accountant must use: the worst-case (largest) per-example
        inclusion probability at this site."""
        return float(self._q.max())

    @property
    def local_steps(self) -> int:
        n, b = self.num_samples, self.fed_cfg.local_batch_size
        return max(1, int(np.ceil(n / b))) * self.fed_cfg.local_epochs

    def _batches(self):
        """Poisson subsampling - the sampling scheme the RDP accountant assumes."""
        n = self.num_samples
        for _ in range(self.local_steps):
            idx = np.flatnonzero(self.rng.random(n) < self._q)
            if idx.size == 0:
                idx = self.rng.integers(0, n, size=1)
            xb = torch.stack([self._data[int(i)][0] for i in idx])
            yb = torch.stack([self._data[int(i)][1] for i in idx])
            yield xb, yb

    # -- local optimisation ------------------------------------------------- #
    def _train_plain(self, model: nn.Module, criterion) -> float:
        opt = torch.optim.SGD(model.parameters(), lr=self.fed_cfg.local_lr, momentum=0.9)
        total, nb = 0.0, 0
        model.train()
        for xb, yb in self._batches():
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach())
            nb += 1
        return total / max(nb, 1)

    def _train_dpsgd(self, model: nn.Module, criterion) -> float:
        """True per-example DP-SGD via torch.func (vmap over per-sample grads)."""
        params = {k: v.detach().clone() for k, v in model.named_parameters()}
        buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
        clip, sigma, lr = self.priv.l2_clip, self.priv.noise_multiplier, self.fed_cfg.local_lr

        # Unweighted, reduction="sum" - for a single example this is just its own
        # loss, and it makes the per-example semantics explicit. Class
        # rebalancing is handled by `_sampling_rates`, never by loss weights
        # (see the note there).
        per_example_criterion = nn.CrossEntropyLoss(reduction="sum")
        # Textbook DP-SGD normalises by the *expected* lot size, not the realised
        # one: under Poisson sampling the realised count is itself data-dependent,
        # so dividing by it would leak (a little) outside the accounted-for noise.
        expected_lot = max(float(self._q.sum()), 1.0)

        def loss_fn(p, b, x, y):
            out = functional_call(model, (p, b), (x.unsqueeze(0),))
            return per_example_criterion(out, y.unsqueeze(0))

        per_sample_grad = vmap(grad(loss_fn), in_dims=(None, None, 0, 0), randomness="different")
        velocity = {k: torch.zeros_like(v) for k, v in params.items()}
        total, nb = 0.0, 0
        model.train()
        for xb, yb in self._batches():
            g = per_sample_grad(params, buffers, xb, yb)  # dict of [B, ...] tensors
            keys = list(g.keys())
            flat = torch.cat([g[k].reshape(xb.size(0), -1) for k in keys], dim=1)
            norms = flat.norm(dim=1)
            factor = (clip / (norms + 1e-6)).clamp(max=1.0)          # per-example clipping
            flat = flat * factor.unsqueeze(1)
            summed = flat.sum(0)
            summed = summed + torch.normal(0.0, sigma * clip, size=summed.shape)  # Gaussian noise
            summed = summed / expected_lot
            off = 0
            for k in keys:
                n = params[k].numel()
                gk = summed[off : off + n].view_as(params[k])
                off += n
                velocity[k] = self.priv.dp_local_momentum * velocity[k] + gk
                params[k] = params[k] - lr * velocity[k]
            with torch.no_grad():
                total += float(criterion(functional_call(model, (params, buffers), (xb,)), yb))
            nb += 1
        with torch.no_grad():
            for k, v in model.named_parameters():
                v.copy_(params[k])
        return total / max(nb, 1)

    def local_update(self, global_flat: np.ndarray, round_no: int) -> dict:
        """Run local training and return the (privatised) flat delta."""
        model = build_model(width=self.fed_cfg.width)
        set_flat_params(model, global_flat)
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        dp_example = self.priv.enabled and self.priv.mode == "example"
        loss = self._train_dpsgd(model, criterion) if dp_example else self._train_plain(model, criterion)

        delta = get_flat_params(model) - global_flat
        norm = float(np.linalg.norm(delta))
        if self.priv.enabled and self.priv.mode == "client":
            delta, norm = clip_by_l2(delta, self.priv.l2_clip)
        return {
            "client_id": self.client_id,
            "delta": delta,
            "loss": loss,
            "num_samples": self.num_samples,
            "update_norm": norm,
            "local_steps": self.local_steps,
            "sampling_rate": self.sampling_rate,
        }


# --------------------------------------------------------------------------- #
# Server
# --------------------------------------------------------------------------- #
def _epsilon(priv: PrivacyConfig, glob: RDPAccountant, per_client: dict) -> float | None:
    """Federation-level epsilon: max over sites for record-level DP (parallel
    composition), the single global value for client-level DP."""
    if not priv.enabled:
        return None
    if priv.mode == "example":
        eps = [a.epsilon for a in per_client.values() if a.history]
        return max(eps) if eps else 0.0
    return glob.epsilon



def train_federated(
    fed: Federation,
    fed_cfg: FedConfig,
    priv: PrivacyConfig,
    device: str = "cpu",
    eval_every: int = 1,
    verbose: bool = True,
    label: str | None = None,
) -> dict:
    torch.manual_seed(fed_cfg.seed)
    rng = np.random.default_rng(fed_cfg.seed)

    global_model = build_model(width=fed_cfg.width, seed=fed_cfg.seed).to(device)
    global_flat = get_flat_params(global_model)
    quant = Quantizer(priv.quantization_bits, priv.modulus_bits)
    aggregator = SecureAggregator(quant)

    class_weights = fed.class_weights()
    clients = [
        LocalClient(cid, idx, fed, fed_cfg, priv, class_weights)
        for cid, idx in enumerate(fed.parts)
    ]
    num_clients = len(clients)
    cohort_rate = fed_cfg.clients_per_round / num_clients
    # Client-level DP has one global accountant (the unit of privacy is a whole
    # hospital's participation). Example-level DP gets one accountant per
    # hospital: a patient's images live at exactly one site, so the per-record
    # guarantees compose in *parallel* across sites, and the federation's
    # epsilon is the max over hospitals rather than the sum. Summing here would
    # inflate epsilon by ~num_clients for no reason.
    accountant = RDPAccountant(delta=priv.delta)
    per_client_acct = {c: RDPAccountant(delta=priv.delta) for c in range(num_clients)}

    val_loader = fed.loader(fed.splits["val"], 256)
    test_loader = fed.loader(fed.splits["test"], 256)
    server_m = np.zeros_like(global_flat)  # FedAdam first moment
    server_v = np.zeros_like(global_flat)  # FedAdam second moment

    history, best = [], {"balanced_accuracy": -1.0}
    best_flat = global_flat.copy()
    t0 = time.time()
    tag = label or ("dp-fedavg" if priv.enabled else "fedavg")

    for rnd in range(1, fed_cfg.rounds + 1):
        cohort = sorted(rng.choice(num_clients, size=fed_cfg.clients_per_round, replace=False))
        updates = [clients[c].local_update(global_flat, rnd) for c in cohort]

        # Aggregation weights, normalised inside the cohort so the masked
        # integers stay small; the server only needs sum(w_i * delta_i).
        #
        # Client-level DP *requires* uniform weights: the L2 sensitivity of the
        # aggregate to swapping one hospital is w_max * clip, so size-weighting
        # would let the largest hospital dominate the sensitivity while the
        # noise calibration assumed 1/m. Uniform weighting (DP-FedAvg, McMahan
        # et al. 2018) makes the sensitivity exactly clip/m.
        m = len(updates)
        n_total = sum(u["num_samples"] for u in updates)
        uniform = priv.enabled and priv.mode == "client"
        weights = {
            u["client_id"]: (1.0 / m if uniform else u["num_samples"] / n_total) for u in updates
        }

        peers = {c: clients[c].crypto.dh_public_bytes for c in cohort}
        payloads, sigs, sig_keys = {}, {}, {}
        for u in updates:
            cid = u["client_id"]
            contrib = u["delta"] * weights[cid]
            if priv.enabled and priv.mode == "client":
                # Distributed DP. Target noise on the aggregate is
                # sigma * (clip / m) = sigma * sensitivity. Each of the m clients
                # therefore adds sigma*clip/m/sqrt(m); independent Gaussians sum
                # in variance, so the total is exactly right and no trusted
                # aggregator ever holds an un-noised value.
                contrib = contrib + clients[cid].rng.normal(
                    0.0,
                    priv.noise_multiplier * priv.l2_clip / m / np.sqrt(m),
                    size=contrib.shape,
                )
            if priv.secure_aggregation:
                payloads[cid], sigs[cid] = clients[cid].crypto.mask_update(contrib, peers, rnd)
                sig_keys[cid] = clients[cid].crypto.sig_public_bytes
            else:
                payloads[cid] = contrib

        if priv.secure_aggregation:
            aggregate = aggregator.aggregate(payloads, sigs, sig_keys, round_no=rnd)
        else:
            aggregate = np.sum(list(payloads.values()), axis=0)

        # Server update. FedAdam treats the aggregate as a pseudo-gradient; the
        # per-coordinate rescaling is what lets DP-noised, non-IID updates still
        # make progress at a sane number of rounds.
        if fed_cfg.server_optimizer == "adam":
            b1, b2, tau = fed_cfg.server_momentum, fed_cfg.server_beta2, fed_cfg.server_tau
            server_m = b1 * server_m + (1 - b1) * aggregate
            server_v = b2 * server_v + (1 - b2) * aggregate**2
            step = server_m / (np.sqrt(server_v) + tau)
        else:
            if fed_cfg.server_momentum > 0:
                server_m = fed_cfg.server_momentum * server_m + aggregate
                step = server_m
            else:
                step = aggregate
        global_flat = global_flat + fed_cfg.server_lr * step

        # privacy accounting for this round
        if priv.enabled:
            if priv.mode == "example":
                for u in updates:
                    # q is the per-record probability of being touched in a step,
                    # amplified by the chance that this hospital is sampled at all.
                    per_client_acct[u["client_id"]].step(
                        q=u["sampling_rate"] * cohort_rate,
                        noise_multiplier=priv.noise_multiplier,
                        steps=u["local_steps"],
                        tag=f"round{rnd}",
                    )
            else:
                accountant.step(cohort_rate, priv.noise_multiplier, 1, tag=f"round{rnd}")

        row = {
            "round": rnd,
            "cohort": [int(c) for c in cohort],
            "train_loss": float(np.mean([u["loss"] for u in updates])),
            "mean_update_norm": float(np.mean([u["update_norm"] for u in updates])),
            "epsilon": _epsilon(priv, accountant, per_client_acct),
            "elapsed_s": time.time() - t0,
        }
        if rnd % eval_every == 0 or rnd == fed_cfg.rounds:
            set_flat_params(global_model, global_flat)
            val = evaluate(global_model, val_loader, device)
            row.update(
                val_accuracy=val["accuracy"],
                val_balanced_accuracy=val["balanced_accuracy"],
                val_macro_f1=val["macro_f1"],
                val_malignant_sensitivity=val["malignant_sensitivity"],
            )
            if val["balanced_accuracy"] > best["balanced_accuracy"]:
                best, best_flat = val, global_flat.copy()
            if verbose:
                eps = row["epsilon"]
                print(
                    f"[{tag}] round {rnd:>3}/{fed_cfg.rounds} loss={row['train_loss']:.4f} "
                    f"val_bacc={val['balanced_accuracy']:.4f} f1={val['macro_f1']:.4f}"
                    + (f" eps={eps:.2f}" if eps else "")
                )
        history.append(row)

    set_flat_params(global_model, best_flat)
    test = evaluate(global_model, test_loader, device)
    if verbose:
        print(f"[{tag}] TEST bacc={test['balanced_accuracy']:.4f} f1={test['macro_f1']:.4f}")
    return {
        "mode": tag,
        "config": {"fed": fed_cfg.__dict__, "privacy": priv.__dict__},
        "num_params": count_params(global_model),
        "federation": fed.report,
        "history": history,
        "val": best,
        "test": test,
        "wall_time_s": time.time() - t0,
        "privacy": (
            {
                "epsilon": _epsilon(priv, accountant, per_client_acct),
                "delta": priv.delta,
                "mode": priv.mode,
                "unit": "patient record" if priv.mode == "example" else "hospital participation",
                "composition": "parallel across sites, sequential within a site"
                if priv.mode == "example"
                else "sequential over rounds",
                "per_client_epsilon": {
                    str(c): a.epsilon for c, a in per_client_acct.items() if a.history
                }
                if priv.mode == "example"
                else None,
                "secure_aggregation": priv.secure_aggregation,
                "l2_clip": priv.l2_clip,
                "noise_multiplier": priv.noise_multiplier,
            }
            if priv.enabled
            else {
                "epsilon": None,
                "secure_aggregation": priv.secure_aggregation,
                "note": "FedAvg without DP noise: raw images never leave the hospital and "
                "the server sees only masked sums, but there is no formal bound on what "
                "the released weights reveal about a patient.",
            }
        ),
        "communication": {
            "params_per_message": int(global_flat.size),
            "bytes_per_client_round": int(global_flat.size * 8),
            "total_uplink_mb": float(
                global_flat.size * 8 * fed_cfg.clients_per_round * fed_cfg.rounds / 1e6
            ),
        },
        "model": global_model,
    }


def run(rounds: int | None = None, dp: bool = True, out: Path | None = None) -> dict:
    fed = build_federation(DataConfig())
    fed_cfg, priv = FedConfig(), PrivacyConfig(enabled=dp)
    if rounds:
        fed_cfg.rounds = rounds
    res = train_federated(fed, fed_cfg, priv)
    model = res.pop("model")
    out = out or RESULTS / ("federated_dp.json" if dp else "federated.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), out.with_suffix(".pt"))
    print(f"[fed] saved {out}")
    return res


if __name__ == "__main__":
    run()
