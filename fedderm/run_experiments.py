"""Run the full comparative study and write results/*.json.

    python -m fedderm.run_experiments --preset full
    python -m fedderm.run_experiments --preset quick      # ~5 min smoke test
    python -m fedderm.run_experiments --only centralized federated

Produces, in results/:
    centralized.json          all data pooled on one server (upper bound)
    federated.json            FedAvg + secure aggregation, no DP noise
    federated_dp_client.json  DP-FedAvg: client-level DP + secure aggregation
    federated_dp_example.json DP-SGD inside each client (per-example DP)
    comparison.json           side-by-side table + privacy/utility summary
    *.pt                      the corresponding model weights
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .config import RESULTS, DataConfig, FedConfig, PrivacyConfig, TrainConfig
from .torchimpl.centralized import train_centralized
from .torchimpl.data import build_federation
from .torchimpl.federated import train_federated

PRESETS = {
    # (centralized epochs, federated rounds, dp-sgd rounds)
    "quick": (4, 5, 3),
    "medium": (15, 20, 10),
    "full": (30, 45, 20),
}


def _strip(res: dict) -> tuple[dict, torch.nn.Module]:
    model = res.pop("model")
    return res, model


def _save(name: str, res: dict, model: torch.nn.Module) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), RESULTS / f"{name}.pt")
    print(f"[save] results/{name}.json (+ .pt)")


def build_comparison() -> dict:
    """Assemble the headline comparison table from whatever runs exist."""
    runs = {}
    for p in RESULTS.glob("*.json"):
        payload = json.loads(p.read_text(encoding="utf-8"))
        # results/ also holds non-run artefacts (comparison.json, privacy_sweep.json);
        # a run is anything carrying test metrics.
        if isinstance(payload, dict) and "test" in payload:
            runs[p.stem] = payload
    order = ["centralized", "federated", "federated_dp_client", "federated_dp_example"]
    table = []
    for name in [o for o in order if o in runs] + [k for k in runs if k not in order]:
        r = runs[name]
        t, priv = r["test"], r.get("privacy") or {}
        table.append(
            {
                "run": name,
                "mode": r.get("mode"),
                "data_centralised": name == "centralized",
                "accuracy": t["accuracy"],
                "balanced_accuracy": t["balanced_accuracy"],
                "macro_f1": t["macro_f1"],
                "macro_auc": t.get("macro_auc"),
                "malignant_sensitivity": t["malignant_sensitivity"],
                "malignant_specificity": t.get("malignant_specificity"),
                "epsilon": priv.get("epsilon"),
                "delta": priv.get("delta"),
                "dp_mode": priv.get("mode"),
                "secure_aggregation": priv.get("secure_aggregation", False),
                "wall_time_s": r.get("wall_time_s"),
                "rounds_or_epochs": len(r.get("history", [])),
            }
        )
    base = next((x for x in table if x["run"] == "centralized"), None)
    if base:
        for row in table:
            row["balanced_accuracy_gap_vs_centralized"] = (
                row["balanced_accuracy"] - base["balanced_accuracy"]
            )
            row["utility_retained_pct"] = (
                100.0 * row["balanced_accuracy"] / base["balanced_accuracy"]
                if base["balanced_accuracy"]
                else None
            )
    out = {
        "generated_at": time.time(),
        "table": table,
        "notes": [
            "All runs share the same lesion-grouped test split and the same model "
            "(GroupNorm CNN, 81,527 parameters), so differences are attributable "
            "to the training regime and not to capacity.",
            "Centralized is the privacy-violating upper bound: it requires pooling "
            "raw dermatoscopic images from every hospital.",
            "Federated runs never move an image; only masked parameter deltas.",
            "epsilon is reported at the stated delta; lower epsilon = stronger "
            "privacy = (usually) lower utility.",
        ],
    }
    (RESULTS / "comparison.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[save] results/comparison.json")
    return out


def print_table(cmp: dict) -> None:
    print("\n" + "=" * 108)
    print(
        f"{'run':<22}{'acc':>8}{'bal-acc':>10}{'macroF1':>10}{'AUC':>8}"
        f"{'mal-sens':>10}{'epsilon':>10}{'secagg':>8}{'time(s)':>10}"
    )
    print("-" * 108)
    for r in cmp["table"]:
        eps = "-" if r["epsilon"] is None else f"{r['epsilon']:.2f}"
        auc = "-" if r["macro_auc"] is None else f"{r['macro_auc']:.3f}"
        print(
            f"{r['run']:<22}{r['accuracy']:>8.4f}{r['balanced_accuracy']:>10.4f}"
            f"{r['macro_f1']:>10.4f}{auc:>8}{r['malignant_sensitivity']:>10.4f}"
            f"{eps:>10}{str(bool(r['secure_aggregation'])):>8}{r['wall_time_s']:>10.1f}"
        )
    print("=" * 108 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="full", choices=list(PRESETS))
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        choices=["centralized", "federated", "federated_dp_client", "federated_dp_example"],
    )
    ap.add_argument("--noise-multiplier", type=float, default=0.8, help="client-level DP sigma")
    ap.add_argument(
        "--noise-multiplier-example", type=float, default=0.3, help="example-level DP-SGD sigma"
    )
    ap.add_argument("--l2-clip", type=float, default=1.0)
    ap.add_argument("--table-only", action="store_true", help="just rebuild comparison.json")
    ap.add_argument("--sweep", action="store_true", help="also trace the privacy/utility frontier")
    ap.add_argument("--sweep-rounds", type=int, default=10)
    args = ap.parse_args()

    if args.table_only:
        print_table(build_comparison())
        return

    epochs, rounds, dp_rounds = PRESETS[args.preset]
    # `--only` with no values => run no main experiments (useful with --sweep);
    # omitting it entirely => run all four.
    todo = (
        args.only
        if args.only is not None
        else ["centralized", "federated", "federated_dp_client", "federated_dp_example"]
    )
    data_cfg = DataConfig()
    fed = build_federation(data_cfg)
    print(
        f"[data] {len(fed.splits['train'])} train / {len(fed.splits['val'])} val / "
        f"{len(fed.splits['test'])} test images; {data_cfg.num_clients} hospitals, "
        f"sizes {fed.report['size_min']}-{fed.report['size_max']}, "
        f"label-skew TV={fed.report['mean_total_variation']:.3f}"
    )

    if "centralized" in todo:
        res, model = _strip(train_centralized(fed, TrainConfig(epochs=epochs)))
        _save("centralized", res, model)

    if "federated" in todo:
        res, model = _strip(
            train_federated(
                fed,
                FedConfig(rounds=rounds),
                PrivacyConfig(enabled=False, secure_aggregation=True),
                label="fedavg-secagg",
            )
        )
        _save("federated", res, model)

    if "federated_dp_client" in todo:
        res, model = _strip(
            train_federated(
                fed,
                FedConfig(rounds=rounds),
                PrivacyConfig(
                    enabled=True,
                    mode="client",
                    l2_clip=args.l2_clip,
                    noise_multiplier=args.noise_multiplier,
                    secure_aggregation=True,
                ),
                label="dp-fedavg-client",
            )
        )
        _save("federated_dp_client", res, model)

    if "federated_dp_example" in todo:
        # Per-example DP-SGD needs a large local batch: the injected noise has
        # norm ~ sigma*C*sqrt(d)/B while the clipped signal has norm <= C, so the
        # SNR is ~B/(sigma*sqrt(d)). B=128 with d=81527 is the practical limit on
        # CPU. Many local steps (local_epochs=8) compensate for the small steps,
        # and momentum is off because clipped gradients have a fixed norm.
        res, model = _strip(
            train_federated(
                fed,
                FedConfig(
                    rounds=dp_rounds, local_batch_size=128, local_epochs=8, local_lr=0.04
                ),
                PrivacyConfig(
                    enabled=True,
                    mode="example",
                    l2_clip=2.0,
                    noise_multiplier=args.noise_multiplier_example,
                    dp_local_momentum=0.0,
                    balanced_sampling=True,
                    secure_aggregation=True,
                ),
                label="dp-sgd-example",
            )
        )
        _save("federated_dp_example", res, model)

    if args.sweep:
        run_privacy_sweep(fed, rounds=args.sweep_rounds)

    print_table(build_comparison())


def run_privacy_sweep(fed, rounds: int = 10, sigmas=(0.3, 1.0, 3.0)) -> dict:
    """Trace the privacy/utility frontier: same everything, different noise.

    This is the headline scientific result of the project. At this scale (81k
    parameters, 10 silos, ~7k training images) DP is expensive: the injected
    noise has norm ~sigma*clip*sqrt(d)/batch while the clipped signal is bounded
    by clip, so utility falls off quickly as epsilon tightens.
    """
    points = []
    for sigma in sigmas:
        res, _ = _strip(
            train_federated(
                fed,
                FedConfig(rounds=rounds, local_batch_size=128, local_epochs=8, local_lr=0.04),
                PrivacyConfig(
                    enabled=True,
                    mode="example",
                    l2_clip=2.0,
                    noise_multiplier=sigma,
                    dp_local_momentum=0.0,
                    balanced_sampling=True,
                    secure_aggregation=True,
                ),
                label=f"dp-sgd-sigma{sigma}",
            )
        )
        points.append(
            {
                "noise_multiplier": sigma,
                "epsilon": res["privacy"]["epsilon"],
                "delta": res["privacy"]["delta"],
                "balanced_accuracy": res["test"]["balanced_accuracy"],
                "macro_f1": res["test"]["macro_f1"],
                "malignant_sensitivity": res["test"]["malignant_sensitivity"],
                "noise_to_signal": sigma * (res["num_params"] ** 0.5) / 128,
            }
        )
        print(
            f"[sweep] sigma={sigma} eps={points[-1]['epsilon']:.1f} "
            f"bacc={points[-1]['balanced_accuracy']:.4f}"
        )
    out = {"rounds": rounds, "mode": "example", "points": points}
    (RESULTS / "privacy_sweep.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[save] results/privacy_sweep.json")
    return out


if __name__ == "__main__":
    main()
