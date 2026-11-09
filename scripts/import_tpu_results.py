"""Import a Kaggle TPU run into results/ so the API and dashboard pick it up.

    python scripts/import_tpu_results.py path/to/tpu_results.json [--suffix _tpu]

The TPU script writes one JSON containing every run; the API expects one file per
run (that is what makes the dashboard's comparison table generic).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fedderm.config import RESULTS

NAME_MAP = {
    "centralized": "centralized",
    "federated": "federated",
    "federated_dp": "federated_dp_client",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--suffix", default="_tpu", help="appended to each run name")
    args = ap.parse_args()

    payload = json.loads(args.path.read_text(encoding="utf-8"))
    RESULTS.mkdir(parents=True, exist_ok=True)
    for key, run in payload.items():
        # dp_mode decides whether this was hospital-level or record-level DP
        name = NAME_MAP.get(key, key)
        if key == "federated_dp" and (run.get("privacy") or {}).get("mode") == "example":
            name = "federated_dp_example"
        out = RESULTS / f"{name}{args.suffix}.json"
        out.write_text(json.dumps(run, indent=2), encoding="utf-8")
        t = run.get("test", {})
        print(
            f"[import] {out.name:<34} bacc={t.get('balanced_accuracy', float('nan')):.4f} "
            f"eps={(run.get('privacy') or {}).get('epsilon')}"
        )
    print("\nRestart the API (or just reload the dashboard) to see them.")


if __name__ == "__main__":
    main()
