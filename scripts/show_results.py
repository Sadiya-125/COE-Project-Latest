"""Print a quick summary of whatever is in results/ (dev convenience)."""

import json
from pathlib import Path

from fedderm.config import RESULTS

if __name__ == "__main__":
    for p in sorted(Path(RESULTS).glob("*.json")):
        if p.stem in ("comparison", "privacy_sweep"):
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        t, priv = r.get("test", {}), r.get("privacy") or {}
        eps = priv.get("epsilon")
        print(
            f"{p.stem:<26} acc={t.get('accuracy', 0):.4f} bacc={t.get('balanced_accuracy', 0):.4f} "
            f"f1={t.get('macro_f1', 0):.4f} auc={t.get('macro_auc') or float('nan'):.3f} "
            f"malsens={t.get('malignant_sensitivity', 0):.4f} "
            f"eps={'-' if eps is None else format(eps, '.2f')} "
            f"rounds={len(r.get('history', []))} time={r.get('wall_time_s', 0) / 60:.1f}m"
        )
