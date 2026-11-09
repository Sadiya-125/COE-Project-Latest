"""Model loading + single-image inference for the serving prototype."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fedderm import CLASS_NAMES, CLASSES, MALIGNANT
from fedderm.config import ARTIFACTS, DataConfig, RESULTS
from fedderm.torchimpl.data import MEAN, STD
from fedderm.torchimpl.models import build_model

# Which checkpoint to serve, in order of preference: we prefer the privacy
# preserving model, because that is the point of the system.
CHECKPOINT_PREFERENCE = [
    ("federated_dp", RESULTS / "federated_dp.pt"),
    ("federated", RESULTS / "federated.pt"),
    ("centralized", RESULTS / "centralized.pt"),
]


class Diagnoser:
    def __init__(self, width: int = 16, image_size: int = DataConfig().image_size):
        self.image_size = image_size
        self.width = width
        self.model = build_model(width=width)
        self.model.eval()
        self.source = "random-init"
        self.checkpoint: Path | None = None

    def load_best(self) -> str:
        for name, path in CHECKPOINT_PREFERENCE:
            if path.exists():
                state = torch.load(path, map_location="cpu")
                self.model.load_state_dict(state)
                self.model.eval()
                self.source, self.checkpoint = name, path
                return name
        return self.source

    def load(self, path: Path, name: str | None = None) -> str:
        self.model.load_state_dict(torch.load(path, map_location="cpu"))
        self.model.eval()
        self.source, self.checkpoint = name or path.stem, path
        return self.source

    # -- preprocessing mirrors fedderm/data/preprocess.py exactly ------------ #
    def preprocess(self, payload: bytes) -> torch.Tensor:
        with Image.open(io.BytesIO(payload)) as im:
            im = im.convert("RGB")
            w, h = im.size
            s = min(w, h)
            im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
            im = im.resize((self.image_size, self.image_size), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32) / 255.0
        arr = (arr - MEAN) / STD
        return torch.from_numpy(arr.transpose(2, 0, 1)[None].copy())

    @torch.no_grad()
    def predict(self, payload: bytes, tta: bool = True) -> dict:
        x = self.preprocess(payload)
        views = [x]
        if tta:  # flip/rotation TTA: cheap and stabilises low-resolution inputs
            views += [torch.flip(x, [3]), torch.flip(x, [2]), torch.rot90(x, 1, [2, 3])]
        logits = torch.stack([self.model(v) for v in views]).mean(0)
        probs = torch.softmax(logits, dim=1)[0].numpy()
        order = np.argsort(-probs)
        malignant_p = float(sum(probs[CLASSES.index(c)] for c in MALIGNANT))
        top = CLASSES[int(order[0])]
        return {
            "predicted_class": top,
            "predicted_label": CLASS_NAMES[top],
            "confidence": float(probs[order[0]]),
            "malignant_probability": malignant_p,
            "triage": _triage(malignant_p),
            "probabilities": [
                {
                    "code": CLASSES[i],
                    "name": CLASS_NAMES[CLASSES[i]],
                    "probability": float(probs[i]),
                    "malignant": CLASSES[i] in MALIGNANT,
                }
                for i in order
            ],
            "model_source": self.source,
            "tta_views": len(views),
        }


def _triage(malignant_p: float) -> dict:
    """Map probability to an actionable band. Thresholds are deliberately
    conservative: in dermatology a false negative is far costlier than a false
    positive, so anything above 15% is escalated."""
    if malignant_p >= 0.5:
        return {"level": "urgent", "message": "High malignancy probability - urgent dermatologist review"}
    if malignant_p >= 0.15:
        return {"level": "review", "message": "Elevated malignancy probability - specialist review advised"}
    return {"level": "routine", "message": "Low malignancy probability - routine monitoring"}


def sample_images(n: int = 12, seed: int = 0) -> list[dict]:
    """A few held-out test images the UI can offer as one-click examples.

    Only images from the *test* split are exposed, so the demo never shows the
    model something it trained on.
    """
    from fedderm.data.metadata import load_metadata, make_splits

    cfg = DataConfig()
    df = load_metadata()
    splits = make_splits(df, cfg.test_fraction, cfg.val_fraction, cfg.group_column, cfg.seed)
    test = df.loc[splits["test"]]
    rng = np.random.default_rng(seed)
    picks = []
    for cls in CLASSES:  # one per diagnosis first, so all 7 classes are visible
        sub = test[test["dx"] == cls]
        if len(sub):
            picks.append(sub.iloc[int(rng.integers(len(sub)))])
    while len(picks) < n:
        picks.append(test.iloc[int(rng.integers(len(test)))])
    return [
        {
            "image_id": r["image_id"],
            "truth": r["dx"],
            "truth_name": CLASS_NAMES[r["dx"]],
            "age": float(r["age"]),
            "sex": r["sex"],
            "localization": r["localization"],
        }
        for r in (p.to_dict() for p in picks[:n])
    ]


def image_bytes(image_id: str) -> bytes:
    from fedderm.data.metadata import _resolve_image_path

    return Path(_resolve_image_path(image_id)).read_bytes()
