"""One-off JPEG -> compact uint8 tensor cache.

The 10015 dermatoscopic JPEGs are 600x450 each (2.9 GB). Decoding them every
epoch dominates runtime on a CPU, so we decode once into a single .npz whose
row order matches `load_metadata()` exactly (sorted by image_id). The TPU job
on Kaggle reads the JPEGs directly through tf.data instead - see kaggle/.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import ARTIFACTS
from .metadata import load_metadata


def _load_one(args: tuple[str, int]) -> np.ndarray:
    path, size = args
    with Image.open(path) as im:
        im = im.convert("RGB")
        # centre-crop to square first: lesions are centred, the 600x450 frame
        # has vignetting/rulers at the edges.
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        im = im.resize((size, size), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def cache_path(size: int) -> Path:
    return ARTIFACTS / f"ham10000_{size}px.npz"


def build_cache(size: int = 32, workers: int | None = None, force: bool = False) -> Path:
    out = cache_path(size)
    if out.exists() and not force:
        print(f"[cache] reusing {out}")
        return out
    df = load_metadata()
    print(f"[cache] decoding {len(df)} images at {size}x{size} ...")
    jobs = [(p, size) for p in df["path"].tolist()]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        images = list(ex.map(_load_one, jobs, chunksize=64))
    x = np.stack(images)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        images=x,
        labels=df["label"].to_numpy(np.int64),
        image_ids=df["image_id"].to_numpy(dtype="U16"),
    )
    print(f"[cache] wrote {out} ({out.stat().st_size / 1e6:.1f} MB, shape {x.shape})")
    return out


def load_cache(size: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = cache_path(size)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run: python -m fedderm.data.preprocess --size {size}")
    z = np.load(p)
    return z["images"], z["labels"], z["image_ids"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    build_cache(a.size, a.workers, a.force)
