"""FedDerm serving API - the secure deployment prototype.

Endpoints
  GET  /api/health               liveness + which checkpoint is being served
  GET  /api/security             active security controls (T1..T8)
  GET  /api/experiments          centralized vs federated vs DP-federated results
  GET  /api/federation           hospital partition (sizes, label mix, skew)
  GET  /api/privacy/curve        epsilon as a function of rounds / noise
  GET  /api/samples              held-out test images offered as demo cases
  GET  /api/samples/{id}/image   the raw JPEG for a demo case
  POST /api/predict              upload a lesion image -> diagnosis + triage
  POST /api/secagg/demo          run one live secure-aggregation round
  GET  /api/audit                recent audit records (digests only)

Run:
  uvicorn server.app:app --port 8000
  uvicorn server.app:app --port 8443 --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from fedderm import CLASS_NAMES, CLASSES
from fedderm.config import RESULTS
from fedderm.privacy import accountant
from fedderm.privacy.secure_agg import Quantizer, masked_stats, simulate_round

from .inference import Diagnoser, image_bytes, sample_images
from .security import (
    SECURITY_HEADERS,
    audit,
    content_digest,
    controls_summary,
    demo_key,
    read_audit,
    require_api_key,
    validate_upload,
)

app = FastAPI(
    title="FedDerm API",
    version="1.0.0",
    description="Privacy-Preserving Federated Skin-Lesion Diagnosis (HAM10000)",
)

# Loopback-only origin allow-list. A regex rather than a fixed list because dev
# servers and IDE browser previews pick arbitrary high ports; it still refuses
# any non-loopback origin. In production this must be narrowed to the single
# origin the SPA is served from.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

diagnoser = Diagnoser()
_startup: dict = {}


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    for k, v in SECURITY_HEADERS.items():
        response.headers[k] = v
    return response


@asynccontextmanager
async def lifespan(_: FastAPI):
    src = diagnoser.load_best()
    _startup["model_source"] = src
    _startup["started_at"] = time.time()
    audit("server.start", {"model_source": src})
    print(f"[api] serving checkpoint: {src}")
    print(f"[api] demo API key: {demo_key()}")
    yield


app.router.lifespan_context = lifespan


# --------------------------------------------------------------------------- #
# Results / dashboard data
# --------------------------------------------------------------------------- #
def _load_results() -> dict:
    out = {}
    for path in sorted(RESULTS.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        # skip non-run artefacts (comparison.json, privacy_sweep.json)
        if isinstance(payload, dict) and "test" in payload:
            out[path.stem] = payload
    return out


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_source": diagnoser.source,
        "checkpoint": str(diagnoser.checkpoint) if diagnoser.checkpoint else None,
        "image_size": diagnoser.image_size,
        "classes": [{"code": c, "name": CLASS_NAMES[c]} for c in CLASSES],
        "uptime_s": time.time() - _startup.get("started_at", time.time()),
    }


@app.get("/api/security")
def security() -> dict:
    return {"controls": controls_summary(), "demo_api_key": demo_key()}


@app.get("/api/experiments")
def experiments() -> dict:
    """Everything needed for the centralized-vs-federated comparison view."""
    res = _load_results()
    rows = []
    for name, r in res.items():
        test, priv = r.get("test", {}), r.get("privacy", {}) or {}
        rows.append(
            {
                "run": name,
                "mode": r.get("mode", name),
                "accuracy": test.get("accuracy"),
                "balanced_accuracy": test.get("balanced_accuracy"),
                "macro_f1": test.get("macro_f1"),
                "macro_auc": test.get("macro_auc"),
                "malignant_sensitivity": test.get("malignant_sensitivity"),
                "malignant_specificity": test.get("malignant_specificity"),
                "epsilon": priv.get("epsilon"),
                "delta": priv.get("delta"),
                "dp_mode": priv.get("mode"),
                "secure_aggregation": priv.get("secure_aggregation"),
                "wall_time_s": r.get("wall_time_s"),
                "num_params": r.get("num_params"),
                "data_left_premises": str(r.get("mode", name)).startswith("centralized"),
                "communication": r.get("communication"),
                "per_class_recall": test.get("per_class_recall"),
                "confusion_matrix": test.get("confusion_matrix"),
                "curve": [
                    {
                        "x": h.get("round", h.get("epoch")),
                        "balanced_accuracy": h.get("val_balanced_accuracy"),
                        "macro_f1": h.get("val_macro_f1"),
                        "loss": h.get("train_loss"),
                        "epsilon": h.get("epsilon"),
                    }
                    for h in r.get("history", [])
                ],
            }
        )
    return {"runs": rows, "classes": CLASSES, "class_names": CLASS_NAMES}


@app.get("/api/federation")
def federation() -> dict:
    for r in _load_results().values():
        if "federation" not in r:
            continue
        report = dict(r["federation"])
        clients = report.get("clients") or []
        if not clients:
            continue
        sizes = np.asarray([c.get("num_samples", 0) for c in clients], dtype=np.int64)
        counts = np.asarray(
            [[c.get("label_counts", {}).get(label, 0) for label in CLASSES] for c in clients],
            dtype=np.float64,
        )
        local_mix = counts / np.clip(counts.sum(axis=1, keepdims=True), 1, None)
        global_mix = counts.sum(axis=0) / max(float(counts.sum()), 1.0)
        mean_tv = float((0.5 * np.abs(local_mix - global_mix).sum(axis=1)).mean())
        report.update(
            num_clients=len(clients),
            size_min=int(sizes.min()),
            size_max=int(sizes.max()),
            mean_total_variation=mean_tv,
        )
        return report
    raise HTTPException(404, "no federated run found - train first")


@app.get("/api/privacy/curve")
def privacy_curve(
    rounds: int = 40,
    sampling_rate: float = 0.5,
    delta: float = 1e-5,
    noise_multipliers: str = "0.4,0.6,0.8,1.2",
) -> dict:
    """epsilon(round) for several noise levels: the privacy/utility knob."""
    sigmas = [float(s) for s in noise_multipliers.split(",")]
    series = []
    for s in sigmas:
        one_step_rdp = accountant.rdp_sampled_gaussian(sampling_rate, s)
        pts = []
        for r in range(1, rounds + 1):
            eps, order = accountant.rdp_to_dp(one_step_rdp * r, delta)
            pts.append({"round": r, "epsilon": eps, "order": order})
        series.append({"noise_multiplier": s, "points": pts})
    return {"delta": delta, "sampling_rate": sampling_rate, "series": series}


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
@app.get("/api/privacy/sweep")
def privacy_sweep() -> dict:
    """The measured privacy/utility frontier, plus why it looks the way it does."""
    path = RESULTS / "privacy_sweep.json"
    sweep = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    runs = _load_results()
    params = next((r.get("num_params") for r in runs.values() if r.get("num_params")), 81527)
    return {
        "sweep": sweep,
        "scaling": {
            "num_params": params,
            "sqrt_d": params**0.5,
            "explanation": (
                "DP noise added to a gradient has norm ~ sigma*C*sqrt(d)/B while the "
                "clipped signal is bounded by C, so the noise-to-signal ratio is "
                "sigma*sqrt(d)/B for record-level DP-SGD and sigma*sqrt(d)/m for "
                "client-level DP-FedAvg with m participants per round. With d="
                f"{params:,} (sqrt(d)={params ** 0.5:.0f}), record-level DP is workable "
                "at batch 128 but client-level DP is not viable with only 10 hospitals: "
                "it needs the hundreds-of-participants cross-device regime."
            ),
            "record_level_ratio_at_b128": params**0.5 / 128,
            "client_level_ratio_at_m5": params**0.5 / 5,
        },
    }


@app.get("/api/samples")
def samples(n: int = 8, caller: str = Depends(require_api_key)) -> dict:
    return {"samples": sample_images(n)}


@app.get("/api/samples/{image_id}/image")
def sample_image(image_id: str, caller: str = Depends(require_api_key)) -> Response:
    if not image_id.startswith("ISIC_") or not image_id[5:].isdigit():
        raise HTTPException(400, "malformed image id")  # blocks path traversal
    try:
        return Response(image_bytes(image_id), media_type="image/jpeg")
    except FileNotFoundError:
        raise HTTPException(404, "unknown image id")


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    caller: str = Depends(require_api_key),
) -> dict:
    payload = await file.read()
    validate_upload(file.content_type, payload)
    digest = content_digest(payload)
    t0 = time.perf_counter()
    try:
        result = diagnoser.predict(payload)
    except Exception as exc:  # malformed/adversarial image bytes
        audit("predict.error", {"caller": caller, "digest": digest, "error": type(exc).__name__})
        raise HTTPException(400, f"could not decode image: {type(exc).__name__}")
    latency_ms = (time.perf_counter() - t0) * 1000
    # NOTE: `payload` is never written to disk; only its digest is retained.
    audit("predict.ok", {
        "caller": caller,
        "digest": digest,
        "predicted": result["predicted_class"],
        "latency_ms": round(latency_ms, 2),
    })
    priv = _privacy_of_served_model()
    return {
        **result,
        "latency_ms": round(latency_ms, 2),
        "image_sha256": digest,
        "privacy": priv,
        "disclaimer": (
            "Research prototype trained on HAM10000. Not a medical device; "
            "not for clinical decision making."
        ),
    }


def _privacy_of_served_model() -> dict:
    res = _load_results().get(
        "federated_dp" if diagnoser.source == "federated_dp" else diagnoser.source, {}
    )
    p = res.get("privacy") or {}
    return {
        "trained_with_dp": p.get("epsilon") is not None,
        "epsilon": p.get("epsilon"),
        "delta": p.get("delta"),
        "dp_mode": p.get("mode"),
        "secure_aggregation": p.get("secure_aggregation"),
        "model_source": diagnoser.source,
    }


# --------------------------------------------------------------------------- #
# Live secure-aggregation demo
# --------------------------------------------------------------------------- #
class SecAggRequest(BaseModel):
    num_clients: int = Field(4, ge=2, le=12)
    dimension: int = Field(2000, ge=8, le=200_000)
    round_no: int = Field(1, ge=0)
    noise_multiplier: float = Field(0.0, ge=0.0, le=8.0)
    l2_clip: float = Field(1.0, gt=0.0, le=10.0)


@app.post("/api/secagg/demo")
def secagg_demo(req: SecAggRequest, caller: str = Depends(require_api_key)) -> dict:
    """Run one real masked aggregation round and show what the server sees.

    Each simulated hospital gets a distinctive update (a constant offset), so it
    is obvious that (a) the individual masked payloads look like uniform noise
    and (b) the recovered aggregate matches the true sum to quantization error.
    """
    rng = np.random.default_rng(req.round_no)
    updates = {
        cid: np.full(req.dimension, 0.01 * (cid + 1)) + rng.normal(0, 0.002, req.dimension)
        for cid in range(req.num_clients)
    }
    weights = {cid: 1.0 / req.num_clients for cid in updates}
    t0 = time.perf_counter()
    out = simulate_round(updates, round_no=req.round_no, weights=weights)
    elapsed = (time.perf_counter() - t0) * 1000

    true_mean = sum(weights[c] * u for c, u in updates.items())
    dp_noise = (
        rng.normal(0, req.noise_multiplier * req.l2_clip / req.num_clients, req.dimension)
        if req.noise_multiplier > 0
        else np.zeros(req.dimension)
    )
    q = Quantizer()
    audit("secagg.demo", {"caller": caller, "clients": req.num_clients, "dim": req.dimension})
    return {
        "num_clients": req.num_clients,
        "dimension": req.dimension,
        "elapsed_ms": round(elapsed, 2),
        "clients": [
            {
                "client_id": cid,
                "true_update_mean": float(updates[cid].mean()),
                "true_sample": [float(v) for v in updates[cid][:8]],
                "masked_seen_by_server": masked_stats(out["payloads"][cid], q),
            }
            for cid in sorted(updates)
        ],
        "aggregate": {
            "recovered_mean": float(out["aggregate"].mean()),
            "true_mean": float(true_mean.mean()),
            "max_abs_error": float(np.max(np.abs(out["aggregate"] - true_mean))),
            "recovered_sample": [float(v) for v in out["aggregate"][:8]],
            "with_dp_noise_sample": [float(v) for v in (out["aggregate"] + dp_noise)[:8]],
        },
        "explanation": {
            "masking": "pairwise X25519 ECDH -> HKDF -> PRG masks that cancel in the sum",
            "integrity": "each payload is Ed25519-signed; tampering aborts the round",
            "domain": "fixed-point Z_M arithmetic (M = 2^62) so cancellation is exact",
        },
    }


@app.get("/api/audit")
def audit_tail(limit: int = 50, caller: str = Depends(require_api_key)) -> dict:
    return {"records": read_audit(limit)}


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "path": request.url.path},
        headers=SECURITY_HEADERS,
    )
