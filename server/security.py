"""Security controls for the deployment prototype.

Threat model for a hospital-facing inference endpoint:

  T1 unauthenticated access          -> API-key auth (constant-time compare)
  T2 credential replay in bulk       -> per-key sliding-window rate limiting
  T3 patient data at rest on the box -> images are never written to disk; the
                                        upload is consumed from memory only
  T4 oversized / malicious uploads   -> size + content-type + decode limits
  T5 silent tampering / no forensics -> append-only audit log storing a SHA-256
                                        of the image, never the image itself
  T6 traffic interception            -> TLS (see scripts/make_certs.py)
  T7 model inversion / memorization  -> the served weights were trained with
                                        DP; epsilon is reported next to every
                                        prediction so clinicians see the bound

Deliberately *not* claimed: this is a prototype, not a certified medical device,
and the audit log is a local file rather than a tamper-proof append-only store.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import Header, HTTPException, Request, status

ROOT = Path(__file__).resolve().parent.parent
AUDIT_LOG = ROOT / "artifacts" / "audit.log"
MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
RATE_LIMIT_REQUESTS = 60
RATE_LIMIT_WINDOW_S = 60

DEMO_KEY_FILE = ROOT / "artifacts" / "demo_key.txt"


def _load_or_mint_demo_key() -> str:
    """Stable demo key across restarts.

    In a real deployment keys come from a secret manager and are provisioned per
    hospital (set FEDDERM_API_KEYS). For the demo we persist one generated key so
    that restarting the API does not invalidate the key the browser has already
    cached - a regenerated key silently 401s every thumbnail request.
    """
    if os.environ.get("FEDDERM_DEMO_KEY"):
        return os.environ["FEDDERM_DEMO_KEY"]
    if DEMO_KEY_FILE.exists():
        stored = DEMO_KEY_FILE.read_text(encoding="utf-8").strip()
        if stored:
            return stored
    key = "demo-" + secrets.token_urlsafe(12)
    DEMO_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    DEMO_KEY_FILE.write_text(key, encoding="utf-8")
    return key


_DEMO_KEY = _load_or_mint_demo_key()


def configured_keys() -> dict[str, str]:
    raw = os.environ.get("FEDDERM_API_KEYS", "").strip()
    if raw:
        return {k.strip(): f"client-{i}" for i, k in enumerate(raw.split(",")) if k.strip()}
    return {_DEMO_KEY: "demo"}


_KEYS = configured_keys()
_hits: dict[str, deque[float]] = defaultdict(deque)


def demo_key() -> str:
    return _DEMO_KEY


def _check_rate(identity: str) -> None:
    now = time.time()
    q = _hits[identity]
    while q and now - q[0] > RATE_LIMIT_WINDOW_S:
        q.popleft()
    if len(q) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"rate limit: {RATE_LIMIT_REQUESTS} requests / {RATE_LIMIT_WINDOW_S}s",
        )
    q.append(now)


async def require_api_key(
    request: Request, x_api_key: str | None = Header(default=None)
) -> str:
    """FastAPI dependency: authenticate + rate limit. Returns the caller id."""
    if not x_api_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="missing X-API-Key header")
    for key, name in _KEYS.items():
        if hmac.compare_digest(key, x_api_key):  # constant time: no key oracle
            _check_rate(name)
            return name
    audit("auth.denied", {"path": request.url.path, "ip": request.client.host if request.client else "?"})
    raise HTTPException(status.HTTP_403_FORBIDDEN, detail="invalid API key")


def validate_upload(content_type: str | None, payload: bytes) -> None:
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"content-type must be one of {sorted(ALLOWED_CONTENT_TYPES)}",
        )
    if not payload:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="empty upload")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413,
            detail=f"upload exceeds {MAX_UPLOAD_BYTES} bytes",
        )


def content_digest(payload: bytes) -> str:
    """Pseudonymous handle for an image: lets an auditor prove which bytes were
    scored without the server ever retaining the image itself."""
    return hashlib.sha256(payload).hexdigest()


def audit(event: str, detail: dict | None = None) -> dict:
    record = {"ts": time.time(), "event": event, "detail": detail or {}}
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return record


def read_audit(limit: int = 50) -> list[dict]:
    if not AUDIT_LOG.exists():
        return []
    lines = AUDIT_LOG.read_text(encoding="utf-8").strip().splitlines()[-limit:]
    return [json.loads(x) for x in lines]


SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",  # never cache diagnostic responses
    "Content-Security-Policy": "default-src 'self'; img-src 'self' data: blob:",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
}


def controls_summary() -> list[dict]:
    """Machine-readable list of active controls, rendered by the dashboard."""
    return [
        {"id": "T1", "control": "API-key authentication", "detail": "X-API-Key, constant-time compare", "active": True},
        {"id": "T2", "control": "Rate limiting", "detail": f"{RATE_LIMIT_REQUESTS} req / {RATE_LIMIT_WINDOW_S}s per key", "active": True},
        {"id": "T3", "control": "No patient data at rest", "detail": "images processed in memory, never persisted", "active": True},
        {"id": "T4", "control": "Upload validation", "detail": f"<= {MAX_UPLOAD_BYTES // (1024*1024)} MB, jpeg/png/webp only", "active": True},
        {"id": "T5", "control": "Audit trail", "detail": "append-only log of SHA-256 digests, no images", "active": True},
        {"id": "T6", "control": "TLS transport", "detail": "run with --ssl-certfile (scripts/make_certs.py)", "active": bool(os.environ.get("FEDDERM_TLS"))},
        {"id": "T7", "control": "DP-trained weights", "detail": "epsilon reported with every prediction", "active": True},
        {"id": "T8", "control": "Secure aggregation", "detail": "server never sees individual hospital updates", "active": True},
    ]
