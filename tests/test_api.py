"""End-to-end tests for the serving prototype (auth, validation, inference)."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from server.app import app
from server.security import demo_key

KEY = {"X-API-Key": demo_key()}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _jpeg(size=(600, 450), colour=(180, 120, 110)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format="JPEG")
    return buf.getvalue()


def test_health_is_public(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert len(body["classes"]) == 7


def test_security_headers_present(client):
    r = client.get("/api/health")
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["Cache-Control"] == "no-store"


def test_cors_allows_any_loopback_port(client):
    """IDE browser previews and dev servers bind arbitrary high ports, so the
    allow-list is a loopback regex rather than a fixed set."""
    for origin in (
        "http://localhost:5173",
        "http://127.0.0.1:64757",
        "http://localhost:4173",
    ):
        r = client.options(
            "/api/privacy/sweep",
            headers={"Origin": origin, "Access-Control-Request-Method": "GET"},
        )
        assert r.status_code == 200, origin
        assert r.headers["access-control-allow-origin"] == origin


def test_cors_rejects_non_loopback_origin(client):
    r = client.get("/api/health", headers={"Origin": "http://evil.example.com"})
    assert "access-control-allow-origin" not in r.headers


def test_results_artefacts_are_not_treated_as_runs(client):
    """results/ also holds comparison.json and privacy_sweep.json; those must
    never show up as experiment rows."""
    names = {row["run"] for row in client.get("/api/experiments").json()["runs"]}
    assert "privacy_sweep" not in names
    assert "comparison" not in names


def test_experiment_data_movement_matches_regime(client):
    rows = {row["run"]: row for row in client.get("/api/experiments").json()["runs"]}
    assert rows["centralized_b0_288_ft120_w085_b16_tuned_v5"]["data_left_premises"] is True
    assert rows["fedavg_ft20_balanced_e3_tuned_v6"]["data_left_premises"] is False


@pytest.mark.parametrize(
    "run",
    [
        "centralized_b0_288_ft120_w085_b16_tuned_v5",
        "fedavg_ft20_balanced_e3_tuned_v6",
        "federated_dp_example",
        "federated_dp_client_tpu",
    ],
)
def test_featured_benchmark_metrics_match_confusion_matrix(client, run):
    rows = {row["run"]: row for row in client.get("/api/experiments").json()["runs"]}
    row = rows[run]
    cm = row["confusion_matrix"]
    total = sum(sum(values) for values in cm)
    true_positive = [cm[i][i] for i in range(len(cm))]
    recalls = [true_positive[i] / sum(cm[i]) for i in range(len(cm))]
    f1 = []
    for i in range(len(cm)):
        predicted = sum(values[i] for values in cm)
        denominator = sum(cm[i]) + predicted
        f1.append(2 * true_positive[i] / denominator if denominator else 0.0)
    malignant = {0, 1, 4}
    malignant_total = sum(sum(cm[i]) for i in malignant)
    malignant_true_positive = sum(cm[i][j] for i in malignant for j in malignant)

    assert row["accuracy"] == pytest.approx(sum(true_positive) / total)
    assert row["balanced_accuracy"] == pytest.approx(sum(recalls) / len(recalls))
    assert row["macro_f1"] == pytest.approx(sum(f1) / len(f1))
    assert row["malignant_sensitivity"] == pytest.approx(
        malignant_true_positive / malignant_total
    )
    for i, label in enumerate(("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")):
        assert row["per_class_recall"][label] == pytest.approx(recalls[i])


def test_featured_privacy_values_are_consistent(client):
    rows = {row["run"]: row for row in client.get("/api/experiments").json()["runs"]}
    assert rows["centralized_b0_288_ft120_w085_b16_tuned_v5"]["epsilon"] is None
    assert rows["fedavg_ft20_balanced_e3_tuned_v6"]["epsilon"] is None
    assert rows["federated_dp_example"]["epsilon"] == pytest.approx(1620.4107819522085)
    assert rows["federated_dp_client_tpu"]["epsilon"] == pytest.approx(23.408034292281812)


def test_federation_summary_is_complete(client):
    r = client.get("/api/federation")
    assert r.status_code == 200
    body = r.json()
    sizes = [site["num_samples"] for site in body["clients"]]
    assert body["num_clients"] == len(body["clients"]) == 10
    assert body["size_min"] == min(sizes)
    assert body["size_max"] == max(sizes)
    assert 0.0 <= body["mean_total_variation"] <= 1.0


def test_predict_requires_api_key(client):
    r = client.post("/api/predict", files={"file": ("a.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 401


def test_predict_rejects_bad_key(client):
    r = client.post(
        "/api/predict",
        files={"file": ("a.jpg", _jpeg(), "image/jpeg")},
        headers={"X-API-Key": "not-the-key"},
    )
    assert r.status_code == 403


def test_predict_returns_full_distribution(client):
    r = client.post(
        "/api/predict", files={"file": ("a.jpg", _jpeg(), "image/jpeg")}, headers=KEY
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["probabilities"]) == 7
    assert abs(sum(p["probability"] for p in body["probabilities"]) - 1.0) < 1e-4
    assert body["triage"]["level"] in {"routine", "review", "urgent"}
    assert len(body["image_sha256"]) == 64
    assert "not a medical device" in body["disclaimer"].lower()


def test_predict_rejects_wrong_content_type(client):
    r = client.post(
        "/api/predict", files={"file": ("a.txt", b"hello", "text/plain")}, headers=KEY
    )
    assert r.status_code == 415


def test_predict_rejects_undecodable_image(client):
    r = client.post(
        "/api/predict", files={"file": ("a.jpg", b"not-a-jpeg", "image/jpeg")}, headers=KEY
    )
    assert r.status_code == 400


def test_predict_rejects_oversized_upload(client):
    big = _jpeg((5000, 5000)) + b"\x00" * (9 * 1024 * 1024)
    r = client.post("/api/predict", files={"file": ("a.jpg", big, "image/jpeg")}, headers=KEY)
    assert r.status_code == 413


def test_sample_image_id_is_validated(client):
    r = client.get("/api/samples/..%2F..%2Fetc%2Fpasswd/image", headers=KEY)
    assert r.status_code in (400, 404)


def test_secagg_demo_recovers_the_sum(client):
    r = client.post(
        "/api/secagg/demo",
        json={"num_clients": 4, "dimension": 500, "round_no": 3, "noise_multiplier": 0.0,
              "l2_clip": 1.0},
        headers=KEY,
    )
    assert r.status_code == 200
    body = r.json()
    agg = body["aggregate"]
    assert agg["max_abs_error"] < 1e-4
    assert abs(agg["recovered_mean"] - agg["true_mean"]) < 1e-4
    # the masked view of any single client must not resemble its true update
    for c in body["clients"]:
        assert abs(c["masked_seen_by_server"]["mean"] - c["true_update_mean"]) > 1.0


def test_secagg_rejects_absurd_dimension(client):
    r = client.post(
        "/api/secagg/demo",
        json={"num_clients": 4, "dimension": 10_000_000, "round_no": 1,
              "noise_multiplier": 0.0, "l2_clip": 1.0},
        headers=KEY,
    )
    assert r.status_code == 422


def test_privacy_curve_is_monotone(client):
    r = client.get("/api/privacy/curve?rounds=10&noise_multipliers=0.5,1.0")
    assert r.status_code == 200
    for series in r.json()["series"]:
        eps = [p["epsilon"] for p in series["points"]]
        assert all(a <= b for a, b in zip(eps, eps[1:]))
    lo, hi = r.json()["series"]
    assert lo["points"][-1]["epsilon"] > hi["points"][-1]["epsilon"]


def test_privacy_curve_computes_rdp_once_per_noise_level(client, monkeypatch):
    from fedderm.privacy import accountant

    calls = 0
    original = accountant.rdp_sampled_gaussian

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(accountant, "rdp_sampled_gaussian", counted)
    r = client.get("/api/privacy/curve?rounds=3&noise_multipliers=0.5,1.0")
    assert r.status_code == 200
    assert calls == 2


def test_privacy_curve_matches_accountant(client):
    from fedderm.privacy.accountant import compute_epsilon

    r = client.get("/api/privacy/curve?rounds=3&noise_multipliers=0.8")
    assert r.status_code == 200
    for point in r.json()["series"][0]["points"]:
        epsilon, order = compute_epsilon(0.5, 0.8, point["round"], 1e-5)
        assert point["epsilon"] == pytest.approx(epsilon)
        assert point["order"] == pytest.approx(order)


def test_audit_log_records_predictions(client):
    client.post("/api/predict", files={"file": ("a.jpg", _jpeg(), "image/jpeg")}, headers=KEY)
    r = client.get("/api/audit?limit=10", headers=KEY)
    assert r.status_code == 200
    events = [x["event"] for x in r.json()["records"]]
    assert "predict.ok" in events
    # the log must never contain image bytes, only digests
    assert all("image" not in x["detail"] for x in r.json()["records"])
