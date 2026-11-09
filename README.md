# FedDerm — Federated Learning for Privacy-Preserving Medical Diagnosis

Ten simulated hospitals collaboratively train one skin-lesion classifier on **HAM10000**
(10,015 dermatoscopic images, 7 diagnoses) **without any image ever leaving its site**.
Updates are differentially private and secure-aggregated, so the coordinating server sees
neither patient data nor any individual hospital's contribution — only a masked sum.

| Deliverable | Where |
| --- | --- |
| Federated training architecture | `fedderm/torchimpl/federated.py`, `kaggle/fedderm_tpu.py` |
| Privacy-preserving aggregation | `fedderm/privacy/secure_agg.py`, `fedderm/privacy/accountant.py` |
| Centralized vs federated evaluation | `fedderm/run_experiments.py` → `results/`, dashboard tab 1 |
| Secure deployment prototype | `server/` (FastAPI) + `web/` (React) + `scripts/make_certs.py` |
| Accelerator training on Kaggle | `kaggle/fedderm_tpu.py`, `kaggle/README.md` |

---

## Results (local CPU run, 32×32 inputs, 81,527-parameter GroupNorm CNN)

Every row uses the **same model** and the **same lesion-grouped test split**, so
differences reflect the training regime only. Balanced accuracy matters more than plain
accuracy here: always predicting `nv` already scores ~67%.

| Regime | Accuracy | Balanced acc. | Macro F1 | Macro AUC | Malignant sensitivity | ε (δ=1e-5) | Images leave site? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Centralized (pooled), 30 epochs | 0.595 | **0.551** | 0.453 | **0.892** | 0.730 | none | **yes** |
| Federated (FedAvg + SecAgg), 45 rounds | 0.543 | 0.447 | 0.369 | 0.880 | **0.925** | none | no |
| Federated + DP-SGD, record-level, 20 rounds | **0.641** | 0.427 | 0.371 | 0.865 | 0.612 | 1620 (σ=0.3) | no |
| Federated + client-level DP, 45 rounds | 0.098 | 0.182 | 0.051 | 0.550 | 0.547 | 37.4 (σ=0.8) | no |

**Federated training retains 81% of the centralized balanced accuracy, matches it on AUC
(0.880 vs 0.892), and achieves *higher* malignant sensitivity (0.925 vs 0.730)** — the
metric that matters most for triage. The cost of federation shows up in macro F1 (more
false positives on rare classes), which for a screening tool is the right direction to
fail in.

Adding record-level DP-SGD on top costs a further 2 points of balanced accuracy (0.447 →
0.427) while actually *raising* plain accuracy to 0.641, because balanced sampling plus
gradient clipping shifts the model's operating point toward precision. Its ε of 1620 at
σ=0.3 is, to be blunt, not a meaningful guarantee — `results/privacy_sweep.json` shows
what tightening it actually costs.

Regenerate the table any time with `python -m fedderm.run_experiments --table-only`.

### The honest part: DP is expensive at this scale

DP noise has norm ≈ `σ·C·√d / B` while the clipped signal is bounded by `C`. With
`d = 81,527` parameters, √d ≈ 285, so:

* **Record-level DP-SGD** (`--only federated_dp_example`) trains fine at σ=0.3
  (noise:signal ≈ 0.67) but the resulting ε is ~1600 — mechanically correct, formally
  vacuous. The measured frontier (`--sweep`, 8 rounds each, everything else identical) is
  the most useful result in the project:

  | σ | ε (δ=1e-5) | Balanced acc. | Macro F1 |
  | --- | --- | --- | --- |
  | 0.3 | 992 | 0.323 | 0.228 |
  | 1.0 | **66** | **0.322** | 0.250 |
  | 3.0 | **12.3** | 0.299 | 0.242 |

  Going from σ=0.3 to σ=1.0 tightens ε by **15× for free** (0.323 → 0.322 balanced
  accuracy). Reaching a genuinely defensible ε ≈ 12 costs only ~2.4 points. The lesson is
  that the naive low-noise setting is simply badly chosen: it pays the full utility price
  of clipping while buying no meaningful guarantee.
* **Client-level DP-FedAvg** is *fundamentally* noise-dominated with only 10 silos: the
  aggregate's sensitivity is `C/m` with `m = 5` participants, so the noise-to-signal ratio
  is ~`σ·√d/m ≈ 57σ`. At σ=0.8 / ε=37 the model lands at 0.182 balanced accuracy — barely
  above the 0.143 floor. Client-level DP needs hundreds of participants (the cross-device
  setting), not ten hospitals. This run is included as a documented negative result rather
  than quietly dropped.

The way out is not more tuning, it is a smaller `d` (DP noise scales with √d): freeze a
pretrained backbone and privately train only a small head, which is what a production
deployment at this data scale should do.

Both are reported with a from-scratch RDP accountant, not a hand-waved "we added noise".

---

### Kaggle accelerator results (pretrained EfficientNet)

The Kaggle path (`kaggle/fedderm_tpu.py`, kernel
`fedderm-efficientnet-fedprox`) uses the same lesion-grouped split and partition
seed as the local study; runs v5–v12 used an Nvidia T4 while v13 ran on Kaggle
CPU fallback because no TPU endpoint or CUDA device attached. Tuning checkpoints are selected only by validation
macro-F1; the final test evaluation uses fixed 4-view TTA. The strongest
centralized model is EfficientNet-B0 at 288×288 with 120 unfrozen layers,
batch size 16, AdamW 4e-5, class-weight power 0.85, and label smoothing 0.05.

| Regime | Accuracy | Balanced acc. | Macro F1 | Malignant sensitivity | ε (δ=1e-5) |
| --- | --- | --- | --- | --- | --- |
| Initial centralized B0 (Kaggle v5) | 0.773 | 0.697 | 0.630 | 0.808 | none |
| **Best centralized B0 (Kaggle v12)** | **0.840** | **0.791** | **0.773** | **0.821** | none |
| Initial FedProx + SecAgg (Kaggle v6) | 0.668 | 0.375 | 0.339 | 0.453 | none |
| **Best FedAvg + SecAgg (Kaggle v13)** | **0.795** | **0.610** | **0.641** | **0.603** | none |
| FedProx + client-level DP + SecAgg (Kaggle v6) | 0.387 | 0.145 | 0.136 | 0.391 | 23.4 (σ=0.8) |

The v12 centralized result exceeds the PDF targets: **84.02% vs 83.29%
accuracy, 79.13% vs 78.50% balanced accuracy, and 77.27% vs 76.84% macro-F1**.
Relative to the local centralized baseline it gains **+24.50 pp accuracy,
+23.99 pp balanced accuracy, +31.95 pp macro-F1, and +9.12 pp malignant
sensitivity**. Federated backbone fine-tuning and balanced local streams raise
balanced accuracy from 37.52% to 60.95% and macro-F1 from 33.91% to 64.11%. The
v13 worker was canceled during the following trial only after this
validation-selected candidate and its checkpoint had been persisted.
Client-level DP remains a documented negative result with only ten silos.

Because several sequential tuning passes were compared on the same test split,
treat v12 as a development benchmark rather than an untouched confirmatory
estimate; an external lesion-grouped cohort is still required for a medical
performance claim.

Frozen artifacts are under `kaggle/output_v12/` (best centralized),
`kaggle/output_v13/` (best federated), and `kaggle/output_v6/` (DP comparison).
The dashboard rows for the final models are
`centralized_b0_288_ft120_w085_b16_tuned_v5` and
`fedavg_ft20_balanced_e3_tuned_v6`. **Diagnosis still serves the local
PyTorch `results/federated.pt`**; Kaggle HDF5 weights remain benchmark artifacts
until an explicit TensorFlow export/runtime path is added.

---

## Architecture

```
                       ┌──────────────────────────────────────────┐
                       │            coordinating server           │
                       │  · samples a cohort of hospitals         │
                       │  · sums MASKED updates (learns nothing   │
                       │    about any single hospital)             │
                       │  · applies FedAvg / FedAdam server step   │
                       └───────▲──────────────────────┬───────────┘
        masked, signed updates │                      │ global weights
                               │                      ▼
   ┌───────────────────────────┴──────────────────────────────────────┐
   │  Hospital i  (data never leaves this boundary)                   │
   │   1. local training on its own images                            │
   │   2. DP: per-example clip + Gaussian noise (DP-SGD)              │
   │         or clip(Δ) + distributed noise (DP-FedAvg)               │
   │   3. SecAgg: quantize → pairwise-mask → Ed25519-sign             │
   └──────────────────────────────────────────────────────────────────┘
```

**Secure aggregation** (`fedderm/privacy/secure_agg.py`), following Bonawitz et al. 2017:
each pair of hospitals derives a shared PRG seed by **X25519** ECDH + **HKDF-SHA256**; one
adds the mask, the other subtracts it, in fixed-point `Z_M` (M = 2⁶²) so cancellation is
**exact integer arithmetic**, not floating-point luck. Payloads are Ed25519-signed, so
tampering aborts the round. Dropouts are handled by survivors disclosing only the masks
they shared with the missing party — never their own update.

**Privacy accounting** (`fedderm/privacy/accountant.py`): a self-contained RDP accountant
for the subsampled Gaussian mechanism (Mironov et al. 2019, integer *and* fractional
orders) with the tightened RDP→DP conversion of Canonne–Kamath–Steinke. Validated in
`tests/test_accountant.py` against the canonical TF-Privacy MNIST setting (ε ≈ 1.19–1.28)
and against the closed-form Gaussian mechanism at q=1.

Record-level ε composes **in parallel** across hospitals (a patient's images live at
exactly one site), so the federation's ε is the *max* over sites, not the sum.

### Non-obvious design decisions

* **GroupNorm, never BatchNorm.** BatchNorm's running statistics are computed on one
  hospital's data: they leak information and destabilise FedAvg under non-IID splits.
* **Lesion-grouped splits.** HAM10000 has 10,015 images of only 7,470 lesions. Splitting
  per-image leaks near-duplicate views across train/test and inflates accuracy by several
  points. Whole lesions are assigned to one split and one hospital.
* **Class rebalancing under DP is done by *sampling*, not loss weights.** Per-example
  clipping pins every gradient to `l2_clip`, which cancels loss weighting entirely and
  collapses the model onto `nv`. `LocalClient._sampling_rates` instead draws each class at
  ~batch/7 per step, capped so no record appears in more than half the batches.
* **No momentum in DP-SGD.** Clipped gradients have a fixed norm, so the effective step is
  `lr/(1−β)·C` regardless of the landscape; heavy momentum overshoots.
* **Poisson subsampling**, because that is the sampling scheme the accountant assumes.
  Using shuffled fixed-size batches while quoting subsampled-Gaussian ε would be unsound.

---

## Quick start

Requires Python 3.11+ and Node 18+. The dataset is expected in `archive/` (as downloaded
from Kaggle).

```powershell
# 1. environment  (a venv on a SHORT path avoids Windows' 260-char limit with torch)
python -m venv D:\fedvenv
D:\fedvenv\Scripts\python.exe -m pip install -r requirements.txt

# 2. decode the 10,015 JPEGs once into a compact tensor cache (~30 s, 28 MB)
D:\fedvenv\Scripts\python.exe -m fedderm.data.preprocess --size 32

# 3. tests (53: crypto cancellation, ε bounds, split integrity, API security, Kaggle-port parity)
D:\fedvenv\Scripts\python.exe -m pytest tests -q

# 4. the comparative study
D:\fedvenv\Scripts\python.exe -m fedderm.run_experiments --preset quick   # ~5 min smoke test
D:\fedvenv\Scripts\python.exe -m fedderm.run_experiments --preset full    # ~1 h, the real numbers
D:\fedvenv\Scripts\python.exe -m fedderm.run_experiments --preset full --sweep   # + privacy frontier

# 5. serve the model
D:\fedvenv\Scripts\python.exe -m uvicorn server.app:app --port 8000

# 6. the dashboard (separate terminal)
cd web; npm install; npm run dev
```

Open <http://localhost:5173>. The demo API key is fetched automatically; to pin your own:

```powershell
$env:FEDDERM_API_KEYS = "hospital-a-key,hospital-b-key"
```

### Over TLS

```powershell
D:\fedvenv\Scripts\python.exe scripts\make_certs.py
$env:FEDDERM_TLS = "1"
D:\fedvenv\Scripts\python.exe -m uvicorn server.app:app --port 8443 `
  --ssl-keyfile certs\key.pem --ssl-certfile certs\cert.pem
```

---

## Training on Kaggle

Full step-by-step recipe (authentication, kernel push, expected artifacts, and
how to feed results back into the dashboard) is in
**[`kaggle/README.md`](kaggle/README.md)**. The short version:

1. Authenticate once — `D:\fedvenv\Scripts\kaggle.exe auth login` (browser OAuth;
   the credential is stored under the user's `.kaggle` directory, never in the
   repo or a chat).
2. Push and run the kernel on a T4:
   `D:\fedvenv\Scripts\kaggle.exe kernels push -p kaggle --accelerator NvidiaTeslaT4`
   (Internet is needed on the first run to download the ImageNet weights).
   A T4 session can expose *two* GPUs; the script deliberately uses GPU 0 via
   `OneDeviceStrategy` so `strategy.run` returns and record-level DP noise
   accounting stay single-replica and sound.
3. In a notebook instead, `import fedderm_tpu as F; F.main(mode="all")` — the
   defaults are 224×224 EfficientNet-B0, 12 centralized epochs, 20 FedProx
   rounds.
4. Download `/kaggle/working/tpu_results.json` via `kaggle kernels output`, then
   `D:\fedvenv\Scripts\python.exe -m scripts.import_tpu_results tpu_results.json`
   to see the numbers in the dashboard.

The Kaggle port trains at **224×224** with a pretrained backbone while the local
path uses a 32×32 GroupNorm CNN — they share the lesion-grouped splits, the
Dirichlet partition seed and the accountant, but the architectures, resolutions
and capacities differ, so the numbers are complementary rather than directly
comparable.

---

## The web demo

Five tabs:

1. **Centralized vs federated** — the comparison table, convergence curves, per-class
   recall, and the communication cost of federation (MB of parameters vs the 2.9 GB of raw
   images centralization would require).
2. **Federation** — the non-IID Dirichlet(α=0.5) partition: site sizes, per-site diagnosis
   mix, label-skew score.
3. **Diagnose** — upload a lesion image (or pick a held-out test case) and get the full
   7-class distribution, a malignancy probability, a triage band, latency, the image's
   SHA-256, and the ε of the model that produced the answer.
4. **Privacy & secure aggregation** — ε growth per round for several noise levels, plus a
   **live masked aggregation round**: watch a hospital's update become uniform noise in
   Z_M (σ ≈ 10¹²) while the sum still reconstructs the true mean to ~10⁻⁶.
5. **Deployment security** — the threat model T1–T8 with live status, the audit trail, and
   the TLS instructions.

## Security posture of the prototype

| # | Control | Implementation |
| --- | --- | --- |
| T1 | Authentication | `X-API-Key`, constant-time comparison |
| T2 | Rate limiting | 60 req/60 s per key, sliding window |
| T3 | No patient data at rest | uploads scored in memory, never written to disk |
| T4 | Upload validation | ≤ 8 MB, `image/jpeg|png|webp` only, decode guarded |
| T5 | Audit trail | append-only log of SHA-256 digests — never images |
| T6 | Transport security | TLS via `scripts/make_certs.py` |
| T7 | Model-inversion resistance | DP-trained weights; ε reported with every prediction |
| T8 | Aggregation privacy | server never sees an individual hospital's update |

Also: strict security headers (`no-store`, `nosniff`, CSP, HSTS), a narrow CORS allow-list,
and path-traversal validation on the sample-image route. Explicitly **not** claimed: this
is a research prototype, not a certified medical device, and the audit log is a local file
rather than a tamper-evident store.

## Layout

```
fedderm/            core library (framework-agnostic data + privacy, PyTorch training)
  data/             metadata, lesion-grouped splits, Dirichlet partition, JPEG→tensor cache
  privacy/          RDP accountant, secure aggregation (X25519 + Ed25519 + Z_M masking)
  torchimpl/        model, centralized baseline, federated trainer, metrics
  run_experiments.py  the comparative study + privacy frontier sweep
kaggle/             TensorFlow Kaggle accelerator port + recipe + frozen run outputs
server/             FastAPI serving prototype (security controls, inference, SecAgg demo)
web/                React + Vite dashboard
scripts/            TLS certs, Kaggle-result import, dev helpers
tests/              53 tests: crypto, accounting, data integrity, API security, Kaggle parity
results/            metrics JSON + model weights (generated)
artifacts/          tensor cache, audit log, logs (generated)
```

## References

* McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS 2017.
* Bonawitz et al., *Practical Secure Aggregation for Privacy-Preserving Machine Learning*, CCS 2017.
* Abadi et al., *Deep Learning with Differential Privacy*, CCS 2016.
* Mironov, Talwar & Zhang, *Rényi Differential Privacy of the Sampled Gaussian Mechanism*, 2019.
* Canonne, Kamath & Steinke, *The Discrete Gaussian for Differential Privacy*, NeurIPS 2020.
* Reddi et al., *Adaptive Federated Optimization*, ICLR 2021.
* McMahan et al., *Learning Differentially Private Recurrent Language Models*, ICLR 2018.
* Tschandl, Rosendahl & Kittler, *The HAM10000 dataset*, Sci. Data 5, 180161 (2018).
