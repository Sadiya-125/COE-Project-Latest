# Running FedDerm on Kaggle

`fedderm_tpu.py` is the TensorFlow/Keras port of the federated framework, sized for
**224×224** inputs instead of the 32×32 tensors the local CPU path uses. It runs
centralized training, plain FedAvg/FedProx, and DP-FedAvg / DP-SGD on Kaggle
accelerators, and writes `tpu_results.json` in the same schema the web dashboard
reads.

**What this version closes relative to the problem statement:** the default recipe
is now a pretrained **EfficientNet-B0** backbone (image size 224, only the last
`fine_tune_layers=20` layers unfrozen centrally and the whole backbone frozen in
federation), **label smoothing** 0.05 on a class-weighted cross-entropy, **AdamW**
(lr 2e-4, weight decay 1e-4) for both centralized and client optimizers, best
checkpoint selection by **validation macro-F1**, and **FedProx** (µ = 0.01) as the
default federated algorithm. The original from-scratch GroupNorm CNN is still
available via `--architecture dermcnn`.

---

## Measured runs

Kernel: <https://www.kaggle.com/code/sadiyamaheensiddiqui/fedderm-efficientnet-fedprox>
(v5–v12 on Nvidia T4; v13 ran on Kaggle CPU fallback after no TPU endpoint or
CUDA device attached). Frozen outputs are under `kaggle/output_v12/` for the best
centralized model, `kaggle/output_v13/` for the best federated model, and
`kaggle/output_v6/` for the DP comparison.

| Regime | Accuracy | Balanced acc. | Macro F1 | Malignant sensitivity | ε (δ=1e-5) |
| --- | --- | --- | --- | --- | --- |
| Initial centralized B0 (v5) | 0.773 | 0.697 | 0.630 | 0.808 | none |
| **Best centralized B0, 288px/batch 16 (v12)** | **0.840** | **0.791** | **0.773** | **0.821** | none |
| Initial FedProx + SecAgg (v6) | 0.668 | 0.375 | 0.339 | 0.453 | none |
| **Best FedAvg + SecAgg, 3 local epochs (v13)** | **0.795** | **0.610** | **0.641** | **0.603** | none |
| FedProx + client-level DP + SecAgg (v6) | 0.387 | 0.145 | 0.136 | 0.391 | 23.4 (σ=0.8) |

Honest interpretation:

* The v12 centralized model exceeds the PDF targets: **84.02% vs 83.29%
  accuracy, 79.13% vs 78.50% balanced accuracy, and 77.27% vs 76.84% macro-F1**.
* Balanced streams plus 20-layer federated backbone fine-tuning improve the v6
  non-private federated result by **+12.72 pp accuracy, +23.43 pp balanced
  accuracy, and +30.20 pp macro-F1**.
* The v13 worker was canceled during the trial that followed the winning
  candidate; the winning trial's metrics, per-round history and checkpoint were
  already fully persisted, so the result is complete rather than partial.
* Client-level DP at ten silos remains a documented negative result. No DP
  tuning was used to inflate the score-search results.
* Multiple sequential tuning passes were compared on the same lesion-grouped
  test split. Treat v12 as a development benchmark; use an external cohort for
  a confirmatory medical estimate.

## Choosing the accelerator

* **GPU is the recommended accelerator for API-pushed runs** of this path — the
  EfficientNet backbone trains well on Kaggle's T4 sessions, and every mode
  (centralized, federated, both DP modes) works there. A Kaggle T4 session can
  expose *two* GPUs, but the script deliberately runs everything on GPU 0 via
  `tf.distribute.OneDeviceStrategy` so federated `strategy.run` return values
  and record-level DP noise accounting stay single-replica and sound; GPU 1 is
  intentionally unused.
* **TPU VM v3-8 remains supported** for centralized training, FedAvg/FedProx and
  client-level DP (`dp_mode="client"`).
* **`dp_mode="example"` is deliberately single-device.** Record-level DP-SGD needs
  per-example clipping, and replicated noise accounting across TPU cores is not
  implemented, so the script raises a clear error if it sees more than one
  replica. Run it on one GPU/CPU. With the defaults `local_batch_size=32` and
  `microbatches=32` you get true per-example clipping.

## 1. Authenticate the Kaggle CLI (once)

The CLI is installed in the project venv. Authenticate it yourself — **never paste
API tokens into chat, notebooks, or source files**:

```powershell
D:\fedvenv\Scripts\kaggle.exe auth login
```

This opens the Kaggle OAuth flow in your browser; on success the CLI stores the
credential under `C:\Users\SadiyaMaheenSiddiqui\.kaggle\` on Windows. Nothing else
is needed — subsequent `kaggle` commands pick it up automatically.

## 2. Push the kernel via the API

From `D:\COE Project\kaggle`:

```powershell
# once per kernel: creates kernel-metadata.json (edit id/title after init)
D:\fedvenv\Scripts\kaggle.exe kernels init -p .

# push a new version and let Kaggle run it on a T4
D:\fedvenv\Scripts\kaggle.exe kernels push -p . --accelerator NvidiaTeslaT4

# check progress / fetch outputs
D:\fedvenv\Scripts\kaggle.exe kernels status <your-user>/<kernel-slug>
D:\fedvenv\Scripts\kaggle.exe kernels output <your-user>/<kernel-slug> -p out
```

The kernel should select a **GPU** accelerator (see above). The script falls back
to TPU/CPU automatically where the mode allows it.

## 3. Or run it in a notebook

1. Go to <https://www.kaggle.com/code> → **New Notebook**.
2. **Add Input** (right sidebar) → search `Skin Cancer MNIST: HAM10000` → add
   `kmader/skin-cancer-mnist-ham10000`. It mounts somewhere under
   `/kaggle/input` (the exact layout varies); `discover_data_dir()` finds it.
3. Set **Accelerator → GPU** (or TPU VM v3-8 for centralized / client-level DP).
4. Internet must be **on** the first time so `weights="imagenet"` can download the
   pretrained EfficientNet-B0 weights; afterwards they are cached.

Get the script into the session, either by pasting it into a cell or (cleaner) by
uploading it as a Kaggle Dataset and adding it as an input:

```python
!cp /kaggle/input/fedderm-code/fedderm_tpu.py .
import fedderm_tpu as F
```

## 4. Run it

Full comparative study (centralized + federated + federated with DP), all on the
new defaults — 224×224, 12 centralized epochs, 20 federated rounds, FedProx:

```python
import fedderm_tpu as F
results = F.main(mode="all")
```

Individual pieces, if you want to iterate:

```python
F.main(mode="centralized")                                  # pooled-data upper bound
F.main(mode="federated")                                    # FedProx + secure aggregation
F.main(mode="federated_dp", dp_mode="client",
       noise_multiplier=0.8, l2_clip=1.0)                   # DP-FedProx (hospital-level DP)
F.main(mode="federated_dp", dp_mode="example",
       noise_multiplier=0.3, l2_clip=2.0)                   # DP-SGD — single GPU/CPU only
```

Or from a shell cell:

```bash
!python fedderm_tpu.py --mode all
!python fedderm_tpu.py --mode federated_dp --dp-mode example --rounds 25 --noise-multiplier 0.3
!python fedderm_tpu.py --architecture dermcnn --federated-algorithm fedavg   # legacy recipe
```

Useful overrides: `--architecture {efficientnetb0,efficientnetb1,efficientnetb2,efficientnetv2b0,efficientnetv2b1,dermcnn}`,
`--federated-algorithm {fedavg,fedprox}`, `--fedprox-mu`, `--batch-size`,
`--local-batch-size`, `--local-epochs`, `--lr`, `--local-lr`, plus the existing
`--mode`, `--rounds`, `--epochs`, `--image-size`, `--num-clients`, `--dp-mode`,
`--noise-multiplier`, `--l2-clip`.

### Tuning modes

All tuning modes select checkpoints by **validation macro-F1**, write
`tpu_results.json` incrementally after every trial, and use fixed 4-view TTA
(original, horizontal flip, vertical flip, rot90) only for final test
evaluation. DP is excluded from score search because ten-silo client DP is
already a measured negative result.

* `--mode tune` / `TUNING_PLAN`: balanced FedAvg/FedProx with four local epochs,
  plus B0 fine-tuning depths 40 and 80.
* `--mode tune2` / `TUNING_PLAN_V2`: full-client balanced FedAvg and B0/B1/B2
  capacity trials with class-weight power 0.75.
* `--mode tune3` / `TUNING_PLAN_V3`: 20-layer federated backbone fine-tuning and
  focused B0 class-weight/depth/resolution trials. This produced the best
  federated result.
* `--mode tune4` / `TUNING_PLAN_V4`: B0 at 256/288px plus EfficientNetV2-B0/B1.
* `--mode tune5` / `TUNING_PLAN_V5` (**CLI default**): final 288/320px B0
  refinement over class-weight power, depth, learning rate, and batch size. The
  winning trial is `centralized_b0_288_ft120_w085_b16`.

Explicit `--mode all` still runs the centralized, federated, and DP comparison;
`--balanced-federated-sampling` and `--no-test-time-augmentation` toggle the
corresponding behaviours for ad hoc runs.

## 5. Collect the outputs

Everything lands in `/kaggle/working`:

| File | Contents |
| --- | --- |
| `tpu_results.json` | metrics, per-round history, ε, federation layout, timings |
| `centralized_tpu.weights.h5` | pooled-data model weights |
| `fedprox_tpu.weights.h5` / `fedavg_tpu.weights.h5` | federated model weights (no DP) |
| `dp-fedprox-client_tpu.weights.h5` / `dp-fedprox-example_tpu.weights.h5` | DP model weights |

`tpu_results.json` is rewritten after *each* completed regime, so metrics for
finished runs remain downloadable even if a later regime fails. Model artifacts
are weights-only HDF5 (architecture and config live in code/`config` in the
results JSON); rebuild the model with `build_model()` and `load_weights()` to
reuse them.

Download them from the notebook's **Output** tab, with `!zip -r out.zip
/kaggle/working`, or via `kaggle kernels output`.

## 6. Feed the results back into the dashboard

Copy `tpu_results.json` into the repo's `results/` folder, split per run, so the
API picks it up:

```powershell
D:\fedvenv\Scripts\python.exe scripts\import_tpu_results.py path\to\tpu_results.json
```

---

## Notes, and where the accelerator changes the design

* **Pretrained backbone.** EfficientNet-B0 ships with ImageNet weights; the
  pipeline leaves decoded/augmented images in [0,1] and a `Rescaling(255.0)`
  layer inside the model matches what the backbone expects. Its BatchNorm layers
  are always frozen and it is called with `training=False`, so no running
  statistics are ever computed on a hospital's data. `architecture="dermcnn"`
  keeps the old GroupNorm CNN and the HAM10000 mean/std normalization.
* **Trainable-only aggregation.** In federation only `model.trainable_variables`
  are flattened, averaged, masked by secure aggregation and counted in the
  communication stats — with the backbone frozen that is just the small head, so
  uplink volume drops by orders of magnitude relative to full-model FedAvg.
* **FedProx.** The default `federated_algorithm="fedprox"` adds
  `0.5·µ·Σ‖w − w_global‖²` to each client's local loss, which stabilises local
  training under the non-IID Dirichlet partition. `fedavg` disables the term.
* **Data pipeline.** `tf.data` decodes the JPEGs once and `.cache()`s the decoded
  224×224 tensors in host RAM (~1.5 GB for all 10,015 images); on TPU the
  decoding, not the matmul, is the bottleneck. No TFRecords needed.
* **Why not `tf.function` per client?** Each federated round re-seeds a single
  model living inside the strategy scope rather than building one model per
  hospital. Re-tracing the training graph 10 times per round dominates the
  runtime otherwise.
* **Secure aggregation** runs on the host in NumPy (that is where it belongs:
  inside the client's trust boundary, before anything is transmitted). The
  accelerator only ever sees ordinary training.
* **Reproducibility.** The splits, the Dirichlet partition and the RDP accountant
  are reimplemented with the same logic and the same seed (1337) as the local
  PyTorch package — but the architecture (EfficientNet-B0 vs GroupNorm CNN),
  resolution (224 vs 32) and capacity differ, so Kaggle and CPU numbers are
  complementary rather than directly comparable.
* **DP modes.** Client-level DP (`dp_mode="client"`) distributes Gaussian noise
  across the cohort and runs on any strategy. Record-level DP
  (`dp_mode="example"`) is restricted to a single replica by design — see
  "Choosing the accelerator" above.
