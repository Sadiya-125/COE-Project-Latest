# Project notes for agents

## Environment

* **Use the venv at `D:\fedvenv`**, not the system Python. The system interpreter is
  Python 3.14 on a long path, and installing `torch` into it fails with
  `WinError 206 (path too long)` because Windows long paths are disabled
  (`HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled = 0`, and enabling
  it needs admin). A venv on a short path (`D:\fedvenv`) avoids this.
* TensorFlow has **no Python 3.14 wheels**, so it cannot be installed locally. The TF code
  (`kaggle/fedderm_tpu.py`) runs on Kaggle; locally it is covered by
  `tests/test_tpu_port.py`, which imports it with a stubbed `tensorflow`.
* `starlette.testclient` needs `httpx2` (already installed).

## Commands

```powershell
D:\fedvenv\Scripts\python.exe -m pytest tests -q                       # full test suite
D:\fedvenv\Scripts\python.exe -m fedderm.data.preprocess --size 32      # build tensor cache
D:\fedvenv\Scripts\python.exe -m fedderm.run_experiments --preset quick # ~5 min smoke run
D:\fedvenv\Scripts\python.exe -m fedderm.run_experiments --preset full --sweep
D:\fedvenv\Scripts\python.exe -m fedderm.run_experiments --table-only   # rebuild comparison.json
D:\fedvenv\Scripts\python.exe scripts\show_results.py                   # summarise results/
D:\fedvenv\Scripts\python.exe -m uvicorn server.app:app --port 8000
cd web; npm run dev -- --port 5173 --host 127.0.0.1                     # dashboard
cd web; npm run build                                                   # typecheck + bundle
```

Scripts in `scripts/` need `PYTHONPATH` set to the repo root (or run them as `-m`).

## Gotchas learned the hard way

* **Long-running processes must be started in a tool-managed background shell**
  (`exec` with `timeout: 0`). Launching them with PowerShell `Start-Process` gets them
  killed when the one-shot parent shell exits.
* PowerShell pipes like `... | Select-Object -Last 40` **buffer everything until the
  process exits**, so you see no progress. Use `python -u` and read the shell output
  directly instead.
* `python -c "..."` with nested quotes breaks under PowerShell; put throwaway scripts in
  `scripts/` instead.

## Modelling facts worth remembering

* Learning on this task is driven by the **number of local optimizer steps**, not batch
  size: FedAvg with `local_batch_size=256` and 2 local epochs (~6 steps/round) does not
  learn, while `batch=32, local_epochs=4` (~85 steps/round) works well.
* Pure FedAvg (`server_optimizer="sgd"`, `server_lr=1.0`) clearly beats FedAdam here;
  server LRs above ~0.02 with FedAdam, or a centralized LR of 0.05, collapse the model.
* Centralized baseline wants **AdamW at 1e-3**, not SGD.
* **Per-example clipping cancels class weighting.** `CrossEntropyLoss(weight=...)` on a
  single example divides by that example's own weight, and even with `reduction="sum"` all
  weighted gradients exceed the clip norm and saturate. Rebalance DP runs by *sampling*
  (`LocalClient._sampling_rates`), never by loss weights.
* No momentum in DP-SGD: clipped gradients have fixed norm.
* Record-level DP epsilon composes **in parallel** across hospitals (max, not sum).
* Client-level DP-FedAvg is not viable with 10 silos (noise:signal ≈ σ·√d/m ≈ 57σ); it is
  kept as a documented negative result.

## Kaggle ops

* Kaggle OAuth credentials live under the user's `.kaggle` directory — **never read or
  print them**. Refresh an expired session with
  `D:\fedvenv\Scripts\kaggle.exe auth login --force`.
* Set `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8` for `kaggle` CLI logs/output on Windows
  or `kernels logs`/`output` die on cp1252/charmap encoding errors.
* A Kaggle T4 session can expose **two** GPUs; `fedderm_tpu.py` intentionally pins
  `OneDeviceStrategy("/GPU:0")` so `strategy.run` returns and record-level DP noise
  accounting stay single-replica. GPU 1 is unused on purpose.
* Import Kaggle results with
  `D:\fedvenv\Scripts\python.exe -m scripts.import_tpu_results <tpu_results.json>`
  (module form, not direct script invocation).
* Verify the accelerator from the kernel log, not submission intent alone: v13 reported no TPU endpoint and failed CUDA initialization, so it ran on CPU fallback despite the intended TPU submission.
