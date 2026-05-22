"""
run_experiment.py
=================
Top-level launcher for the RNN timing experiment pipeline.

Usage
-----
    python run_experiment.py

Or edit the TASKS / INITS lists below to run a subset.

Pipeline
--------
1. train_ensemble.py  — trains N_REPLICAS networks per (task, init) pair
2. analyze_ensemble.py — classifies each saved network and writes Table 2

Directory layout produced
-------------------------
results/
  <task_tag>_<init>/          # one folder per (task, init) pair
    net_00/
      weights_final.keras
      training_loss.png
      io_samples.png
      stats.txt               # per-network metrics written by analyze step
    net_01/ ...
  table2.txt                  # LaTeX-ready table
  table2_summary.csv          # machine-readable summary
"""

import subprocess
import sys
import os

# ── Experiment configuration ─────────────────────────────────────────────────

# Tasks available (must match names used in train_ensemble.py)
# Table-2 row → task name mapping:
#   Simple Delayed Binary DM          → "Simple DM"
#   Context-dependent Binary DM       → "Simple DM Long-short"
#   Multi-interval Amplitude-based    → "Simple DM 8 times"
#   Multi-interval Distance-based     → "Simple DM 4 times"   (distance encoded)
#   Windowed Evidence Integration     → "Integral DM"

"""
TASKS = [
    "Simple DM",
    "Simple DM Long-short",
    "Simple DM 8 times",
    "Simple DM 4 times",
    "Integral DM",
]
"""
TASKS = [
    "Simple DM Long-short"
]

INITS = ["Normal", "Orthogonal"]

N_REPLICAS = 10          # networks per (task, init)
N_REC      = 100         # recurrent units
RESULTS_DIR = "../results"  # root output directory

# ── Run ───────────────────────────────────────────────────────────────────────

os.makedirs(RESULTS_DIR, exist_ok=True)

python = sys.executable

for task in TASKS:
    for init in INITS:
        tag = task.replace(" ", "_") + "_" + init
        out_dir = os.path.join(RESULTS_DIR, tag)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"TRAINING:  task={task!r}  init={init!r}")
        print(f"Output dir: {out_dir}")
        print(f"{'='*60}")

        ret = subprocess.run(
            [python, "train_ensemble.py",
             "--task",       task,
             "--init",       init,
             "--n_replicas", str(N_REPLICAS),
             "--n_rec",      str(N_REC),
             "--out_dir",    out_dir],
            check=False,
        )
        if ret.returncode != 0:
            print(f"[WARNING] Training returned non-zero exit code for {tag}")

print(f"\n{'='*60}")
print("ANALYSIS — classifying networks and building Table 2")
print(f"{'='*60}")

ret = subprocess.run(
    [python, "analyze_ensemble.py",
     "--results_dir", RESULTS_DIR],
    check=False,
)
if ret.returncode != 0:
    print("[WARNING] Analysis step returned non-zero exit code")

print("\nDone. See results/table2.txt and results/table2_summary.csv")
