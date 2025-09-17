#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===========================================================
# File: run_b0.py
# Purpose: Baseline B0 training (CE + Triplet), multi-seed
#          with auto-resume, robust re-test, and summary.
# Author: Hang Zhang (hzha0521)
# ===========================================================
# Change Log
# [2025-09-17 | Hang Zhang] Initial B0 launcher:
#   - Run 3 seeds (0/1/2), auto-detect progress
#   - Robust test skipping (summary.json/test_log.txt/dist_mat.npy)
#   - Pick best epoch per seed (by mAP, tie Rank-1)
#   - Aggregate mean/std across seeds, save JSON summary
# ===========================================================

import json, os, re, subprocess, sys, statistics as stats
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# ========== User settings ==========
CONFIG      = "configs/VeRi/deit_transreid_stride_b0_baseline.yml"
FULL_EPOCHS = 30
SEEDS       = [0, 1, 2]
# ===================================

# ----- Env helpers -----
def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False

def pick_log_root() -> Path:
    env = os.getenv("OUTPUT_ROOT")
    if env:
        return Path(env)
    if _in_colab():
        try:
            from google.colab import drive
            if not Path("/content/drive/MyDrive").exists():
                drive.mount("/content/drive", force_remount=False)
        except Exception:
            pass
        dflt = "/content/drive/MyDrive/5703(hzha0521)/Optimized Model B (transformer based)/logs"
        return Path(os.getenv("DRIVE_LOG_ROOT", dflt))
    return Path("logs")

LOG_ROOT = pick_log_root()
LOG_ROOT.mkdir(parents=True, exist_ok=True)
SUMMARY_JSON = LOG_ROOT / "b0_baseline_best_summary.json"

# ----- Dataset root detection -----
def detect_data_root() -> str:
    env = os.getenv("DATASETS_ROOT")
    if env and (Path(env) / "VeRi").exists():
        return env
    candidates = [
        "/content/drive/MyDrive/datasets",
        "/content/drive/MyDrive/5703(hzha0521)/datasets",
        "/content/datasets",
        "/workspace/datasets",
        str(Path.cwd().parents[1] / "datasets"),
        str(Path.cwd() / "datasets"),
    ]
    for c in candidates:
        if (Path(c) / "VeRi").exists():
            return c
    return str(Path.cwd().parents[1] / "datasets")

DATA_ROOT = detect_data_root()
print(f"[B0] Using DATASETS.ROOT_DIR={DATA_ROOT}")

# ----- Helpers -----
TEST_MARKER_FILES = (
    "summary.json", "test_summary.txt", "results.txt",
    "log.txt", "test_log.txt", "dist_mat.npy"
)

def _fmt(v: float) -> str:
    return str(v).replace(".", "p")

def _re_pick(text: str, pat: str) -> float:
    m = re.search(pat, text, re.I)
    return float(m.group(1)) if m else -1.0

def parse_metrics_from_epoch_dir(epoch_dir: Path) -> Tuple[float, float, float, float]:
    sj = epoch_dir / "summary.json"
    if sj.exists():
        try:
            obj = json.loads(sj.read_text())
            return float(obj.get("mAP", -1)), float(obj.get("Rank-1", -1)), \
                   float(obj.get("Rank-5", -1)), float(obj.get("Rank-10", -1))
        except Exception:
            pass
    for name in ("test_summary.txt", "results.txt", "log.txt", "test_log.txt"):
        p = epoch_dir / name
        if p.exists():
            s = p.read_text(encoding="utf-8", errors="ignore")
            mAP = _re_pick(s, r"mAP[^0-9]*([0-9]+(?:\.[0-9]+)?)")
            r1  = _re_pick(s, r"Rank[-\s]?1[^0-9]*([0-9]+(?:\.[0-9]+)?)")
            r5  = _re_pick(s, r"Rank[-\s]?5[^0-9]*([0-9]+(?:\.[0-9]+)?)")
            r10 = _re_pick(s, r"Rank[-\s]?10[^0-9]*([0-9]+(?:\.[0-9]+)?)")
            return mAP, r1, r5, r10
    return -1.0, -1.0, -1.0, -1.0

def pick_best_epoch_metrics(test_dir: Path) -> Optional[Dict[str, Any]]:
    epochs = sorted([p for p in test_dir.glob("epoch_*") if p.is_dir()],
                    key=lambda p: int(p.name.split("_")[-1]))
    best, best_key = None, None
    for ep in epochs:
        idx = int(ep.name.split("_")[-1])
        mAP, r1, r5, r10 = parse_metrics_from_epoch_dir(ep)
        if mAP < 0: 
            continue
        key = (mAP, r1)
        if (best_key is None) or (key > best_key):
            best_key = key
            best = {"epoch": idx, "mAP": mAP, "Rank-1": r1, "Rank-5": r5, "Rank-10": r10}
    return best

def safe_mean(values: List[float]) -> float:
    vals = [v for v in values if v >= 0]
    return round(stats.mean(vals), 4) if vals else -1.0

def safe_stdev(values: List[float]) -> float:
    vals = [v for v in values if v >= 0]
    return round(stats.stdev(vals), 4) if len(vals) >= 2 else 0.0

# ----- Progress detection -----
def _max_test_epoch(test_dir: Path) -> int:
    max_ep = 0
    for ep in test_dir.glob("epoch_*"):
        if not ep.is_dir(): continue
        try:
            idx = int(ep.name.split("_")[-1])
        except Exception:
            continue
        for f in TEST_MARKER_FILES:
            if (ep / f).exists():
                max_ep = max(max_ep, idx)
                break
    return max_ep

def _max_epoch_from_checkpoints(run_dir: Path) -> int:
    max_ep = 0
    for p in run_dir.glob("transformer_*.pth"):
        m = re.search(r"transformer_(\d+)\.pth", p.name)
        if m: max_ep = max(max_ep, int(m.group(1)))
    return max_ep

# ----- Robust re-test -----
def eval_missing_epochs_via_test_py(tag: str, config_path: str, log_root: Path):
    run_dir  = log_root / f"veri776_{tag}_deit_run"
    test_dir = log_root / f"veri776_{tag}_deit_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    ckpts = []
    rx = re.compile(r"transformer_(\d+)\.pth$")
    for p in run_dir.glob("transformer_*.pth"):
        m = rx.search(p.name)
        if m: ckpts.append((int(m.group(1)), p))
    ckpts.sort()

    for ep, ck in ckpts:
        out_ep = test_dir / f"epoch_{ep}"
        if any((out_ep / f).exists() for f in TEST_MARKER_FILES):
            continue
        out_ep.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "test.py", "--config_file", str(config_path),
            "MODEL.DEVICE", "cuda",
            "TEST.WEIGHT", str(ck),
            "OUTPUT_DIR", str(out_ep),
            "DATASETS.ROOT_DIR", DATA_ROOT,
        ]
        print("[B0][eval] Launch:", " ".join(cmd))
        subprocess.call(cmd)

# ----- Idempotent runner per seed -----
def ensure_full_run(seed: int, epochs: int, log_root: Path) -> Optional[Dict[str, Any]]:
    tag = f"b0_baseline_seed{seed}"
    run_dir  = log_root / f"veri776_{tag}_deit_run"
    test_dir = log_root / f"veri776_{tag}_deit_test"

    tested_max = _max_test_epoch(test_dir)
    if tested_max >= epochs:
        best_ep = pick_best_epoch_metrics(test_dir)
        if best_ep:
            print(f"[B0] Already tested seed={seed}, best={best_ep}")
            return best_ep

    trained_max = _max_epoch_from_checkpoints(run_dir)
    if trained_max >= epochs:
        print(f"[B0] Trained to {trained_max}, reconstruct tests for seed={seed}")
        eval_missing_epochs_via_test_py(tag, CONFIG, log_root)
    else:
        print(f"[B0] Training seed={seed} (resume if exists, {trained_max}/{epochs})")
        subprocess.check_call([
            sys.executable, "run_modelB_deit.py",
            "--config", CONFIG,
            "--opts",
            "SOLVER.MAX_EPOCHS", str(epochs),
            "SOLVER.SEED", str(seed),
            "SOLVER.CHECKPOINT_PERIOD", "1",
            "SOLVER.EVAL_PERIOD", "1",
            "DATASETS.ROOT_DIR", DATA_ROOT,
            "OUTPUT_DIR", str(log_root),
            "TAG", "b0_baseline",  # seed 后缀由 trainer 自动加
        ])
        eval_missing_epochs_via_test_py(tag, CONFIG, log_root)

    best_ep = pick_best_epoch_metrics(test_dir)
    if best_ep:
        print(f"[B0] Seed {seed} best epoch: {best_ep}")
        return best_ep
    return None

# ----- Main -----
def main():
    seed_best: Dict[int, Dict[str, Any]] = {}
    for seed in SEEDS:
        rec = ensure_full_run(seed, FULL_EPOCHS, LOG_ROOT)
        if rec: seed_best[seed] = rec

    if seed_best:
        mAPs = [rec["mAP"] for rec in seed_best.values()]
        R1s  = [rec["Rank-1"] for rec in seed_best.values()]
        R5s  = [rec["Rank-5"] for rec in seed_best.values()]
        R10s = [rec["Rank-10"] for rec in seed_best.values()]
        summary = {
            "config": CONFIG,
            "epochs": FULL_EPOCHS,
            "data_root": DATA_ROOT,
            "seeds": {str(k): v for k, v in seed_best.items()},
            "mean": {
                "mAP": safe_mean(mAPs),
                "Rank-1": safe_mean(R1s),
                "Rank-5": safe_mean(R5s),
                "Rank-10": safe_mean(R10s),
            },
            "std": {
                "mAP": safe_stdev(mAPs),
                "Rank-1": safe_stdev(R1s),
                "Rank-5": safe_stdev(R5s),
                "Rank-10": safe_stdev(R10s),
            },
            "note": "Best epoch per seed picked by mAP (tie-breaker Rank-1)."
        }
        SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
        print(f"[B0] Wrote summary → {SUMMARY_JSON}")
    else:
        print("[B0][warn] No valid results collected.")

    print("[B0] Done.")

if __name__ == "__main__":
    main()
