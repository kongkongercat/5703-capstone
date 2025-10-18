#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===========================================================
# File: run_b0.py
# Purpose: Baseline B0 training (CE + Triplet), multi-seed
#          with configurable epochs, save period, and seeds.
# Author: Hang Zhang (hzha0521)
# ===========================================================
# Change Log
# [2025-09-17 | Hang Zhang] Initial version (multi-seed baseline launcher)
# [2025-10-11 | Hang Zhang] Enhanced CLI:
#   - Add argparse for epochs/save-every/seeds
#   - Auto-tag log folders with date-time (YYYYMMDD_HHMM)
# [2025-10-12 | Hang Zhang] Unify naming to:
#   veri776_b0_baseline_<DATE_TAG>_seed<seed>_deit_{run|test}
# [2025-10-18 | Hang Zhang] Make --save-every dynamic for checkpoints
#   and sync SOLVER.EVAL_PERIOD to avoid ZeroDivisionError
# ===========================================================

from __future__ import annotations
import json, os, re, subprocess, sys, statistics as stats, argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# ========== Default user settings ==========
CONFIG = "configs/VeRi/deit_transreid_stride_b0_baseline.yml"
DEFAULT_EPOCHS = 30
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_SAVE_EVERY = 1
# ===========================================


def parse_args():
    parser = argparse.ArgumentParser(description="B0 Baseline Launcher")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Total training epochs")
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY, help="Checkpoint save frequency (epochs)")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="List of seeds")
    return parser.parse_args()


# ----- Colab detection & log root -----
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
            from google.colab import drive  # type: ignore
            if not Path("/content/drive/MyDrive").exists():
                drive.mount("/content/drive", force_remount=False)
        except Exception:
            pass
        dflt = "/content/drive/MyDrive/5703(hzha0521)/Optimized Model B (transformer based)/logs"
        return Path(os.getenv("DRIVE_LOG_ROOT", dflt))
    return Path("logs")


LOG_ROOT = pick_log_root()
LOG_ROOT.mkdir(parents=True, exist_ok=True)

# Tag with current date/time
DATE_TAG = datetime.now().strftime("%Y%m%d_%H%M")
SUMMARY_JSON = LOG_ROOT / f"b0_baseline_best_summary_{DATE_TAG}.json"

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
    return str(Path.cwd() / "datasets")


DATA_ROOT = detect_data_root()
print(f"[B0] Using DATASETS.ROOT_DIR={DATA_ROOT}")

if not (Path(DATA_ROOT) / "VeRi").exists():
    print("[B0][warn] VeRi folder not found under DATASETS.ROOT_DIR.")


# ----- Helper functions -----
TEST_MARKER_FILES = (
    "summary.json", "test_summary.txt", "results.txt",
    "log.txt", "test_log.txt", "dist_mat.npy",
)


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


# ----- Eval device selection -----
def pick_eval_device() -> str:
    env = os.getenv("EVAL_DEVICE")
    if env:
        return env
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore
            return "mps"
    except Exception:
        pass
    return "cpu"


# ----- Eval missing (unified naming: date first, then seed) -----
def eval_missing_epochs_via_test_py(tag: str, config_path: str, log_root: Path):
    run_dir  = log_root / f"veri776_{tag}_deit_run"
    test_dir = log_root / f"veri776_{tag}_deit_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    ckpts = []
    rx = re.compile(r"transformer_(\d+)\.pth$")
    for p in run_dir.glob("transformer_*.pth"):
        m = rx.search(p.name)
        if m:
            ckpts.append((int(m.group(1)), p))
    ckpts.sort()

    device = pick_eval_device()

    for ep, ck in ckpts:
        out_ep = test_dir / f"epoch_{ep}"
        if any((out_ep / f).exists() for f in TEST_MARKER_FILES):
            continue
        out_ep.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "test.py", "--config_file", str(config_path),
            "MODEL.DEVICE", device,
            "TEST.WEIGHT", str(ck),
            "OUTPUT_DIR", str(out_ep),
            "DATASETS.ROOT_DIR", DATA_ROOT,
        ]
        print("[B0][eval] Launch:", " ".join(cmd))
        subprocess.call(cmd)


# ----- Training runner (unified naming) -----
def ensure_full_run(seed: int, epochs: int, save_every: int, log_root: Path) -> Optional[Dict[str, Any]]:
    tag = f"b0_baseline_{DATE_TAG}_seed{seed}"
    run_dir  = log_root / f"veri776_{tag}_deit_run"
    test_dir = log_root / f"veri776_{tag}_deit_test"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[B0] Training seed={seed}, total={epochs}, save every={save_every}")

    # Sync EVAL_PERIOD with save_every to avoid ZeroDivisionError inside the trainer
    eval_every = max(1, int(save_every))

    cmd = [
        sys.executable, "run_modelB_deit.py",
        "--config", CONFIG,
        "--opts",
        "SOLVER.MAX_EPOCHS", str(epochs),
        "SOLVER.SEED", str(seed),
        "SOLVER.CHECKPOINT_PERIOD", str(save_every),
        "SOLVER.EVAL_PERIOD", str(eval_every),
        "DATASETS.ROOT_DIR", DATA_ROOT,
        "OUTPUT_DIR", str(log_root),
        "TAG", f"b0_baseline_{DATE_TAG}",
    ]
    print("[B0][train] Launch:", " ".join(cmd))
    subprocess.check_call(cmd)

    # Evaluate all saved checkpoints into the unified test directory
    eval_missing_epochs_via_test_py(tag, CONFIG, log_root)

    best_ep = pick_best_epoch_metrics(test_dir)
    if best_ep:
        (log_root / f"b0_baseline_seed{seed}_best_{DATE_TAG}.json").write_text(json.dumps(best_ep, indent=2))
    return best_ep


# ----- Main -----
def main():
    args = parse_args()
    seed_best: Dict[int, Dict[str, Any]] = {}
    for seed in args.seeds:
        rec = ensure_full_run(seed, args.epochs, args.save_every, LOG_ROOT)
        if rec:
            seed_best[seed] = rec

    if seed_best:
        mAPs = [rec["mAP"] for rec in seed_best.values()]
        R1s  = [rec["Rank-1"] for rec in seed_best.values()]
        R5s  = [rec["Rank-5"] for rec in seed_best.values()]
        R10s = [rec["Rank-10"] for rec in seed_best.values()]
        summary = {
            "config": CONFIG,
            "epochs": args.epochs,
            "save_every": args.save_every,
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
            "note": f"Auto-tagged {DATE_TAG}, best epoch per seed by mAP (tie Rank-1).",
        }
        SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
        print(f"[B0] Wrote summary → {SUMMARY_JSON}")
    else:
        print("[B0][warn] No valid results collected.")

    print("[B0] Done.")


if __name__ == "__main__":
    main()
