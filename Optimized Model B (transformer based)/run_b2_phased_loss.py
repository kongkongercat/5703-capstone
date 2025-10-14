#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# B2_phased_loss launcher — run 3 seeds and summarize (SupCon T/W inherited from B1).
# =============================================================================
# File: run_b2_phased_loss.py
# Based on: run_b2.py
# Change Log (2025-10-14 | Hang Zhang)
# - Switch config to deit_transreid_stride_b2_phased_loss.yml (phased loss schedule).
# - Set FULL_EPOCHS=120 to align with A(0–30)/B(30–60)/C(60–120).
# - Keep SupCon T/W injection from B1 JSON; note: W is superseded by runtime phased weights.
# - Keep seamless resume/test flow; enforce CHECKPOINT_PERIOD=1 & EVAL_PERIOD=1.
# - No behavior changes beyond phased-loss adoption and epoch default.
# =============================================================================
"""
B2_phased_loss (ImageNet init):
- Use T/W from B1 JSON (W will be scaled by phased schedule at runtime).
- Always initialize from ImageNet (no warm-start from B1 checkpoint).
- Run seeds=[0,1,2] with FULL_EPOCHS, pick best epoch per seed (by mAP; tie Rank-1),
  then aggregate mean/std across seeds.

Seamless behavior:
- If tests already cover FULL_EPOCHS -> skip training/testing; parse directly.
- Else if checkpoints reached FULL_EPOCHS -> reconstruct missing tests via test.py.
- Else -> (re)train with CHECKPOINT_PERIOD=1 & EVAL_PERIOD=1, then reconstruct tests.
"""

from __future__ import annotations
import argparse
import json
import os
import re
import statistics as stats
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# ---- User-configurable ----
B2_CONFIG    = "configs/VeRi/deit_transreid_stride_b2_phased_loss.yml"
FULL_EPOCHS  = 120
SEEDS        = [0, 1, 2]

# Prefer writing logs/checkpoints to Google Drive by default
DRIVE_LOG_ROOT = "/content/drive/MyDrive/5703(hzha0521)/Optimized Model B (transformer based)/logs"

# Prefer local project dataset first, then Drive mirrors and common fallbacks
DATASET_CANDIDATES = [
    "/content/5703-capstone/Optimized Model B (transformer based)/datasets",
    "/content/drive/MyDrive/5703(hzha0521)/Optimized Model B (transformer based)/datasets",
    "/content/drive/MyDrive/datasets",
    "/content/datasets",
    "/workspace/datasets",
]

# ---- Test completion markers (unified) ----
TEST_MARKER_FILES_DEFAULT = (
    "summary.json", "test_summary.txt", "results.txt", "log.txt",
    "test_log.txt",
    "dist_mat.npy",
)

# ---------- Environment helpers ----------

def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False

def pick_log_root(cli_output_root: Optional[str]) -> Path:
    """
    Choose output root in priority:
    1) CLI argument (--output-root)
    2) Environment variable OUTPUT_ROOT
    3) Google Drive project logs (preferred)
    4) ./logs fallback
    """
    if cli_output_root:
        base = Path(cli_output_root)
    else:
        base = Path(os.getenv("OUTPUT_ROOT", DRIVE_LOG_ROOT))
        if not base.exists():
            base = Path("logs")
    base.mkdir(parents=True, exist_ok=True)
    return base

# ---------- Dataset root detection ----------

def detect_data_root() -> str:
    """Return a directory path whose subfolder 'VeRi' exists."""
    env = os.getenv("DATASETS_ROOT")
    if env and (Path(env) / "VeRi").exists():
        return env
    for c in DATASET_CANDIDATES:
        if (Path(c) / "VeRi").exists():
            return c
    return str(Path.cwd() / "datasets")

DATA_ROOT = detect_data_root()
print(f"[B2_phased] Using DATASETS.ROOT_DIR={DATA_ROOT}")

# ---------- Helpers ----------

def _fmt(v: float) -> str:
    return str(v).replace(".", "p")

def _re_pick(text: str, pat: str) -> float:
    m = re.search(pat, text, re.I)
    return float(m.group(1)) if m else -1.0

# ---------- Metrics parsing ----------

def parse_metrics_from_epoch_dir(epoch_dir: Path) -> Tuple[float, float, float, float]:
    sj = epoch_dir / "summary.json"
    if sj.exists():
        try:
            obj = json.loads(sj.read_text())
            return (
                float(obj.get("mAP", -1)),
                float(obj.get("Rank-1", obj.get("Rank1", -1))),
                float(obj.get("Rank-5", obj.get("Rank5", -1))),
                float(obj.get("Rank-10", obj.get("Rank10", -1))),
            )
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
    epochs = sorted(
        [p for p in test_dir.glob("epoch_*") if p.is_dir()],
        key=lambda p: int(p.name.split("_")[-1])
    )
    best, best_key = None, None
    for ep in epochs:
        ep_idx = int(ep.name.split("_")[-1])
        mAP, r1, r5, r10 = parse_metrics_from_epoch_dir(ep)
        if mAP < 0:
            continue
        key = (mAP, r1)
        if (best_key is None) or (key > best_key):
            best_key = key
            best = {"epoch": ep_idx, "mAP": mAP, "Rank-1": r1, "Rank-5": r5, "Rank-10": r10}
    return best

def safe_mean(vals: List[float]) -> float:
    valid = [v for v in vals if v >= 0]
    return round(stats.mean(valid), 4) if valid else -1.0

def safe_stdev(vals: List[float]) -> float:
    valid = [v for v in vals if v >= 0]
    return round(stats.stdev(valid), 4) if len(valid) >= 2 else 0.0

# ---------- CLI args ----------

def parse_seeds(arg_list: List[str]) -> List[int]:
    flat = []
    for a in arg_list:
        flat += [x for x in a.split(",") if x.strip()]
    return [int(x) for x in flat]

def build_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run B2_phased_loss with flexible options.")
    p.add_argument("--test", dest="test", action="store_true", help="Enable test phase (default).")
    p.add_argument("--no-test", dest="test", action="store_false", help="Skip test phase (train only).")
    p.set_defaults(test=True)
    p.add_argument("--epochs", type=int, default=FULL_EPOCHS, help="Number of training epochs.")
    p.add_argument("--seeds", nargs="+", default=[str(s) for s in SEEDS],
                   help="Comma or space separated seeds list. Default: [0,1,2].")
    p.add_argument("--output-root", type=str, default=None, help="Override output root directory.")
    p.add_argument("--tag", type=str, default=None, help="Custom tag (used in output dir names).")
    return p.parse_args()

# ---------- Progress detection & test ----------

def _max_test_epoch(test_dir: Path) -> int:
    max_ep = 0
    for ep in test_dir.glob("epoch_*"):
        if not ep.is_dir():
            continue
        try:
            idx = int(ep.name.split("_")[-1])
        except:
            continue
        for f in TEST_MARKER_FILES_DEFAULT:
            if (ep / f).exists():
                max_ep = max(max_ep, idx)
                break
    return max_ep

def _max_epoch_from_checkpoints(run_dir: Path) -> int:
    max_ep = 0
    if not run_dir.exists():
        return 0
    for p in run_dir.glob("transformer_*.pth"):
        m = re.search(r"transformer_(\d+)\.pth", p.name)
        if m:
            max_ep = max(max_ep, int(m.group(1)))
    return max_ep

def eval_missing_epochs_via_test_py(tag: str, config: str, log_root: Path):
    run_dir = log_root / f"veri776_{tag}_deit_run"
    test_dir = log_root / f"veri776_{tag}_deit_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    ckpts = sorted(run_dir.glob("transformer_*.pth"), key=lambda p: int(re.search(r"(\d+)", p.name).group(1)))
    if not ckpts:
        print(f"[B2_phased][eval] No checkpoints under {run_dir}")
        return
    for ck in ckpts:
        ep = int(re.search(r"(\d+)", ck.name).group(1))
        out_ep = test_dir / f"epoch_{ep}"
        if any((out_ep / f).exists() for f in TEST_MARKER_FILES_DEFAULT):
            print(f"[B2_phased][eval] Skip epoch {ep} (already tested)")
            continue
        out_ep.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "test.py", "--config_file", config,
            "MODEL.DEVICE", "cuda",
            "TEST.WEIGHT", str(ck),
            "OUTPUT_DIR", str(out_ep),
            "DATASETS.ROOT_DIR", DATA_ROOT,
        ]
        print("[B2_phased][eval] Launch:", " ".join(cmd))
        subprocess.call(cmd)

# ---------- Seed runner ----------

def ensure_full_run_seed(T, W, seed, epochs, log_root, test_enabled, tag=None):
    tag = tag or f"b2_phased_T{_fmt(T)}_W{_fmt(W)}_seed{seed}"
    run_dir = log_root / f"veri776_{tag}_deit_run"
    test_dir = log_root / f"veri776_{tag}_deit_test"

    trained_max = _max_epoch_from_checkpoints(run_dir)
    tested_max = _max_test_epoch(test_dir) if test_enabled else 0
    print(f"[B2_phased][diagnose] seed={seed} trained_max={trained_max} tested_max={tested_max}")

    if test_enabled and tested_max >= epochs:
        best_ep = pick_best_epoch_metrics(test_dir)
        if best_ep:
            print(f"[B2_phased] Already tested up to {tested_max}. Best={best_ep}")
            return best_ep

    if trained_max < epochs:
        cmd = [
            sys.executable, "run_modelB_deit.py",
            "--config", B2_CONFIG,
            "--opts",
            "LOSS.SUPCON.ENABLE", "True",
            "LOSS.SUPCON.T", str(T),
            "LOSS.SUPCON.W", str(W),  # Note: runtime phased weights will rescale this.
            "LOSS.TRIPLETX.ENABLE", "True",
            "MODEL.TRAINING_MODE", "supervised",
            "SOLVER.MAX_EPOCHS", str(epochs),
            "SOLVER.SEED", str(seed),
            "SOLVER.CHECKPOINT_PERIOD", "1",
            "SOLVER.EVAL_PERIOD", "1",
            "DATASETS.ROOT_DIR", DATA_ROOT,
            "OUTPUT_DIR", str(log_root),
            "TAG", tag,
        ]
        print("[B2_phased] Launch:", " ".join(cmd))
        subprocess.check_call(cmd)

    if not test_enabled:
        print(f"[B2_phased] Seed {seed} training done (test skipped).")
        return None

    eval_missing_epochs_via_test_py(tag, B2_CONFIG, log_root)
    best_ep = pick_best_epoch_metrics(test_dir)
    if not best_ep:
        print(f"[B2_phased][warn] No metrics found under {test_dir}")
        return None
    print(f"[B2_phased] Seed {seed} best epoch → {best_ep}")
    return best_ep

# ---------- Main ----------

def main():
    args = build_cli()
    log_root = pick_log_root(args.output_root)
    print(f"[B2_phased] Using log_root={log_root}")

    # Locate SupCon (T, W) JSON from multiple candidates
    drive_root = Path("/content/drive/MyDrive/5703(hzha0521)/Optimized Model B (transformer based)")
    candidates = [
        log_root / "b1_supcon_best.json",                    # preferred: current log root
        drive_root / "logs/b1_supcon_best.json",             # project logs on Drive
        Path("logs/b1_supcon_best.json"),                    # local fallback
        Path(os.getenv("B1_JSON", "")) if os.getenv("B1_JSON") else None,  # env override
    ]
    best_json = next((p for p in candidates if p and p.exists()), None)
    if not best_json:
        raise SystemExit(f"[B2_phased] Missing b1_supcon_best.json in any of: {candidates}")

    obj = json.loads(best_json.read_text())
    T, W = float(obj["T"]), float(obj["W"])
    print(f"[B2_phased] Using SupCon T={T}, W={W} from {best_json}")

    seeds = parse_seeds(args.seeds)
    seed_best = {}
    for s in seeds:
        rec = ensure_full_run_seed(T, W, s, args.epochs, log_root, args.test, args.tag)
        if rec:
            seed_best[s] = rec

    if seed_best:
        summary = {
            "T": T, "W": W, "epochs": args.epochs, "config": B2_CONFIG,
            "data_root": DATA_ROOT,
            "seeds": {str(k): v for k, v in seed_best.items()},
            "mean": {k: safe_mean([r[k] for r in seed_best.values()]) for k in ["mAP","Rank-1","Rank-5","Rank-10"]},
            "std":  {k: safe_stdev([r[k] for r in seed_best.values()]) for k in ["mAP","Rank-1","Rank-5","Rank-10"]},
        }
        (log_root / "b2_phased_best_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[B2_phased] Summary written → {log_root/'b2_phased_best_summary.json'}")
    else:
        print("[B2_phased] No summary (test disabled or no metrics).")
    print("[B2_phased] Done.")

if __name__ == "__main__":
    main()
