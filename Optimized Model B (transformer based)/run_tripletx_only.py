#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===========================================================
# File: run_tripletx_only_120e.py
# Purpose: TripletX-only training (CE + TripletX), multi-seed
#          with auto-resume, robust re-test, and summary.
# Author: Hang Zhang (hzha0521) — adapted from run_b0.py
# ===========================================================
# Default policy:
# - Train 120 epochs
# - Save checkpoint every 3 epochs
# - Evaluate every 3 epochs
# - SupCon disabled (TripletX only)
# - Output folders auto-tagged with current date-time (YYYYMMDD_HHMM)
# ===========================================================

from __future__ import annotations
import json, os, re, subprocess, sys, statistics as stats, argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# ========== User defaults ==========
CONFIG              = "configs/VeRi/deit_transreid_stride_b2_supcon_tripletx.yml"
FULL_EPOCHS         = 120
CHECKPOINT_PERIOD   = 3
EVAL_PERIOD         = 3
SEEDS_DEFAULT       = [0]
TAG_BASE            = "tripletx_only"
# ===================================

TEST_MARKER_FILES = (
    "summary.json", "test_summary.txt", "results.txt",
    "log.txt", "test_log.txt", "dist_mat.npy",
)

# ----- Colab detection & log root -----

def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False

def pick_log_root(cli_output_root: Optional[str]) -> Path:
    if cli_output_root:
        root = Path(cli_output_root)
    else:
        env = os.getenv("OUTPUT_ROOT")
        if env:
            root = Path(env)
        elif _in_colab():
            dflt = "/content/drive/MyDrive/5703(hzha0521)/Optimized Model B (transformer based)/logs"
            root = Path(os.getenv("DRIVE_LOG_ROOT", dflt))
        else:
            root = Path("logs")
    root.mkdir(parents=True, exist_ok=True)
    return root

# ----- Dataset root detection -----

def detect_data_root() -> str:
    env = os.getenv("DATASETS.ROOT_DIR") or os.getenv("DATASETS_ROOT")
    if env and (Path(env) / "VeRi").exists():
        return env
    candidates = [
        "/content/5703-capstone/Optimized Model B (transformer based)/datasets",
        "/content/drive/MyDrive/5703(hzha0521)/Optimized Model B (transformer based)/datasets",
        "/content/drive/MyDrive/datasets",
        "/content/datasets",
        "/workspace/datasets",
        str(Path.cwd() / "datasets"),
    ]
    for c in candidates:
        if (Path(c) / "VeRi").exists():
            return c
    return str(Path.cwd() / "datasets")

DATA_ROOT = detect_data_root()
print(f"[TX] Using DATASETS.ROOT_DIR={DATA_ROOT}")

# ----- Helpers -----

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

# ----- Eval device -----

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

# ----- Robust re-test -----

def eval_missing_epochs_via_test_py(tag: str, config_path: str, log_root: Path):
    run_dir  = log_root / f"veri776_{tag}_deit_run"
    test_dir = log_root / f"veri776_{tag}_deit_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    ckpts = sorted(run_dir.glob("transformer_*.pth"),
                   key=lambda p: int(re.search(r"(\d+)", p.name).group(1)))
    device = pick_eval_device()

    for ck in ckpts:
        ep = int(re.search(r"(\d+)", ck.name).group(1))
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
        print("[TX][eval] Launch:", " ".join(cmd))
        subprocess.call(cmd)

# ----- Per-seed run -----

def ensure_full_run(seed: int, epochs: int, ckp_period: int, eval_period: int,
                    log_root: Path, tag_base: str, train_only: bool = False
                    ) -> Optional[Dict[str, Any]]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    tag = f"{tag_base}_seed{seed}_{timestamp}"
    run_dir  = log_root / f"veri776_{tag}_deit_run"
    test_dir = log_root / f"veri776_{tag}_deit_test"

    trained_max = max([int(re.search(r"(\d+)", p.name).group(1)) for p in run_dir.glob("transformer_*.pth")], default=0)

    if trained_max >= epochs:
        print(f"[TX] Trained to {trained_max}, reconstruct tests for seed={seed}")
        if not train_only:
            eval_missing_epochs_via_test_py(tag, CONFIG, log_root)
    else:
        print(f"[TX] Training seed={seed} (resume {trained_max}/{epochs})")
        ckpt_path = run_dir / f"transformer_{trained_max}.pth"
        cmd = [
            sys.executable, "run_modelB_deit.py",
            "--config", CONFIG,
            "--opts",
            "SOLVER.MAX_EPOCHS", str(epochs),
            "SOLVER.SEED", str(seed),
            "SOLVER.CHECKPOINT_PERIOD", str(ckp_period),
            "SOLVER.EVAL_PERIOD", str(eval_period),
            "LOSS.SUPCON.ENABLE", "False",
            "LOSS.TRIPLETX.ENABLE", "True",
            "MODEL.TRAINING_MODE", "supervised",
            "DATASETS.ROOT_DIR", DATA_ROOT,
            "OUTPUT_DIR", str(log_root),
            "TAG", f"{tag_base}_{timestamp}",
        ]
        if ckpt_path.exists() and trained_max > 0:
            cmd += ["MODEL.PRETRAIN_CHOICE", "resume", "MODEL.PRETRAIN_PATH", str(ckpt_path)]
        print("[TX][train] Launch:", " ".join(cmd))
        subprocess.check_call(cmd)
        if not train_only:
            eval_missing_epochs_via_test_py(tag, CONFIG, log_root)

    if train_only:
        print(f"[TX] Seed {seed} training finished (test skipped).")
        return None

    best_ep = pick_best_epoch_metrics(test_dir)
    if best_ep:
        print(f"[TX] Seed {seed} best epoch: {best_ep}")
        (log_root / f"{tag}_best.json").write_text(json.dumps(best_ep, indent=2))
        return best_ep
    return None

# ----- CLI & Main -----

def build_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TripletX-only runner (multi-seed, robust resume/test).")
    p.add_argument("--epochs", type=int, default=FULL_EPOCHS)
    p.add_argument("--ckp", type=int, default=CHECKPOINT_PERIOD)
    p.add_argument("--eval", type=int, default=EVAL_PERIOD)
    p.add_argument("--seeds", nargs="+", default=[str(s) for s in SEEDS_DEFAULT])
    p.add_argument("--output-root", type=str, default=None)
    p.add_argument("--tag", type=str, default=TAG_BASE)
    p.add_argument("--train-only", action="store_true")
    return p.parse_args()

def parse_seeds(raw: List[str]) -> List[int]:
    flat = []
    for s in raw:
        flat += [x for x in s.split(",") if x.strip()]
    return [int(x) for x in flat]

def main():
    args = build_cli()
    seeds = parse_seeds(args.seeds)
    log_root = pick_log_root(args.output_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    print(f"[TX] Using log_root={log_root}")

    seed_best: Dict[int, Dict[str, Any]] = {}
    for seed in seeds:
        rec = ensure_full_run(seed, args.epochs, args.ckp, args.eval,
                              log_root, args.tag, train_only=args.train_only)
        if rec:
            seed_best[seed] = rec

    if seed_best:
        mAPs = [r["mAP"] for r in seed_best.values()]
        R1s  = [r["Rank-1"] for r in seed_best.values()]
        summary_name = f"{args.tag}_summary_{timestamp}.json"
        summary = {
            "config": CONFIG,
            "epochs": args.epochs,
            "ckp_period": args.ckp,
            "eval_period": args.eval,
            "log_root": str(log_root),
            "tag": args.tag,
            "mean": {"mAP": safe_mean(mAPs), "Rank-1": safe_mean(R1s)},
            "std": {"mAP": safe_stdev(mAPs), "Rank-1": safe_stdev(R1s)},
            "note": "Auto-tagged by date-time; best epoch per seed picked by mAP.",
        }
        (log_root / summary_name).write_text(json.dumps(summary, indent=2))
        print(f"[TX] Wrote summary → {summary_name}")
    else:
        print("[TX] No valid results collected.")
    print("[TX] Done.")

if __name__ == "__main__":
    main()
