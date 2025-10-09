#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# [2025-10-10 | Hang Zhang (hzha0521)]
# Ablation launcher — CE + TripletX + SupCon (no SSL pretraining)
# Based on: run_b2.py
# =============================================================================

from __future__ import annotations
import argparse, json, os, re, subprocess, sys
from pathlib import Path
from datetime import datetime
import statistics as stats
from typing import Dict, Any, List, Optional, Tuple

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
CONFIG_PATH = "configs/VeRi/deit_transreid_stride_b2_supcon_tripletx.yml"
FULL_EPOCHS_DEFAULT = 120
SAVE_EVERY_DEFAULT  = 5
EVAL_EVERY_DEFAULT  = 5
DEFAULT_SEEDS       = [0, 1, 2]

# Default log root on Google Drive (can be overridden by --output-root or OUTPUT_ROOT env)
DRIVE_LOG_ROOT = "/content/drive/MyDrive/5703(hzha0521)/Optimized Model B (transformer based)/logs"

# Prefer local project dataset first, then Drive mirrors and common fallbacks
DATASET_CANDIDATES = [
    "/content/5703-capstone/Optimized Model B (transformer based)/datasets",
    "/content/drive/MyDrive/5703(hzha0521)/Optimized Model B (transformer based)/datasets",
    "/content/drive/MyDrive/datasets",
    "/content/datasets",
    "/workspace/datasets",
]

TEST_MARKERS = ("summary.json","test_summary.txt","results.txt","log.txt","test_log.txt","dist_mat.npy")

# -------------------------------------------------------------------------
# Environment helpers
# -------------------------------------------------------------------------
def detect_data_root() -> str:
    """Detect dataset root directory containing subfolder 'VeRi'."""
    env = os.getenv("DATASETS_ROOT")
    if env and (Path(env) / "VeRi").exists():
        return env
    for c in DATASET_CANDIDATES:
        if (Path(c) / "VeRi").exists():
            return c
    return str(Path.cwd() / "datasets")

DATA_ROOT = detect_data_root()
print(f"[ABL] Using DATASETS.ROOT_DIR={DATA_ROOT}")

def pick_base_log_root(cli_output_root: Optional[str]) -> Path:
    """Prefer Drive; allow env override; fallback to ./logs."""
    if cli_output_root:
        base = Path(cli_output_root)
    else:
        base = Path(os.getenv("OUTPUT_ROOT", DRIVE_LOG_ROOT))
        if not base.exists():
            base = Path("logs")
    base.mkdir(parents=True, exist_ok=True)
    return base

def per_seed_out_root(base_root: Path, folder_name: str, seed: int) -> Path:
    """logs/<folder_name>_seed{seed}_YYYYMMDD"""
    date_str = datetime.now().strftime("%Y%m%d")
    out = base_root / f"{folder_name}_seed{seed}_{date_str}"
    out.mkdir(parents=True, exist_ok=True)
    return out

# -------------------------------------------------------------------------
# Metric parsing
# -------------------------------------------------------------------------
def _re_pick(text: str, pat: str) -> float:
    m = re.search(pat, text, re.I)
    return float(m.group(1)) if m else -1.0

def parse_metrics(ep_dir: Path) -> Tuple[float,float,float,float]:
    """Parse mAP / Rank-1 / Rank-5 / Rank-10 from summary or log."""
    sj = ep_dir / "summary.json"
    if sj.exists():
        try:
            obj = json.loads(sj.read_text())
            return (
                float(obj.get("mAP", -1)),
                float(obj.get("Rank-1", obj.get("Rank1", -1))),
                float(obj.get("Rank-5", obj.get("Rank5", -1))),
                float(obj.get("Rank-10", obj.get("Rank10", -1))),
            )
        except:
            pass
    for name in ("test_summary.txt","results.txt","log.txt","test_log.txt"):
        p = ep_dir / name
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
        ep_idx = int(ep.name.split("_")[-1])
        mAP, r1, r5, r10 = parse_metrics(ep)
        if mAP < 0:
            continue
        key = (mAP, r1)
        if (best_key is None) or (key > best_key):
            best_key = key
            best = {"epoch": ep_idx, "mAP": mAP, "Rank-1": r1,
                    "Rank-5": r5, "Rank-10": r10}
    return best

# -------------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------------
def parse_seeds(arg_list: List[str]) -> List[int]:
    flat = []
    for a in arg_list:
        flat += [x for x in a.split(",") if x.strip()]
    return [int(x) for x in flat]

def _max_test_epoch(test_dir: Path) -> int:
    max_ep = 0
    for ep in test_dir.glob("epoch_*"):
        if not ep.is_dir(): continue
        try: idx = int(ep.name.split("_")[-1])
        except: continue
        for f in TEST_MARKERS:
            if (ep / f).exists():
                max_ep = max(max_ep, idx)
                break
    return max_ep

def _max_epoch_from_ckpts(run_dir: Path) -> int:
    max_ep = 0
    if not run_dir.exists(): return 0
    for p in run_dir.glob("transformer_*.pth"):
        m = re.search(r"transformer_(\d+)\.pth", p.name)
        if m: max_ep = max(max_ep, int(m.group(1)))
    return max_ep

# -------------------------------------------------------------------------
# CLI builder
# -------------------------------------------------------------------------
def build_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablation: CE + TripletX + SupCon (no SSL).")
    p.add_argument("--epochs", type=int, default=FULL_EPOCHS_DEFAULT,
                   help="Number of training epochs (default: 120).")
    p.add_argument("--save-every", type=int, default=SAVE_EVERY_DEFAULT,
                   help="Checkpoint save period (default: 5).")
    p.add_argument("--eval-every", type=int, default=EVAL_EVERY_DEFAULT,
                   help="Evaluation period (default: 5).")
    p.add_argument("--seeds", nargs="+", default=[str(s) for s in DEFAULT_SEEDS],
                   help="Comma or space separated seeds (default: 0 1 2).")
    p.add_argument("--folder-name", type=str, default="ablation_ce_tripletx_supcon",
                   help="Folder prefix under log root.")
    p.add_argument("--output-root", type=str, default=None,
                   help="Override entire output root path (disables Drive default).")
    p.add_argument("--tag", type=str, default=None,
                   help="Optional tag appended to run/test subfolders (dedup with seed).")
    p.add_argument("--no-test", action="store_true",
                   help="Train only, skip evaluation.")
    return p.parse_args()

# -------------------------------------------------------------------------
# Evaluation of saved checkpoints (resume-safe)
# -------------------------------------------------------------------------
def eval_saved_checkpoints(tag: str, config: str, out_root: Path):
    """Evaluate all saved checkpoints missing test results."""
    run_dir  = out_root / f"veri776_{tag}_deit_run"
    test_dir = out_root / f"veri776_{tag}_deit_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    ckpts = sorted(run_dir.glob("transformer_*.pth"),
                   key=lambda p: int(re.search(r"(\d+)", p.name).group(1)))
    if not ckpts:
        print(f"[ABL][eval] No checkpoints under {run_dir}")
        return
    for ck in ckpts:
        ep = int(re.search(r"(\d+)", ck.name).group(1))
        out_ep = test_dir / f"epoch_{ep}"
        if any((out_ep / f).exists() for f in TEST_MARKERS):
            print(f"[ABL][eval] Skip epoch {ep} (already tested)")
            continue
        out_ep.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "test.py", "--config_file", config,
            "MODEL.DEVICE", "cuda",
            "TEST.WEIGHT", str(ck),
            "OUTPUT_DIR", str(out_ep),
            "DATASETS.ROOT_DIR", DATA_ROOT,
        ]
        print("[ABL][eval] Launch:", " ".join(cmd))
        subprocess.call(cmd)

# -------------------------------------------------------------------------
# Core: per-seed run (resume supported)
# -------------------------------------------------------------------------
def run_one_seed(T: float, W: float, seed: int, epochs: int,
                 save_every: int, eval_every: int, out_root: Path,
                 do_test: bool, tag_extra: Optional[str]):
    # Inner folder tag (no seed/date here to avoid duplication)
    base_tag = "ablation_ce_tripletx_supcon"
    # Dedup tag if it already contains the seed pattern
    if tag_extra and f"seed{seed}" not in str(tag_extra):
        base_tag += f"_{tag_extra}"

    run_dir  = out_root / f"veri776_{base_tag}_deit_run"
    test_dir = out_root / f"veri776_{base_tag}_deit_test"

    trained_max = _max_epoch_from_ckpts(run_dir)
    tested_max  = _max_test_epoch(test_dir) if do_test else 0
    print(f"[ABL][diagnose] seed={seed} trained_max={trained_max} tested_max={tested_max}")

    if do_test and tested_max >= epochs:
        best = pick_best_epoch_metrics(test_dir)
        if best:
            print(f"[ABL] Already fully tested. Best={best}")
            return best

    if trained_max < epochs:
        cmd = [
            sys.executable, "run_modelB_deit.py",
            "--config", CONFIG_PATH,
            "--opts",
            "LOSS.SUPCON.ENABLE", "True",
            "LOSS.SUPCON.T", str(T),
            "LOSS.SUPCON.W", str(W),
            "LOSS.TRIPLETX.ENABLE", "True",
            "MODEL.TRAINING_MODE", "supervised",
            "SOLVER.MAX_EPOCHS", str(epochs),
            "SOLVER.SEED", str(seed),
            "SOLVER.CHECKPOINT_PERIOD", str(save_every),
            "SOLVER.EVAL_PERIOD", str(eval_every),
            "DATASETS.ROOT_DIR", DATA_ROOT,
            "OUTPUT_DIR", str(out_root),
            "TAG", base_tag,
        ]
        print("[ABL] Launch:", " ".join(cmd))
        subprocess.check_call(cmd)

    if not do_test:
        print(f"[ABL] Seed {seed} training done (test skipped).")
        return None

    eval_saved_checkpoints(base_tag, CONFIG_PATH, out_root)
    best = pick_best_epoch_metrics(test_dir)
    if not best:
        print(f"[ABL][warn] No metrics found under {test_dir}")
        return None
    print(f"[ABL] Seed {seed} best → {best}")
    return best

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    args = build_cli()
    base_root = pick_base_log_root(args.output_root)

    # Locate SupCon parameter file (T, W)
    drive_root = Path("/content/drive/MyDrive/5703(hzha0521)/Optimized Model B (transformer based)")
    best_json_candidates = [
        drive_root / "logs/b1_supcon_best.json",
        Path("logs/b1_supcon_best.json"),
    ]
    best_json = next((p for p in best_json_candidates if p.exists()), None)
    if not best_json:
        raise SystemExit(f"[ABL] Missing b1_supcon_best.json in any of: {best_json_candidates}")

    obj = json.loads(best_json.read_text())
    T, W = float(obj["T"]), float(obj["W"])
    print(f"[ABL] Using SupCon T={T}, W={W} from {best_json}")

    seeds = parse_seeds(args.seeds)
    seed_results: Dict[int, Dict[str, Any]] = {}

    # Run each seed in its own outer folder: <folder>_seed{seed}_YYYYMMDD
    for s in seeds:
        out_root = per_seed_out_root(base_root, args.folder_name, s)
        print(f"[ABL] Using output_root={out_root}")
        rec = run_one_seed(T, W, s, args.epochs, args.save_every,
                           args.eval_every, out_root, (not args.no_test), args.tag)
        if rec:
            # Write per-seed summary inside that seed's folder
            summary = {
                "setting": "Ablation CE+TripletX+SupCon (no SSL)",
                "config": CONFIG_PATH,
                "epochs": args.epochs,
                "save_every": args.save_every,
                "eval_every": args.eval_every,
                "data_root": DATA_ROOT,
                "T": T, "W": W, "seed": s, "best": rec,
            }
            (out_root / "ablation_seed_summary.json").write_text(json.dumps(summary, indent=2))
            seed_results[s] = rec

    # Print aggregate stats across seeds (kept lightweight since each seed has its own folder)
    if seed_results:
        mean = {k: round(stats.mean([r[k] for r in seed_results.values()]), 4)
                for k in ["mAP","Rank-1","Rank-5","Rank-10"]}
        std  = {k: round(stats.stdev([r[k] for r in seed_results.values()]), 4)
                if len(seed_results) >= 2 else 0.0
                for k in ["mAP","Rank-1","Rank-5","Rank-10"]}
        print(f"[ABL] Aggregate over seeds → mean={mean} | std={std}")
    else:
        print("[ABL] No summary (test disabled or no metrics).")

    print("[ABL] Done.")

if __name__ == "__main__":
    main()
