#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Phased-loss launcher — run N seeds and summarize (generic; supports B1/B2/B4)
# =============================================================================
# File: run_phased_loss.py
#
# Change Log
# [2025-10-14 | Hang Zhang] Initial migration to phased-loss config (A/B/C schedule)
# [2025-10-15 | Hang Zhang] CLI passthrough & dataset prefix (NEW)
# [2025-10-18 | Team 5703]  Renamed from run_b2_phased_loss.py; unified all tags/prints.
#                           Default config renamed to deit_transreid_stride_phased_loss.yml.
# =============================================================================

from __future__ import annotations
import argparse, json, os, re, statistics as stats, subprocess, sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# ---- User-configurable defaults ----
DEFAULT_CONFIG = "configs/VeRi/deit_transreid_stride_phased_loss.yml"  # <--- updated
FULL_EPOCHS    = 120
SEEDS          = [0, 1, 2]

DRIVE_LOG_ROOT = "/content/drive/MyDrive/5703(hzha0521)/Optimized Model B (transformer based)/logs"

DATASET_CANDIDATES = [
    "/content/5703-capstone/Optimized Model B (transformer based)/datasets",
    "/content/drive/MyDrive/5703(hzha0521)/Optimized Model B (transformer based)/datasets",
    "/content/drive/MyDrive/datasets",
    "/content/datasets",
    "/workspace/datasets",
]

TEST_MARKER_FILES_DEFAULT = ("summary.json", "test_summary.txt", "results.txt",
                             "log.txt", "test_log.txt", "dist_mat.npy")

def pick_log_root(cli_output_root: Optional[str]) -> Path:
    base = Path(cli_output_root or os.getenv("OUTPUT_ROOT", DRIVE_LOG_ROOT))
    if not base.exists():
        base = Path("logs")
    base.mkdir(parents=True, exist_ok=True)
    return base

def detect_data_root() -> str:
    env = os.getenv("DATASETS_ROOT")
    if env and (Path(env) / "VeRi").exists():
        return env
    for c in DATASET_CANDIDATES:
        if (Path(c) / "VeRi").exists():
            return c
    return str(Path.cwd() / "datasets")

DATA_ROOT = detect_data_root()
print(f"[phased_loss] Using DATASETS.ROOT_DIR={DATA_ROOT}")

def _fmt(v: float) -> str: return str(v).replace(".", "p")
def _re_pick(text: str, pat: str) -> float:
    m = re.search(pat, text, re.I); return float(m.group(1)) if m else -1.0

def parse_metrics_from_epoch_dir(epoch_dir: Path) -> Tuple[float, float, float, float]:
    sj = epoch_dir / "summary.json"
    if sj.exists():
        try:
            obj = json.loads(sj.read_text())
            return (float(obj.get("mAP", -1)),
                    float(obj.get("Rank-1", obj.get("Rank1", -1))),
                    float(obj.get("Rank-5", obj.get("Rank5", -1))),
                    float(obj.get("Rank-10", obj.get("Rank10", -1))))
        except Exception:
            pass
    for name in ("test_summary.txt","results.txt","log.txt","test_log.txt"):
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
        ep_idx = int(ep.name.split("_")[-1])
        mAP, r1, r5, r10 = parse_metrics_from_epoch_dir(ep)
        if mAP < 0: continue
        key = (mAP, r1)
        if (best_key is None) or (key > best_key):
            best_key = key
            best = {"epoch": ep_idx, "mAP": mAP, "Rank-1": r1, "Rank-5": r5, "Rank-10": r10}
    return best

def safe_mean(vals: List[float]) -> float:
    valid = [v for v in vals if v >= 0]; return round(stats.mean(valid), 4) if valid else -1.0
def safe_stdev(vals: List[float]) -> float:
    valid = [v for v in vals if v >= 0]; return round(stats.stdev(valid), 4) if len(valid) >= 2 else 0.0
def parse_seeds(arg_list: List[str]) -> List[int]:
    flat = [];  [flat.append(x) for a in arg_list for x in a.split(",") if x.strip()]
    return [int(x) for x in flat]

def build_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run phased-loss with flexible options.")
    p.add_argument("--test", dest="test", action="store_true"); p.add_argument("--no-test", dest="test", action="store_false"); p.set_defaults(test=True)
    p.add_argument("--epochs", type=int, default=FULL_EPOCHS)
    p.add_argument("--seeds", nargs="+", default=[str(s) for s in SEEDS])
    p.add_argument("--output-root", type=str, default=None)
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--save-every", type=int, default=1, help="Checkpointing/testing interval (epochs).")
    p.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Path to config YAML for run_modelB_deit.py")
    p.add_argument("--ds-prefix", type=str, default="veri776", help="Prefix for run/test dir naming")
    p.add_argument("--opts", nargs=argparse.REMAINDER, help="Extra YACS options to pass through")
    return p.parse_args()

def _max_test_epoch(test_dir: Path) -> int:
    max_ep = 0
    for ep in test_dir.glob("epoch_*"):
        if not ep.is_dir(): continue
        try: idx = int(ep.name.split("_")[-1])
        except: continue
        for f in TEST_MARKER_FILES_DEFAULT:
            if (ep / f).exists(): max_ep = max(max_ep, idx); break
    return max_ep

def _max_epoch_from_checkpoints(run_dir: Path) -> int:
    max_ep = 0
    if not run_dir.exists(): return 0
    for p in run_dir.glob("transformer_*.pth"):
        m = re.search(r"transformer_(\d+)\.pth", p.name)
        if m: max_ep = max(max_ep, int(m.group(1)))
    return max_ep

def eval_missing_epochs_via_test_py(tag: str, config: str, log_root: Path, ds_prefix: str):
    run_dir = log_root / f"{ds_prefix}_{tag}_deit_run"
    test_dir = log_root / f"{ds_prefix}_{tag}_deit_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    ckpts = sorted(run_dir.glob("transformer_*.pth"), key=lambda p: int(re.search(r"(\d+)", p.name).group(1)))
    if not ckpts:
        print(f"[phased_loss][eval] No checkpoints under {run_dir}"); return
    for ck in ckpts:
        ep = int(re.search(r"(\d+)", ck.name).group(1))
        out_ep = test_dir / f"epoch_{ep}"
        if any((out_ep / f).exists() for f in TEST_MARKER_FILES_DEFAULT):
            print(f"[phased_loss][eval] Skip epoch {ep} (already tested)"); continue
        out_ep.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "test.py", "--config_file", config,
               "MODEL.DEVICE", "cuda",
               "TEST.WEIGHT", str(ck),
               "OUTPUT_DIR", str(out_ep),
               "DATASETS.ROOT_DIR", DATA_ROOT]
        print("[phased_loss][eval] Launch:", " ".join(cmd)); subprocess.call(cmd)

def ensure_full_run_seed(T, W, seed, epochs, save_every, log_root, test_enabled, tag_base, stamp, *, args):
    tag = f"{stamp}_{tag_base}"
    run_dir  = log_root / f"{args.ds_prefix}_{tag}_deit_run"
    test_dir = log_root / f"{args.ds_prefix}_{tag}_deit_test"
    trained_max = _max_epoch_from_checkpoints(run_dir)
    tested_max  = _max_test_epoch(test_dir) if test_enabled else 0
    print(f"[phased_loss][diagnose] seed={seed} trained_max={trained_max} tested_max={tested_max}")

    if test_enabled and tested_max >= epochs:
        best_ep = pick_best_epoch_metrics(test_dir)
        if best_ep: print(f"[phased_loss] Already tested up to {tested_max}. Best={best_ep}"); return best_ep

    if trained_max < epochs:
        cmd = [sys.executable, "run_modelB_deit.py",
               "--config", args.config, "--opts",
               "LOSS.SUPCON.ENABLE", "True",
               "LOSS.SUPCON.T", str(T),
               "LOSS.SUPCON.W", str(W),
               "LOSS.TRIPLETX.ENABLE", "True",
               "MODEL.TRAINING_MODE", "supervised",
               "SOLVER.MAX_EPOCHS", str(epochs),
               "SOLVER.SEED", str(seed),
               "SOLVER.CHECKPOINT_PERIOD", str(save_every),
               "SOLVER.EVAL_PERIOD", str(save_every),
               "DATASETS.ROOT_DIR", DATA_ROOT,
               "OUTPUT_DIR", str(log_root),
               "TAG", tag]
        if args.opts: cmd += args.opts
        print("[phased_loss] Launch:", " ".join(cmd)); subprocess.check_call(cmd)

    if not test_enabled:
        print(f"[phased_loss] Seed {seed} training done (test skipped)."); return None
    eval_missing_epochs_via_test_py(tag, args.config, log_root, args.ds_prefix)
    best_ep = pick_best_epoch_metrics(test_dir)
    if not best_ep: print(f"[phased_loss][warn] No metrics found under {test_dir}"); return None
    print(f"[phased_loss] Seed {seed} best epoch → {best_ep}"); return best_ep

def main():
    args = build_cli()
    log_root = pick_log_root(args.output_root)
    print(f"[phased_loss] Using log_root={log_root}")
    stamp = os.getenv("RUN_TAG_TS") or datetime.now().strftime("%Y%m%d-%H%M")
    print(f"[phased_loss] Using run timestamp: {stamp}")

    drive_root = Path(os.getenv("MODEL_B_DRIVE_ROOT",
                     "/content/drive/MyDrive/5703(hzha0521)/Optimized Model B (transformer based)"))
    candidates = [log_root / "b1_supcon_best.json",
                  drive_root / "logs/b1_supcon_best.json",
                  Path("logs/b1_supcon_best.json"),
                  Path(os.getenv("B1_JSON", "")) if os.getenv("B1_JSON") else None]
    best_json = next((p for p in candidates if p and p.exists()), None)
    if not best_json:
        raise SystemExit(f"[phased_loss] Missing b1_supcon_best.json in any of: {candidates}")
    obj = json.loads(best_json.read_text()); T, W = float(obj["T"]), float(obj["W"])
    print(f"[phased_loss] Using SupCon T={T}, W={W} from {best_json}")

    seeds = parse_seeds(args.seeds); seed_best = {}
    for s in seeds:
        tag_base = args.tag or f"phased_T{_fmt(T)}_W{_fmt(W)}_seed{s}"
        rec = ensure_full_run_seed(T, W, s, args.epochs, args.save_every, log_root, args.test, tag_base, stamp, args=args)
        if rec: seed_best[s] = rec

    if seed_best:
        summary = {
            "timestamp": stamp, "T": T, "W": W,
            "epochs": args.epochs, "save_every": args.save_every,
            "config": args.config, "data_root": DATA_ROOT,
            "seeds": {str(k): v for k, v in seed_best.items()},
            "mean": {k: safe_mean([r[k] for r in seed_best.values()]) for k in ["mAP","Rank-1","Rank-5","Rank-10"]},
            "std":  {k: safe_stdev([r[k] for r in seed_best.values()]) for k in ["mAP","Rank-1","Rank-5","Rank-10"]},
        }
        (log_root / "phased_loss_best_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[phased_loss] Summary written → {log_root/'phased_loss_best_summary.json'}")
    else:
        print("[phased_loss] No summary (test disabled or no metrics).")
    print("[phased_loss] Done.")

if __name__ == "__main__":
    main()
