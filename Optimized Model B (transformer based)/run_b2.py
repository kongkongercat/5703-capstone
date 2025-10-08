#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# B2 launcher — auto-inherit SupCon (T, W) from B1, run 3 seeds and summarize.
# =============================================================================
# Change Log
# [2025-09-14 | Hang Zhang] Initial launcher: read T/W from logs/b1_supcon_best.json,
#                            explicitly enable SupCon via --opts, unify OUTPUT_DIR
#                            (env OUTPUT_ROOT > Colab Drive > ./logs), no fallback scan.
# [2025-09-15 | Hang Zhang] Add seeds=[0,1,2] long training; per-seed best epoch
#                            selection (by mAP, tiebreaker Rank-1); summary
#                            aggregation to logs/b2_supcon_best_summary.json.
# [2025-09-15 | Hang Zhang] Parse full metrics (mAP/Rank-1/Rank-5/Rank-10)
#                            from summary.json or logs via regex.
# [2025-09-16 | Hang Zhang] NEW: Seamless run (auto progress detection + resume + test guard)
#                            - Skip/resume per seed based on trained/tested progress
#                            - If training finished but no results, auto eval-only
# [2025-09-16 | Hang Zhang] Replace eval-only guessing with robust re-test:
#                            - NEW eval_missing_epochs_via_test_py(): iterate checkpoints
#                              and call test.py only for epochs missing outputs.
#                            - Force CHECKPOINT_PERIOD=1 & EVAL_PERIOD=1 on new runs.
#                            - Keep English code comments; naming aligned with run_b1.py.
# [2025-09-16 | Hang Zhang] NEW: pass DATASETS.ROOT_DIR everywhere
#                            - detect_data_root(): auto-detect VeRi dataset root
#                            - add DATASETS.ROOT_DIR to all run/test invocations
# [2025-09-17 | Hang Zhang] FIX: recognize existing test artifacts correctly:
#                            - Treat `test_log.txt` and `dist_mat.npy` as valid markers
#                              for "epoch tested" to avoid re-testing from epoch 1.
#                            - Add diagnostics for tested_max / paths.
# [2025-09-18 | Hang Zhang] FIX: auto-detect trained_max from checkpoints and
#                            force override SOLVER.MAX_EPOCHS to FULL_EPOCHS on resume,
#                            preventing checkpoint cfg (e.g., 120) from overriding.
# [2025-09-18 | Hang Zhang] POLICY: Explicitly DISABLE warm-start from B1.
#                            - B2 now always initializes from ImageNet
#                              (B2_no-warmstart), to compare fairly with
#                              B1_full(ImageNet). Removed all b1_weight logic.
# [2025-10-08 | Hang Zhang] EXTENDED: CLI argument support
#                            - Added --test/--no-test, --epochs, --seeds, --output-root, --tag.
#                            - Allows flexible run configuration without editing source.
#                            - Default behavior identical to seamless B2_no-warmstart(ImageNet).
# =============================================================================
"""
B2_no-warmstart(ImageNet):
- Use the best (T, W) from B1 JSON only.
- Always initialize from ImageNet (no warm-start from B1 ckpt).
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
B2_CONFIG    = "configs/VeiRi/deit_transreid_stride_b2_supcon_tripletx.yml".replace("VeiRi","VeRi")
FULL_EPOCHS  = 30
SEEDS        = [0, 1, 2]

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
    1) CLI argument
    2) Environment variable OUTPUT_ROOT
    3) Colab Drive path
    4) ./logs
    """
    if cli_output_root:
        return Path(cli_output_root)
    env = os.getenv("OUTPUT_ROOT")
    if env:
        return Path(env)
    if _in_colab():
        try:
            from google.colab import drive  # noqa: F401
            if not Path("/content/drive/MyDrive").exists():
                drive.mount("/content/drive", force_remount=False)
        except Exception:
            pass
        dflt = "/content/drive/MyDrive/5703(hzha0521)/Optimized Model B (transformer based)/logs"
        return Path(os.getenv("DRIVE_LOG_ROOT", dflt))
    return Path("logs")

# ---------- Dataset root detection ----------

def detect_data_root() -> str:
    """Return a directory path whose subfolder 'VeRi' exists."""
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
print(f"[B2] Using DATASETS.ROOT_DIR={DATA_ROOT}")

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
    p = argparse.ArgumentParser(description="Run B2_no-warmstart with flexible options.")
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
        if not ep.is_dir(): continue
        try: idx = int(ep.name.split("_")[-1])
        except: continue
        for f in TEST_MARKER_FILES_DEFAULT:
            if (ep / f).exists():
                max_ep = max(max_ep, idx)
                break
    return max_ep

def _max_epoch_from_checkpoints(run_dir: Path) -> int:
    max_ep = 0
    if not run_dir.exists(): return 0
    for p in run_dir.glob("transformer_*.pth"):
        m = re.search(r"transformer_(\d+)\.pth", p.name)
        if m: max_ep = max(max_ep, int(m.group(1)))
    return max_ep

def eval_missing_epochs_via_test_py(tag: str, config: str, log_root: Path):
    run_dir = log_root / f"veri776_{tag}_deit_run"
    test_dir = log_root / f"veri776_{tag}_deit_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    ckpts = sorted(run_dir.glob("transformer_*.pth"), key=lambda p: int(re.search(r"(\d+)", p.name).group(1)))
    if not ckpts:
        print(f"[B2][eval] No checkpoints under {run_dir}")
        return
    for ck in ckpts:
        ep = int(re.search(r"(\d+)", ck.name).group(1))
        out_ep = test_dir / f"epoch_{ep}"
        if any((out_ep / f).exists() for f in TEST_MARKER_FILES_DEFAULT):
            print(f"[B2][eval] Skip epoch {ep} (already tested)")
            continue
        out_ep.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "test.py", "--config_file", config,
            "MODEL.DEVICE", "cuda",
            "TEST.WEIGHT", str(ck),
            "OUTPUT_DIR", str(out_ep),
            "DATASETS.ROOT_DIR", DATA_ROOT,
        ]
        print("[B2][eval] Launch:", " ".join(cmd))
        subprocess.call(cmd)

# ---------- Seed runner ----------

def ensure_full_run_seed(T, W, seed, epochs, log_root, test_enabled, tag=None):
    tag = tag or f"b2_with_b1best_T{_fmt(T)}_W{_fmt(W)}_seed{seed}"
    run_dir = log_root / f"veri776_{tag}_deit_run"
    test_dir = log_root / f"veri776_{tag}_deit_test"

    trained_max = _max_epoch_from_checkpoints(run_dir)
    tested_max = _max_test_epoch(test_dir) if test_enabled else 0
    print(f"[B2][diagnose] seed={seed} trained_max={trained_max} tested_max={tested_max}")

    if test_enabled and tested_max >= epochs:
        best_ep = pick_best_epoch_metrics(test_dir)
        if best_ep:
            print(f"[B2] Already tested up to {tested_max}. Best={best_ep}")
            return best_ep

    if trained_max < epochs:
        cmd = [
            sys.executable, "run_modelB_deit.py",
            "--config", B2_CONFIG,
            "--opts",
            "LOSS.SUPCON.ENABLE", "True",
            "LOSS.SUPCON.T", str(T),
            "LOSS.SUPCON.W", str(W),
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
        print("[B2] Launch:", " ".join(cmd))
        subprocess.check_call(cmd)

    if not test_enabled:
        print(f"[B2] Seed {seed} training done (test skipped).")
        return None

    eval_missing_epochs_via_test_py(tag, B2_CONFIG, log_root)
    best_ep = pick_best_epoch_metrics(test_dir)
    if not best_ep:
        print(f"[B2][warn] No metrics found under {test_dir}")
        return None
    print(f"[B2] Seed {seed} best epoch → {best_ep}")
    return best_ep

# ---------- Main ----------

def main():
    args = build_cli()
    log_root = pick_log_root(args.output_root)
    log_root.mkdir(parents=True, exist_ok=True)
    print(f"[B2] Using log_root={log_root}")

    best_json = log_root / "b1_supcon_best.json"
    if not best_json.exists():
        raise SystemExit(f"[B2] Missing {best_json}")
    obj = json.loads(best_json.read_text())
    T, W = float(obj["T"]), float(obj["W"])

    seeds = parse_seeds(args.seeds)
    seed_best = {}
    for s in seeds:
        rec = ensure_full_run_seed(T, W, s, args.epochs, log_root, args.test, args.tag)
        if rec: seed_best[s] = rec

    if seed_best:
        summary = {
            "T": T, "W": W, "epochs": args.epochs, "config": B2_CONFIG,
            "data_root": DATA_ROOT,
            "seeds": {str(k): v for k, v in seed_best.items()},
            "mean": {k: safe_mean([r[k] for r in seed_best.values()]) for k in ["mAP","Rank-1","Rank-5","Rank-10"]},
            "std":  {k: safe_stdev([r[k] for r in seed_best.values()]) for k in ["mAP","Rank-1","Rank-5","Rank-10"]},
        }
        (log_root / "b2_supcon_best_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[B2] Summary written → {log_root/'b2_supcon_best_summary.json'}")
    else:
        print("[B2] No summary (test disabled or no metrics).")
    print("[B2] Done.")

if __name__ == "__main__":
    main()
