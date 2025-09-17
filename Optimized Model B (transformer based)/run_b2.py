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
# Presence of ANY of these files under test_dir/epoch_X/ means "tested".
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

def pick_log_root() -> Path:
    """Pick a single root for logs/checkpoints/results: env > Colab Drive > ./logs."""
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
        str(Path.cwd().parents[1] / "datasets"),  # ../../datasets
        str(Path.cwd() / "datasets"),
    ]
    for c in candidates:
        if (Path(c) / "VeRi").exists():
            return c
    # Fallback (visible even if wrong)
    return str(Path.cwd().parents[1] / "datasets")

DATA_ROOT = detect_data_root()
print(f"[B2] Using DATASETS.ROOT_DIR={DATA_ROOT}")

# ---------- Small helpers ----------

def _fmt(v: float) -> str:
    return str(v).replace(".", "p")

def _re_pick(text: str, pat: str) -> float:
    m = re.search(pat, text, re.I)
    return float(m.group(1)) if m else -1.0

# ---------- Metrics parsing ----------

def parse_metrics_from_epoch_dir(epoch_dir: Path) -> Tuple[float, float, float, float]:
    """Return (mAP, Rank-1, Rank-5, Rank-10) from a single epoch_* directory."""
    sj = epoch_dir / "summary.json"
    if sj.exists():
        try:
            obj = json.loads(sj.read_text())
            mAP = float(obj.get("mAP", -1))
            r1  = float(obj.get("Rank-1", obj.get("Rank1", -1)))
            r5  = float(obj.get("Rank-5", obj.get("Rank5", -1)))
            r10 = float(obj.get("Rank-10", obj.get("Rank10", -1)))
            return mAP, r1, r5, r10
        except Exception:
            pass
    # Include test_log.txt as a valid source of metrics
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
    """Iterate epoch_* and return the best epoch by (mAP, Rank-1)."""
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


def safe_mean(values: List[float]) -> float:
    vals = [v for v in values if v >= 0]
    return round(stats.mean(vals), 4) if vals else -1.0


def safe_stdev(values: List[float]) -> float:
    vals = [v for v in values if v >= 0]
    return round(stats.stdev(vals), 4) if len(vals) >= 2 else 0.0

# ---------- Progress detection ----------

def _max_test_epoch(test_dir: Path) -> int:
    """Return max epoch index that has any result file; 0 if none."""
    max_ep = 0
    for ep in test_dir.glob("epoch_*"):
        if not ep.is_dir():
            continue
        try:
            idx = int(ep.name.split("_")[-1])
        except Exception:
            continue
        for f in TEST_MARKER_FILES_DEFAULT:
            if (ep / f).exists():
                max_ep = max(max_ep, idx)
                break
    return max_ep


def _max_epoch_from_checkpoints(run_dir: Path) -> int:
    """Scan common checkpoint names and return max epoch; 0 if none."""
    max_ep = 0
    if not run_dir.exists():
        return 0
    for p in run_dir.glob("state_*.pth"):
        m = re.search(r"state_(\d+)\.pth", p.name)
        if m:
            max_ep = max(max_ep, int(m.group(1)))
    for p in run_dir.glob("transformer_*.pth"):
        m = re.search(r"transformer_(\d+)\.pth", p.name)
        if m:
            max_ep = max(max_ep, int(m.group(1)))
    return max_ep


# ---------- Robust re-test (no reliance on eval-only flags) ----------

def eval_missing_epochs_via_test_py(tag: str, config_path: str, log_root: Path) -> None:
    """
    Reconstruct per-epoch test outputs by directly invoking test.py on every
    checkpoint under .../{tag}_deit_run, writing results to
    .../{tag}_deit_test/epoch_{E}. Only missing epochs are tested.
    """
    run_dir  = log_root / f"veri776_{tag}_deit_run"
    test_dir = log_root / f"veri776_{tag}_deit_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    rx = re.compile(r"transformer_(\d+)\.pth$")
    ckpts: List[Tuple[int, Path]] = []
    for p in run_dir.glob("transformer_*.pth"):
        m = rx.search(p.name)
        if m:
            ckpts.append((int(m.group(1)), p))
    ckpts.sort(key=lambda x: x[0])

    if not ckpts:
        print(f"[B2][eval] No checkpoints found under: {run_dir}")
        return

    print(f"[B2][diagnose] run_root={run_dir}")
    print(f"[B2][diagnose] test_root={test_dir}")

    for ep, ck in ckpts:
        out_ep = test_dir / f"epoch_{ep}"
        already = False
        if out_ep.exists():
            for name in TEST_MARKER_FILES_DEFAULT:
                if (out_ep / name).exists():
                    already = True
                    break
        if already:
            print(f"[B2][eval] Skip epoch {ep}: existing test marker(s) in {out_ep}")
            continue

        out_ep.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "test.py", "--config_file", str(config_path),
            "MODEL.DEVICE", "cuda",
            "TEST.WEIGHT", str(ck),
            "OUTPUT_DIR", str(out_ep),
            "DATASETS.ROOT_DIR", DATA_ROOT,
        ]
        print("[B2][eval] Launch:", " ".join(cmd))
        ret = subprocess.call(cmd)
        if ret != 0:
            print(f"[B2][eval][warn] test.py failed for epoch {ep} (ret={ret})")


# ---------- Idempotent per-seed runner (NO warm-start) ----------

def ensure_full_run_seed(
    T: float, W: float, seed: int, epochs: int,
    log_root: Path
) -> Optional[Dict[str, Any]]:
    """
    Ensure one per-seed long run is completed with results (no warm-start).

    Behavior:
      - If tests already cover 'epochs' -> skip train/test; pick best and return.
      - Else if checkpoints reached 'epochs' -> reconstruct missing tests via test.py; pick best.
      - Else -> (re)train with CHECKPOINT_PERIOD=1 & EVAL_PERIOD=1; enforce MAX_EPOCHS=epochs;
                then reconstruct missing tests and pick best.
    """
    tag = f"b2_with_b1best_T{_fmt(T)}_W{_fmt(W)}_seed{seed}"
    test_dir = log_root / f"veri776_{tag}_deit_test"
    run_dir  = log_root / f"veri776_{tag}_deit_run"

    # 1) Already tested to target epochs? → skip train/test
    tested_max = _max_test_epoch(test_dir)
    print(f"[B2][diagnose] tested_max={tested_max} under {test_dir}")
    if tested_max >= epochs:
        best_ep = pick_best_epoch_metrics(test_dir)
        if best_ep is None:
            print(f"[B2][warn] Already tested to {tested_max}, but cannot parse metrics under: {test_dir}")
            return None
        print(f"[B2] FULL ALREADY-TESTED tag={tag} (tested_max={tested_max}≥{epochs}) best_ep={best_ep}")
        return best_ep

    # 2) Training progress by checkpoints
    trained_max = _max_epoch_from_checkpoints(run_dir)
    print(f"[B2][diagnose] trained_max={trained_max} under {run_dir}")

    if trained_max >= epochs:
        print(f"[B2] FULL trained_max={trained_max}≥{epochs}; ensure tests exist for tag={tag}")
        eval_missing_epochs_via_test_py(tag, B2_CONFIG, log_root)
    else:
        # 3) Need to train/resume — always ImageNet init, and cap at FULL_EPOCHS
        cmd = [
            sys.executable, "run_modelB_deit.py",
            "--config", B2_CONFIG,
            "--opts",
            # SupCon inherited from B1 JSON
            "LOSS.SUPCON.ENABLE", "True",
            "LOSS.SUPCON.T", str(T),
            "LOSS.SUPCON.W", str(W),
            # TripletX and training mode
            "LOSS.TRIPLETX.ENABLE", "True",
            "MODEL.TRAINING_MODE", "supervised",
            # Run settings (strictly cap epochs)
            "SOLVER.MAX_EPOCHS", str(epochs),
            "SOLVER.SEED", str(seed),
            "SOLVER.CHECKPOINT_PERIOD", "1",
            "SOLVER.EVAL_PERIOD", "1",
            # Paths
            "DATASETS.ROOT_DIR", DATA_ROOT,
            "OUTPUT_DIR", str(log_root),
            "TAG", tag,
        ]
        # NOTE: We intentionally do NOT add MODEL.PRETRAIN_CHOICE/self here.
        # The YAML keeps PRETRAIN_CHOICE=imagenet → always ImageNet init.

        print("[B2] Launch:", " ".join(cmd))
        subprocess.check_call(cmd)

        # Reconstruct any missing test epochs after training (idempotent)
        eval_missing_epochs_via_test_py(tag, B2_CONFIG, log_root)

    # 4) Pick best epoch metrics after ensuring tests exist
    best_ep = pick_best_epoch_metrics(test_dir)
    if best_ep is None:
        print(f"[B2][warn] No valid epoch metrics under: {test_dir}")
        return None
    print(f"[B2] Seed {seed} best epoch → {best_ep}")
    return best_ep


# ---------- Main ----------

def main():
    log_root = pick_log_root()
    log_root.mkdir(parents=True, exist_ok=True)
    print(f"[B2] Using log_root={log_root}")

    # Load best (T, W) strictly from B1 output
    best_json = log_root / "b1_supcon_best.json"
    if not best_json.exists():
        raise SystemExit(f"[B2] Missing {best_json}. Run B1 search first.")

    try:
        obj = json.loads(best_json.read_text())
        T = float(obj["T"])
        W = float(obj["W"])
    except Exception as e:
        raise SystemExit(f"[B2] Failed to parse {best_json}: {e}")

    # Per-seed execution (idempotent & seamless). No warm-start.
    seed_best_records: Dict[int, Dict[str, Any]] = {}
    for seed in SEEDS:
        rec = ensure_full_run_seed(T, W, seed, FULL_EPOCHS, log_root)
        if rec is not None:
            seed_best_records[seed] = rec

    # Aggregate mean/std across seeds
    summary_path = log_root / "b2_supcon_best_summary.json"
    if seed_best_records:
        mAPs  = [rec["mAP"] for rec in seed_best_records.values()]
        R1s   = [rec["Rank-1"] for rec in seed_best_records.values()]
        R5s   = [rec["Rank-5"] for rec in seed_best_records.values()]
        R10s  = [rec["Rank-10"] for rec in seed_best_records.values()]
        summary = {
            "T": T, "W": W, "config": B2_CONFIG, "epochs": FULL_EPOCHS,
            "data_root": DATA_ROOT,
            "seeds": {str(k): v for k, v in seed_best_records.items()},
            "mean": {
                "mAP":  safe_mean(mAPs),
                "Rank-1": safe_mean(R1s),
                "Rank-5": safe_mean(R5s),
                "Rank-10": safe_mean(R10s),
            },
            "std": {
                "mAP":  safe_stdev(mAPs),
                "Rank-1": safe_stdev(R1s),
                "Rank-5": safe_stdev(R5s),
                "Rank-10": safe_stdev(R10s),
            },
            "note": "B2_no-warmstart(ImageNet). Best epoch per seed picked by mAP (tiebreaker Rank-1).",
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"[B2] Wrote summary → {summary_path}")
    else:
        print("[B2][warn] No seed best records collected; skip summary.")

    print("[B2] Done.")


if __name__ == "__main__":
    main()
