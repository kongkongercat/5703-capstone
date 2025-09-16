#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# B2 launcher — auto-inherit SupCon (T, W) from B1, run 3 seeds and summarize.
#
# Change Log
# [2025-09-14 | Hang Zhang] Initial launcher: read T/W from logs/b1_supcon_best.json,
#                            explicitly enable SupCon via --opts, unify OUTPUT_DIR
#                            (env OUTPUT_ROOT > Colab Drive > ./logs), no fallback scan.
# [2025-09-15 | Hang Zhang] Add seeds=[0,1,2] long training; per-seed best epoch
#                            selection (by mAP, tiebreaker Rank-1); summary
#                            aggregation to logs/b2_supcon_best_summary.json.
# [2025-09-15 | Hang Zhang] Parse full metrics (mAP/Rank-1/Rank-5/Rank-10)
#                            from summary.json or logs via regex.
# [2025-09-16 | Hang Zhang] NEW: warm-start from B1 best (transformer_12.pth) if available
#                            (MODEL.PRETRAIN_CHOICE=self, not resume); force-enable
#                            TripletX; set TRAINING_MODE=supervised explicitly.
# [2025-09-16 | Hang Zhang] NEW: Seamless run (auto progress detection + resume + test guard)
#                            - Skip/resume per seed based on trained/tested progress
#                            - If training finished but no results, auto eval-only
#                            - Warm-start only when *no progress* exists
# [2025-09-16 | ChatGPT   ] Replace eval-only guessing with robust re-test:
#                            - NEW eval_missing_epochs_via_test_py(): iterate checkpoints
#                              and call test.py only for epochs missing outputs.
#                            - Force CHECKPOINT_PERIOD=1 & EVAL_PERIOD=1 on new runs.
#                            - Keep English code comments; naming aligned with run_b1.py.
# [2025-09-16 | ChatGPT   ] NEW: pass DATASETS.ROOT_DIR everywhere
#                            - detect_data_root(): auto-detect VeRi dataset root
#                            - add DATASETS.ROOT_DIR to all run/test invocations
# =============================================================================
"""
B2: Use the best (T, W) from B1 to run long training with seeds=[0,1,2],
pick best epoch per seed (by mAP, tie Rank-1), then aggregate mean/std.

Seamless behavior (mirrors B1):
- Per-seed run is idempotent:
  * If tests already cover FULL_EPOCHS -> skip training/testing; parse directly.
  * Else if checkpoints reached FULL_EPOCHS -> reconstruct missing tests via test.py.
  * Else -> (re)train with CHECKPOINT_PERIOD=1 & EVAL_PERIOD=1, then reconstruct tests.
- Warm-start only when there is zero progress and B1's best ckpt is found.
"""

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
# ---------------------------

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
    """
    Return a directory path whose subfolder 'VeRi' exists.
    Priority: $DATASETS_ROOT > common Colab/Drive locations > ../../datasets
    """
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
    # Fallback (可能仍会失败，但至少可见报错路径)
    return str(Path.cwd().parents[1] / "datasets")

DATA_ROOT = detect_data_root()
print(f"[B2] Using DATASETS.ROOT_DIR={DATA_ROOT}")

def _fmt(v: float) -> str:
    return str(v).replace(".", "p")

# ---------- Metrics parsing ----------
def _re_pick(text: str, pat: str) -> float:
    m = re.search(pat, text, re.I)
    return float(m.group(1)) if m else -1.0

def _re_pick_last(text: str, pat: str) -> float:
    mm = list(re.finditer(pat, text, re.I))
    return float(mm[-1].group(1)) if mm else -1.0

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
    for name in ("test_summary.txt", "results.txt", "log.txt"):
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
        if (ep / "summary.json").exists():
            max_ep = max(max_ep, idx); continue
        for f in ("test_summary.txt", "results.txt", "log.txt"):
            if (ep / f).exists():
                max_ep = max(max_ep, idx); break
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

def has_any_epoch_results(test_dir: Path) -> bool:
    """Any epoch_* result file or aggregated log present?"""
    if _max_test_epoch(test_dir) > 0:
        return True
    if (test_dir / "test_all.log").exists():
        return True
    return False

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

    for ep, ck in ckpts:
        out_ep = test_dir / f"epoch_{ep}"
        if (out_ep / "summary.json").exists():
            continue
        if any((out_ep / name).exists() for name in ("test_summary.txt", "results.txt", "log.txt")):
            continue

        out_ep.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "test.py", "--config_file", str(config_path),
            "MODEL.DEVICE", "cuda",
            "TEST.WEIGHT", str(ck),
            "OUTPUT_DIR", str(out_ep),
            "DATASETS.ROOT_DIR", DATA_ROOT,  # <-- critical override
        ]
        print("[B2][eval] Launch:", " ".join(cmd))
        ret = subprocess.call(cmd)
        if ret != 0:
            print(f"[B2][eval][warn] test.py failed for epoch {ep} (ret={ret})")

# ---------- Idempotent per-seed runner ----------
def ensure_full_run_seed(
    T: float, W: float, seed: int, epochs: int,
    log_root: Path, b1_weight: Optional[Path], use_warmstart_if_empty: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Ensure one per-seed long run is completed with results.

    Behavior:
      - If tests already cover 'epochs' -> skip train/test; pick best and return.
      - Else if checkpoints reached 'epochs' -> reconstruct missing tests via test.py; pick best.
      - Else -> (re)train with CHECKPOINT_PERIOD=1 & EVAL_PERIOD=1 (resume by TAG);
                if zero progress and b1_weight available, warm-start (not resume);
                then reconstruct any missing tests and pick best.
    """
    tag = f"b2_with_b1best_T{_fmt(T)}_W{_fmt(W)}_seed{seed}"
    test_dir = log_root / f"veri776_{tag}_deit_test"
    run_dir  = log_root / f"veri776_{tag}_deit_run"

    # 1) Already tested to target epochs? → skip train/test
    tested_max = _max_test_epoch(test_dir)
    if tested_max >= epochs:
        best_ep = pick_best_epoch_metrics(test_dir)
        if best_ep is None:
            print(f"[B2][warn] Already tested to {tested_max}, but cannot parse metrics under: {test_dir}")
            return None
        print(f"[B2] FULL ALREADY-TESTED tag={tag} (tested_max={tested_max}≥{epochs}) best_ep={best_ep}")
        return best_ep

    # 2) Training progress by checkpoints
    trained_max = _max_epoch_from_checkpoints(run_dir)
    if trained_max >= epochs:
        print(f"[B2] FULL trained_max={trained_max}≥{epochs}; ensure tests exist for tag={tag}")
        eval_missing_epochs_via_test_py(tag, B2_CONFIG, log_root)
    else:
        # 3) Need to train/resume
        cmd = [
            sys.executable, "run_modelB_deit.py",
            "--config", B2_CONFIG,
            "--opts",
            # SupCon inherited from B1
            "LOSS.SUPCON.ENABLE", "True",
            "LOSS.SUPCON.T", str(T),
            "LOSS.SUPCON.W", str(W),
            # TripletX and training mode
            "LOSS.TRIPLETX.ENABLE", "True",
            "MODEL.TRAINING_MODE", "supervised",
            # Run settings
            "SOLVER.MAX_EPOCHS", str(epochs),
            "SOLVER.SEED", str(seed),
            "SOLVER.CHECKPOINT_PERIOD", "1",
            "SOLVER.EVAL_PERIOD", "1",
            "DATASETS.ROOT_DIR", DATA_ROOT,  # <-- critical override for training/eval
            "OUTPUT_DIR", str(log_root),
            "TAG", tag,
        ]

        # Warm-start only when zero progress and weight exists
        if trained_max == 0 and tested_max == 0 and use_warmstart_if_empty and b1_weight is not None:
            print(f"[B2] First run (no progress). Warm-start from B1 best: {b1_weight}")
            cmd += [
                "MODEL.PRETRAIN_CHOICE", "self",
                "MODEL.PRETRAIN_PATH", str(b1_weight),
            ]
        else:
            print(f"[B2] Resume/continue run: tag={tag} (trained_max={trained_max}/{epochs})")
            # If your trainer needs an explicit resume flag, you may add it here:
            # cmd += ["SOLVER.RESUME", "True"]

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

    # Locate B1 best weight for optional warm-start (transformer_12.pth)
    b1_weight: Optional[Path] = None
    try:
        run_dir_str = json.loads(best_json.read_text()).get("run_dir", "")
        if run_dir_str:
            cand = Path(run_dir_str) / "transformer_12.pth"
            if cand.exists():
                b1_weight = cand
        if b1_weight is None:
            src_tag = obj.get("source_tag")
            if src_tag:
                guess = log_root / f"veri776_{src_tag}_deit_run" / "transformer_12.pth"
                if guess.exists():
                    b1_weight = guess
        if b1_weight:
            print(f"[B2] Warm-start weight found: {b1_weight}")
        else:
            print("[B2] No warm-start weight found; will resume/continue if progress exists.")
    except Exception as e:
        print(f"[B2][warn] Failed to locate B1 weight: {e}")

    # Per-seed execution (idempotent & seamless)
    seed_best_records: Dict[int, Dict[str, Any]] = {}
    for seed in SEEDS:
        rec = ensure_full_run_seed(T, W, seed, FULL_EPOCHS, log_root, b1_weight)
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
            "note": "Best epoch per seed picked by mAP (tiebreaker Rank-1). "
                    "Means/std computed over seeds with valid metrics."
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"[B2] Wrote summary → {summary_path}")
    else:
        print("[B2][warn] No seed best records collected; skip summary.")

    print("[B2] Done.")

if __name__ == "__main__":
    main()
