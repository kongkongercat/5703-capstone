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
# =============================================================================
"""
What it does:
- Picks a log root that works both locally and in Colab (Google Drive).
- Loads (T, W) from logs/b1_supcon_best.json written by run_b1.py.
- Launches run_modelB_deit.py for B2 with seeds=[0,1,2], passing ENABLE/T/W and OUTPUT_DIR.
- After training, for each seed, finds the best epoch (by mAP, tie Rank-1).
- Writes aggregated mean/std across seeds to logs/b2_supcon_best_summary.json.
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
    """
    Decide a single root for logs/checkpoints/results:
    1) OUTPUT_ROOT env → use it directly.
    2) If in Colab and Drive is mounted → use a Drive folder.
    3) Otherwise → use ./logs locally.
    """
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

def _fmt(v: float) -> str:
    return str(v).replace(".", "p")


# ---------- Parsing helpers ----------
def _re_pick(text: str, pat: str) -> float:
    m = re.search(pat, text, re.I)
    return float(m.group(1)) if m else -1.0

def _re_pick_last(text: str, pat: str) -> float:
    mm = list(re.finditer(pat, text, re.I))
    return float(mm[-1].group(1)) if mm else -1.0

def parse_metrics_from_epoch_dir(epoch_dir: Path) -> Tuple[float, float, float, float]:
    """
    Read metrics from a single epoch_* directory.
    Priority: summary.json; fallback: test_summary.txt/results.txt/log.txt.
    Returns (mAP, Rank-1, Rank-5, Rank-10) with -1.0 if missing.
    """
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
    """
    Iterate all epoch_* directories under test_dir, pick the epoch with the best mAP.
    Tiebreaker: higher Rank-1.
    Returns dict: {"epoch": int, "mAP": float, "Rank-1": float, "Rank-5": float, "Rank-10": float}
    or None if not found.
    """
    epochs = sorted(
        [p for p in test_dir.glob("epoch_*") if p.is_dir()],
        key=lambda p: int(p.name.split("_")[-1])
    )
    best = None
    best_key = None
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


# ---------- Main ----------
def main():
    log_root = pick_log_root()
    log_root.mkdir(parents=True, exist_ok=True)
    print(f"[B2] Using log_root={log_root}")

    # Strictly load best (T, W) from JSON (no fallback).
    best_json = log_root / "b1_supcon_best.json"
    if not best_json.exists():
        raise SystemExit(f"[B2] Missing {best_json}. Run B1 search first to create it.")

    try:
        obj = json.loads(best_json.read_text())
        T = float(obj["T"])
        W = float(obj["W"])
    except Exception as e:
        raise SystemExit(f"[B2] Failed to parse {best_json}: {e}")

    tag_base = f"b2_with_b1best_T{_fmt(T)}_W{_fmt(W)}"
    print(f"[B2] (T,W)=({T},{W})  → tag base: {tag_base}")
    seed_best_records: Dict[int, Dict[str, Any]] = {}

    # Launch 3 seeds
    for seed in SEEDS:
        tag = f"{tag_base}_seed{seed}"
        cmd = [
            sys.executable, "run_modelB_deit.py",
            "--config", B2_CONFIG,
            "--opts",
            "LOSS.SUPCON.ENABLE", "True",   # turn on SupCon inside B2
            "LOSS.SUPCON.T", str(T),
            "LOSS.SUPCON.W", str(W),
            "SOLVER.MAX_EPOCHS", str(FULL_EPOCHS),
            "SOLVER.SEED", str(seed),
            "OUTPUT_DIR", str(log_root),
            "TAG", tag,
        ]
        print("[B2] Launch:", " ".join(cmd))
        subprocess.check_call(cmd)

        # After training+per-epoch testing, pick best epoch under *_deit_test
        test_dir = log_root / f"veri776_{tag}_deit_test"
        best_ep = pick_best_epoch_metrics(test_dir)
        if best_ep is None:
            print(f"[B2][warn] No valid epoch metrics found under: {test_dir}")
        else:
            seed_best_records[seed] = best_ep
            print(f"[B2] Seed {seed} best epoch: {best_ep}")

    # Aggregate mean/std across seeds and save summary JSON
    summary_path = log_root / "b2_supcon_best_summary.json"
    if seed_best_records:
        mAPs =  [rec["mAP"] for rec in seed_best_records.values()]
        R1s  =  [rec["Rank-1"] for rec in seed_best_records.values()]
        R5s  =  [rec["Rank-5"] for rec in seed_best_records.values()]
        R10s =  [rec["Rank-10"] for rec in seed_best_records.values()]

        summary = {
            "T": T,
            "W": W,
            "config": B2_CONFIG,
            "epochs": FULL_EPOCHS,
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
