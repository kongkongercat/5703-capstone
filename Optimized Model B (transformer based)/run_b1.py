#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================
# File: run_b1.py
# Purpose: B1 SupCon grid search with YAML-driven grid (LOSS.SUPCON.SEARCH)
# Author: Hang Zhang (hzha0521)
# ==========================================
# Change Log
# [2025-09-14 | Hang Zhang] Initial version: load grid from YAML; quick screen;
#                            parse latest epoch_*; save best; retrain full.
# [2025-09-15 | Hang Zhang] Ensure SupCon is explicitly enabled via --opts
#                            (add LOSS.SUPCON.ENABLE True in all launches).
# [2025-09-15 | Hang Zhang] New: Search uses a single seed (seed=0) for speed;
#                            Full retrain runs 3 seeds (0/1/2) with seed in tag.
# [2025-09-15 | Hang Zhang] Add parsing of Rank-5/Rank-10 (summary.json/log);
#                            save full metrics in JSON and stdout.
# [2025-09-15 | Hang Zhang] Add b1_supcon_best_summary.json:
#                            pick best epoch per seed, then compute mean/std.
# ==========================================
"""
B1 SupCon grid search (minimal) with grid loaded from YAML (LOSS.SUPCON.SEARCH).

What it does:
- Reads T/W candidates from the experiment YAML: LOSS.SUPCON.SEARCH.{T,W}
- Loops over (T, W), runs short training to screen (single-seed, fast)
- Parses metrics (mAP / Rank-1 / Rank-5 / Rank-10) from latest epoch_* subfolder
- Saves the best combo to logs/b1_supcon_best.json
- Retrains the best combo for FULL_EPOCHS with seeds = [0, 1, 2]
- Aggregates the 3 seeds: pick best epoch per seed, compute mean/std, and save to
  logs/b1_supcon_best_summary.json

Notes:
- Uses a local/Colab-aware log root; override via env OUTPUT_ROOT if desired.
- If SEARCH.{T,W} is missing in YAML, falls back to sensible defaults.
"""

import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import statistics as stats

# ====== User settings ======
CONFIG = "configs/VeiRi/deit_transreid_stride_b1_supcon.yml".replace("VeiRi", "VeRi")
SEARCH_EPOCHS = 12        # 12–15 recommended for quick screening
FULL_EPOCHS   = 30        # long training for the best (T,W)
# Seeds policy
SEARCH_SEED = 0           # fast screening: single seed only
FULL_SEEDS  = [0, 1, 2]   # confirm best with 3 seeds
# ===========================

# ---------- Env helpers ----------
def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False

def pick_log_root() -> Path:
    """Choose one root for logs/results, same rule as other model runners."""
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

LOG_ROOT = pick_log_root()
LOG_ROOT.mkdir(parents=True, exist_ok=True)
BEST_JSON = LOG_ROOT / "b1_supcon_best.json"
BEST_SUMMARY_JSON = LOG_ROOT / "b1_supcon_best_summary.json"

def _fmt(v: float) -> str:
    return str(v).replace(".", "p")

# ---------- Load grid from YAML (preferred), fallback to defaults ----------
def _to_float_list(x) -> List[float]:
    if isinstance(x, (int, float)):
        return [float(x)]
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    return []

def load_grid_from_yaml(cfg_path: str) -> Tuple[List[float], List[float]]:
    """
    Try YACS first (if available), then PyYAML. Return (T_list, W_list).
    Fallback to T=[0.07], W=[0.25,0.30,0.35] if not found.
    """
    # Attempt via YACS (robust for full self-contained YAML)
    try:
        from yacs.config import CfgNode as CN
        try:
            cfg = CN(new_allowed=True)
        except TypeError:
            cfg = CN()
            if hasattr(cfg, "set_new_allowed"):
                cfg.set_new_allowed(True)
        cfg.merge_from_file(cfg_path)

        loss = getattr(cfg, "LOSS", None)
        supc = getattr(loss, "SUPCON", None) if loss is not None else None
        search = getattr(supc, "SEARCH", None) if supc is not None else None
        if search is not None:
            t_list = _to_float_list(getattr(search, "T", []))
            w_list = _to_float_list(getattr(search, "W", []))
            if t_list and w_list:
                return sorted(set(t_list)), sorted(set(w_list))
    except Exception:
        pass

    # Attempt via PyYAML (if installed)
    try:
        import yaml  # type: ignore
        with open(cfg_path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        loss = y.get("LOSS", {}) or {}
        supc = loss.get("SUPCON", {}) or {}
        search = supc.get("SEARCH", {}) or {}
        t_list = _to_float_list(search.get("T"))
        w_list = _to_float_list(search.get("W"))
        if t_list and w_list:
            return sorted(set(t_list)), sorted(set(w_list))
    except Exception:
        pass

    # Fallback defaults (your current common choices)
    return [0.07], [0.25, 0.30, 0.35]

# ---------- Parse metrics (latest-epoch helper) ----------
def _latest_epoch_dir(out_test_dir: Path) -> Optional[Path]:
    """Pick the newest epoch_* subfolder inside out_test_dir."""
    epochs = sorted(
        [p for p in out_test_dir.glob("epoch_*") if p.is_dir()],
        key=lambda p: int(p.name.split("_")[-1])
    )
    return epochs[-1] if epochs else None

def _re_pick(text: str, pat: str) -> float:
    m = re.search(pat, text, re.I)
    return float(m.group(1)) if m else -1.0

def _re_pick_last(text: str, pat: str) -> float:
    mm = list(re.finditer(pat, text, re.I))
    return float(mm[-1].group(1)) if mm else -1.0

def parse_metrics(out_test_dir: Path) -> Tuple[float, float, float, float]:
    """
    Return (mAP, Rank-1, Rank-5, Rank-10) from latest epoch_* subfolder.
    Priority: epoch_*/summary.json ; fallback: test_all.log by regex.
    Missing values return -1.0.
    """
    ep = _latest_epoch_dir(out_test_dir)
    # ---- Try summary.json first ----
    if ep:
        sj = ep / "summary.json"
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
        # common text fallbacks inside epoch dir
        for name in ("test_summary.txt", "results.txt", "log.txt"):
            p = ep / name
            if p.exists():
                s = p.read_text(encoding="utf-8", errors="ignore")
                mAP = _re_pick(s, r"mAP[^0-9]*([0-9]+(?:\.[0-9]+)?)")
                r1  = _re_pick(s, r"Rank[-\s]?1[^0-9]*([0-9]+(?:\.[0-9]+)?)")
                r5  = _re_pick(s, r"Rank[-\s]?5[^0-9]*([0-9]+(?:\.[0-9]+)?)")
                r10 = _re_pick(s, r"Rank[-\s]?10[^0-9]*([0-9]+(?:\.[0-9]+)?)")
                return mAP, r1, r5, r10

    # ---- Fallback: aggregated test_all.log ----
    all_log = out_test_dir / "test_all.log"
    if all_log.exists():
        s = all_log.read_text(encoding="utf-8", errors="ignore")
        mAP = _re_pick_last(s, r"mAP[^0-9]*([0-9]+(?:\.[0-9]+)?)")
        r1  = _re_pick_last(s, r"Rank[-\s]?1[^0-9]*([0-9]+(?:\.[0-9]+)?)")
        r5  = _re_pick_last(s, r"Rank[-\s]?5[^0-9]*([0-9]+(?:\.[0-9]+)?)")
        r10 = _re_pick_last(s, r"Rank[-\s]?10[^0-9]*([0-9]+(?:\.[0-9]+)?)")
        return mAP, r1, r5, r10

    return -1.0, -1.0, -1.0, -1.0

# ---------- Per-epoch parsing & aggregation for summary ----------
def parse_metrics_from_epoch_dir(epoch_dir: Path) -> Tuple[float, float, float, float]:
    """
    Read metrics from a single epoch_* directory.
    Priority: summary.json; fallback to text files (test_summary.txt/results.txt/log.txt).
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
    best_key = None  # tuple for comparison
    for ep in epochs:
        ep_idx = int(ep.name.split("_")[-1])
        mAP, r1, r5, r10 = parse_metrics_from_epoch_dir(ep)
        if mAP < 0:
            continue
        key = (mAP, r1)  # primary mAP, secondary Rank-1
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

# ---------- One run (search, single-seed) ----------
def run_one_search(t: float, w: float):
    """
    Run one (T,W) combo for SEARCH_EPOCHS with SEARCH_SEED only.
    Tag does NOT carry seed suffix to keep compatibility with older logs,
    but seed is passed via SOLVER.SEED.
    """
    tag = f"b1_supcon_T{_fmt(t)}_W{_fmt(w)}"
    subprocess.check_call([
        sys.executable, "run_modelB_deit.py",
        "--config", CONFIG,
        "--opts",
        "LOSS.SUPCON.ENABLE", "True",   # ensure SupCon is ON
        "LOSS.SUPCON.T", str(t),
        "LOSS.SUPCON.W", str(w),
        "SOLVER.MAX_EPOCHS", str(SEARCH_EPOCHS),
        "SOLVER.SEED", str(SEARCH_SEED),   # ← single seed for search
        "OUTPUT_DIR", str(LOG_ROOT),
        "TAG", tag,
    ])
    # Must match run_modelB_deit.py naming: veri776_{tag}_deit_test
    out_test_dir = LOG_ROOT / f"veri776_{tag}_deit_test"
    mAP, r1, r5, r10 = parse_metrics(out_test_dir)
    print(f"[B1] SEARCH RESULT tag={tag} T={t} W={w} seed={SEARCH_SEED} "
          f"mAP={mAP} Rank-1={r1} Rank-5={r5} Rank-10={r10}")
    return {"tag": tag, "T": t, "W": w, "mAP": mAP, "R1": r1, "R5": r5, "R10": r10}

# ---------- Main ----------
def main():
    # 1) Load grid from YAML (fallback to defaults if missing)
    T_list, W_list = load_grid_from_yaml(CONFIG)
    print(f"[B1] Search grid → T={T_list} ; W={W_list}")
    print(f"[B1] Search uses single seed: {SEARCH_SEED}")
    print(f"[B1] Full training seeds   : {FULL_SEEDS}")

    # 2) Sweep grid (single-seed) and collect metrics
    runs = [run_one_search(t, w) for t, w in itertools.product(T_list, W_list)]
    if not runs:
        raise SystemExit("[B1] No runs executed. Check your SEARCH grid or paths.")

    # 3) Select best by (mAP, Rank-1)
    best = max(runs, key=lambda x: (x["mAP"], x["R1"]))
    BEST_JSON.write_text(json.dumps({
        "T": best["T"], "W": best["W"],
        "mAP": best["mAP"], "Rank-1": best["R1"],
        "Rank-5": best.get("R5", -1), "Rank-10": best.get("R10", -1),
        "source_tag": best["tag"],
        "note": "Best SupCon params from B1 search (single-seed screening)"
    }, indent=2))
    print(f"[B1] Saved best → {BEST_JSON}")

    # 4) Retrain the best combo for FULL_EPOCHS with 3 seeds
    seed_best_records: Dict[int, Dict[str, Any]] = {}
    for seed in FULL_SEEDS:
        full_tag = f"b1_supcon_best_T{_fmt(best['T'])}_W{_fmt(best['W'])}_seed{seed}"
        print(f"[B1] Full retrain: (T={best['T']}, W={best['W']}) seed={seed} → tag={full_tag}")
        subprocess.check_call([
            sys.executable, "run_modelB_deit.py",
            "--config", CONFIG,
            "--opts",
            "LOSS.SUPCON.ENABLE", "True",   # ensure SupCon is ON
            "LOSS.SUPCON.T", str(best["T"]),
            "LOSS.SUPCON.W", str(best["W"]),
            "SOLVER.MAX_EPOCHS", str(FULL_EPOCHS),
            "SOLVER.SEED", str(seed),       # ← 3 seeds for confirmation
            "OUTPUT_DIR", str(LOG_ROOT),
            "TAG", full_tag,
        ])

        # After each seed is done, pick its best epoch
        test_dir = LOG_ROOT / f"veri776_{full_tag}_deit_test"
        best_ep = pick_best_epoch_metrics(test_dir)
        if best_ep is None:
            print(f"[B1][warn] No valid epoch metrics found under: {test_dir}")
        else:
            seed_best_records[seed] = best_ep
            print(f"[B1] Seed {seed} best epoch: {best_ep}")

    # 5) Aggregate mean/std across seeds and save summary JSON
    if seed_best_records:
        mAPs  = [rec["mAP"] for rec in seed_best_records.values()]
        R1s   = [rec["Rank-1"] for rec in seed_best_records.values()]
        R5s   = [rec["Rank-5"] for rec in seed_best_records.values()]
        R10s  = [rec["Rank-10"] for rec in seed_best_records.values()]

        summary = {
            "T": best["T"],
            "W": best["W"],
            "seeds": {str(k): v for k, v in seed_best_records.items()},
            "mean": {
                "mAP":  round(safe_mean(mAPs), 4),
                "Rank-1": round(safe_mean(R1s), 4),
                "Rank-5": round(safe_mean(R5s), 4),
                "Rank-10": round(safe_mean(R10s), 4),
            },
            "std": {
                "mAP":  round(safe_stdev(mAPs), 4),
                "Rank-1": round(safe_stdev(R1s), 4),
                "Rank-5": round(safe_stdev(R5s), 4),
                "Rank-10": round(safe_stdev(R10s), 4),
            },
            "note": "Each seed uses its best epoch by mAP (tie-breaker Rank-1). "
                    "Means/std computed over seeds with valid metrics.",
        }
        BEST_SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
        print(f"[B1] Wrote summary → {BEST_SUMMARY_JSON}")
    else:
        print("[B1][warn] No seed best records collected; skip summary.")

    print(f"[B1] Full retrains finished under {LOG_ROOT}")

if __name__ == "__main__":
    main()
