#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B1 SupCon grid search (minimal) with grid loaded from YAML (LOSS.SUPCON.SEARCH).

Author: Hang Zhang (hzha0521)
Date  : 2025-09-14

What it does:
- Reads T/W candidates from the experiment YAML: LOSS.SUPCON.SEARCH.{T,W}
- Loops over (T, W), runs short training to screen
- Parses metrics from the latest epoch_* subfolder (summary.json preferred)
- Saves the best combo to logs/b1_supcon_best.json
- Retrains the best combo for FULL_EPOCHS

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
from typing import List, Tuple, Optional

# ====== User settings ======
CONFIG = "configs/VeRi/deit_transreid_stride_b1_supcon.yml"
SEARCH_EPOCHS = 12       # 12–15 recommended for quick screening
FULL_EPOCHS   = 30
# ===========================

# ---------- Env helpers ----------
def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False

def pick_log_root() -> Path:
    """Choose one root for logs/results, same rule as B2 and model runner."""
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
        t_list = _to_float_list(cfg.LOSS.SUPCON.SEARCH.T) if "LOSS" in cfg else []
        w_list = _to_float_list(cfg.LOSS.SUPCON.SEARCH.W) if "LOSS" in cfg else []
        if t_list and w_list:
            return sorted(set(t_list)), sorted(set(w_list))
    except Exception:
        pass

    # Attempt via PyYAML (if installed)
    try:
        import yaml  # type: ignore
        with open(cfg_path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f)
        loss = (y or {}).get("LOSS", {})
        supc = (loss or {}).get("SUPCON", {})
        search = (supc or {}).get("SEARCH", {})
        t_list = _to_float_list(search.get("T"))
        w_list = _to_float_list(search.get("W"))
        if t_list and w_list:
            return sorted(set(t_list)), sorted(set(w_list))
    except Exception:
        pass

    # Fallback defaults (your current common choices)
    return [0.07], [0.25, 0.30, 0.35]

# ---------- Parse metrics ----------
def _latest_epoch_dir(out_test_dir: Path) -> Optional[Path]:
    """Pick the newest epoch_* subfolder inside out_test_dir."""
    epochs = sorted([p for p in out_test_dir.glob("epoch_*") if p.is_dir()],
                    key=lambda p: int(p.name.split("_")[-1]))
    return epochs[-1] if epochs else None

def parse_map_rank1(out_test_dir: Path) -> Tuple[float, float]:
    """
    Parse (mAP, Rank-1) from the latest epoch_* subfolder.
    Priority: summary.json; fallback to aggregated test_all.log.
    Returns (-1, -1) if not found.
    """
    ep = _latest_epoch_dir(out_test_dir)
    if ep:
        sj = ep / "summary.json"
        if sj.exists():
            try:
                obj = json.loads(sj.read_text())
                return float(obj.get("mAP", -1)), float(obj.get("Rank-1", -1))
            except Exception:
                pass
        for name in ("test_summary.txt", "results.txt", "log.txt"):
            p = ep / name
            if p.exists():
                s = p.read_text(encoding="utf-8", errors="ignore")
                m = re.search(r"mAP[^0-9]*([0-9]+(?:\.[0-9]+)?)", s, re.I)
                r = re.search(r"Rank[-\s]?1[^0-9]*([0-9]+(?:\.[0-9]+)?)", s, re.I)
                return (float(m.group(1)) if m else -1,
                        float(r.group(1)) if r else -1)

    all_log = out_test_dir / "test_all.log"
    if all_log.exists():
        s = all_log.read_text(encoding="utf-8", errors="ignore")
        mm = list(re.finditer(r"mAP[^0-9]*([0-9]+(?:\.[0-9]+)?)", s, re.I))
        rr = list(re.finditer(r"Rank[-\s]?1[^0-9]*([0-9]+(?:\.[0-9]+)?)", s, re.I))
        mAP = float(mm[-1].group(1)) if mm else -1
        r1  = float(rr[-1].group(1)) if rr else -1
        return mAP, r1

    return -1, -1

# ---------- One run ----------
def run_one(t: float, w: float):
    tag = f"b1_supcon_T{_fmt(t)}_W{_fmt(w)}"
    subprocess.check_call([
        sys.executable, "run_modelB_deit.py",
        "--config", CONFIG,
        "--opts",
        "LOSS.SUPCON.T", str(t),
        "LOSS.SUPCON.W", str(w),
        "SOLVER.MAX_EPOCHS", str(SEARCH_EPOCHS),
        "OUTPUT_DIR", str(LOG_ROOT),
        "TAG", tag,
    ])
    # Must match run_modelB_deit.py naming: veri776_{tag}_deit_test
    out_test_dir = LOG_ROOT / f"veri776_{tag}_deit_test"
    mAP, r1 = parse_map_rank1(out_test_dir)
    print(f"[B1] RESULT tag={tag} T={t} W={w} mAP={mAP} Rank-1={r1}")
    return {"tag": tag, "T": t, "W": w, "mAP": mAP, "R1": r1}

# ---------- Main ----------
def main():
    # 1) Load grid from YAML (fallback to defaults if missing)
    T_list, W_list = load_grid_from_yaml(CONFIG)
    print(f"[B1] Search grid → T={T_list} ; W={W_list}")

    # 2) Sweep grid and collect metrics
    runs = [run_one(t, w) for t, w in itertools.product(T_list, W_list)]

    # 3) Select best by (mAP, Rank-1)
    best = max(runs, key=lambda x: (x["mAP"], x["R1"]))
    BEST_JSON.write_text(json.dumps({
        "T": best["T"], "W": best["W"], "mAP": best["mAP"], "Rank-1": best["R1"],
        "source_tag": best["tag"], "note": "Best SupCon params from B1 search"
    }, indent=2))
    print(f"[B1] Saved best → {BEST_JSON}")

    # 4) Retrain the best combo for FULL_EPOCHS
    full_tag = f"b1_supcon_best_T{_fmt(best['T'])}_W{_fmt(best['W'])}"
    subprocess.check_call([
        sys.executable, "run_modelB_deit.py",
        "--config", CONFIG,
        "--opts",
        "LOSS.SUPCON.T", str(best["T"]),
        "LOSS.SUPCON.W", str(best["W"]),
        "SOLVER.MAX_EPOCHS", str(FULL_EPOCHS),
        "OUTPUT_DIR", str(LOG_ROOT),
        "TAG", full_tag,
    ])
    print(f"[B1] Full retrain done under {LOG_ROOT}")

if __name__ == "__main__":
    main()
