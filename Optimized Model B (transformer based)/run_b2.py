#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# B2 launcher — auto-inherit SupCon (T, W) from B1
#
# Change Log
# [2025-09-14 | Hang Zhang] Initial launcher: read T/W from logs/b1_supcon_best.json,
#                            explicitly enable SupCon via --opts, unify OUTPUT_DIR
#                            (env OUTPUT_ROOT > Colab Drive > ./logs), no fallback scan.
# =============================================================================
"""
What it does:
- Picks a log root that works both locally and in Colab (Google Drive).
- Loads (T, W) from logs/b1_supcon_best.json written by run_b1.py.
- Launches run_modelB_deit.py for B2, passing ENABLE/T/W and OUTPUT_DIR.
- No folder-name scanning fallback; if JSON missing, exit with an error.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# ---- User-configurable ----
B2_CONFIG   = "configs/VeiRi/deit_transreid_stride_b2_supcon_tripletx.yml".replace("VeiRi","VeRi")
FULL_EPOCHS = 30
# ---------------------------

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

    def _fmt(v: float) -> str:
        return str(v).replace(".", "p")

    tag = f"b2_with_b1best_T{_fmt(T)}_W{_fmt(W)}"

    # IMPORTANT: enable SupCon explicitly; defaults set ENABLE=False.
    cmd = [
        sys.executable, "run_modelB_deit.py",
        "--config", B2_CONFIG,
        "--opts",
        "LOSS.SUPCON.ENABLE", "True",   # turn on SupCon
        "LOSS.SUPCON.T", str(T),
        "LOSS.SUPCON.W", str(W),
        "SOLVER.MAX_EPOCHS", str(FULL_EPOCHS),
        "OUTPUT_DIR", str(log_root),
        "TAG", tag,
    ]
    print("[B2] Launch:", " ".join(cmd))
    subprocess.check_call(cmd)
    print("[B2] Done.")

if __name__ == "__main__":
    main()
