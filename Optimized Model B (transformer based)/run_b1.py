#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===========================================================
# File: run_b1.py
# Purpose: B1 SupCon grid search with YAML-driven grid and
#          seamless resume + smart test skipping.
# Author: Hang Zhang (hzha0521)
# ===========================================================
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
# [2025-09-16 | Hang Zhang] Seamless search/full runs:
#                            - Detect progress by checkpoints (state_*.pth / transformer_*.pth)
#                              and by existing test results (epoch_*).
#                            - If training reached target epochs, avoid retraining and
#                              auto-run eval-only to materialize results if missing.
#                            - Force EVAL_PERIOD=1 & CHECKPOINT_PERIOD=1 for new runs so
#                              future re-entries are robust.
#                            - Fix seeded path pattern: veri776_{TAG}_seed{seed}_deit_run/test
#                            - Add explicit "already-tested" skips (no duplicate testing).
# [2025-09-16 | Hang Zhang] Replace eval-only guessing with robust re-test:
#                            - NEW eval_missing_epochs_via_test_py(): iterate all
#                              transformer_*.pth and call test.py only for epochs
#                              that have no epoch_* results yet.
#                            - When trained_max >= target, always call the new
#                              re-test function (idempotent), which guarantees
#                              "no re-train, no duplicate test".
#                            - Keep English code comments; Chinese guidance in chat.
# [2025-09-16 | ChatGPT   ] NEW: pass DATASETS.ROOT_DIR everywhere
#                            - detect_data_root(): auto-detect VeRi dataset root
#                            - add DATASETS.ROOT_DIR to all run/test invocations
# [2025-09-17 | ChatGPT   ] FIX: recognize existing test artifacts correctly
#                            - Treat `test_log.txt` and `dist_mat.npy` as valid
#                              epoch-completion markers (besides summary/log files)
#                            - Add diagnostics for tested_max / paths
# ===========================================================

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
SEARCH_EPOCHS = 12        # quick screening epochs
FULL_EPOCHS   = 30        # long training for the best (T,W)
SEARCH_SEED   = 0         # single seed for search
FULL_SEEDS    = [0, 1, 2] # seeds for confirmation
# ===========================

# ---- Test completion markers (unified) ----
# Presence of ANY of these files under test_dir/epoch_X/ == "tested".
TEST_MARKER_FILES_DEFAULT = (
    "summary.json", "test_summary.txt", "results.txt", "log.txt",
    "test_log.txt",      # your current artifact
    "dist_mat.npy"       # supported if you enable matrix saving
)

# ---------- Env helpers ----------
def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False

def pick_log_root() -> Path:
    """Choose log root: $OUTPUT_ROOT > Colab Drive > ./logs."""
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

# ---------- Detect dataset root (VeRi) ----------
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
        str(Path.cwd().parents[1] / "datasets"),   # ../../datasets from project root
        str(Path.cwd() / "datasets"),
    ]
    for c in candidates:
        if (Path(c) / "VeRi").exists():
            return c

    # Fallback to YAML-default parent; if wrong, test.py will fail visibly.
    return str(Path.cwd().parents[1] / "datasets")

DATA_ROOT = detect_data_root()
print(f"[B1] Using DATASETS.ROOT_DIR={DATA_ROOT}")

# ---------- Load grid from YAML (preferred), fallback to defaults ----------
def _to_float_list(x) -> List[float]:
    if isinstance(x, (int, float)):
        return [float(x)]
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    return []

def load_grid_from_yaml(cfg_path: str) -> Tuple[List[float], List[float]]:
    """
    Try YACS first, then PyYAML. Return (T_list, W_list).
    Fallback to T=[0.07], W=[0.25,0.30,0.35] if not found.
    """
    # Attempt via YACS
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

    # Attempt via PyYAML
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

    # Fallback defaults
    return [0.07], [0.25, 0.30, 0.35]

# ---------- Metric parsing helpers ----------
def _latest_epoch_dir(out_test_dir: Path) -> Optional[Path]:
    """Pick the newest epoch_* subfolder."""
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
    Return (mAP, Rank-1, Rank-5, Rank-10) from latest epoch_*.
    Priority: epoch_*/summary.json; fallback: epoch_* logs; fallback: test_all.log.
    Missing values return -1.0.
    """
    ep = _latest_epoch_dir(out_test_dir)
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
        # include test_log.txt as a valid source
        for name in ("test_summary.txt", "results.txt", "log.txt", "test_log.txt"):
            p = ep / name
            if p.exists():
                s = p.read_text(encoding="utf-8", errors="ignore")
                mAP = _re_pick(s, r"mAP[^0-9]*([0-9]+(?:\.[0-9]+)?)")
                r1  = _re_pick(s, r"Rank[-\s]?1[^0-9]*([0-9]+(?:\.[0-9]+)?)")
                r5  = _re_pick(s, r"Rank[-\s]?5[^0-9]*([0-9]+(?:\.[0-9]+)?)")
                r10 = _re_pick(s, r"Rank[-\s]?10[^0-9]*([0-9]+(?:\.[0-9]+)?)")
                return mAP, r1, r5, r10

    all_log = out_test_dir / "test_all.log"
    if all_log.exists():
        s = all_log.read_text(encoding="utf-8", errors="ignore")
        mAP = _re_pick_last(s, r"mAP[^0-9]*([0-9]+(?:\.[0-9]+)?)")
        r1  = _re_pick_last(s, r"Rank[-\s]?1[^0-9]*([0-9]+(?:\.[0-9]+)?)")
        r5  = _re_pick_last(s, r"Rank[-\s]?5[^0-9]*([0-9]+(?:\.[0-9]+)?)")
        r10 = _re_pick_last(s, r"Rank[-\s]?10[^0-9]*([0-9]+(?:\.[0-9]+)?)")
        return mAP, r1, r5, r10

    return -1.0, -1.0, -1.0, -1.0

def parse_metrics_from_epoch_dir(epoch_dir: Path) -> Tuple[float, float, float, float]:
    """Read metrics from a single epoch_* directory."""
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
    # include test_log.txt for robustness
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
    """Pick epoch with best (mAP, Rank-1)."""
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

# ---------- Progress & test guards ----------
def count_finished_epochs(test_dir: Path) -> int:
    """Count epoch_* folders present under test_dir."""
    return len([p for p in test_dir.glob("epoch_*") if p.is_dir() and p.name.split("_")[-1].isdigit()])

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
        # unified marker check
        for f in TEST_MARKER_FILES_DEFAULT:
            if (ep / f).exists():
                max_ep = max(max_ep, idx)
                break
    return max_ep

def _max_epoch_from_checkpoints(run_dir: Path) -> int:
    """Scan common checkpoint names and return max epoch seen, else 0."""
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

# ---------- Robust re-test (no reliance on launcher eval-only flags) ----------
def eval_missing_epochs_via_test_py(tag_with_seed: str, config_path: str, log_root: Path) -> None:
    """
    Reconstruct per-epoch test outputs by directly invoking test.py on every
    checkpoint found under .../{tag}_deit_run, writing results to
    .../{tag}_deit_test/epoch_{E}.
    Only epochs without results are tested; existing epoch_* are skipped.
    """
    run_dir  = log_root / f"veri776_{tag_with_seed}_deit_run"
    test_dir = log_root / f"veri776_{tag_with_seed}_deit_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    rx = re.compile(r"transformer_(\d+)\.pth$")
    ckpts: List[Tuple[int, Path]] = []
    for p in run_dir.glob("transformer_*.pth"):
        m = rx.search(p.name)
        if m:
            ckpts.append((int(m.group(1)), p))
    ckpts.sort(key=lambda x: x[0])

    if not ckpts:
        print(f"[B1][eval] No checkpoints found under: {run_dir}")
        return

    print(f"[B1][diagnose] run_root={run_dir}")
    print(f"[B1][diagnose] test_root={test_dir}")

    for ep, ck in ckpts:
        out_ep = test_dir / f"epoch_{ep}"

        # unified "done" check (includes test_log.txt / dist_mat.npy)
        already = False
        if out_ep.exists():
            for name in TEST_MARKER_FILES_DEFAULT:
                if (out_ep / name).exists():
                    already = True
                    break
        if already:
            print(f"[B1][eval] Skip epoch {ep}: existing test markers in {out_ep}")
            continue

        out_ep.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "test.py", "--config_file", str(config_path),
            "MODEL.DEVICE", "cuda",
            "TEST.WEIGHT", str(ck),
            "OUTPUT_DIR", str(out_ep),
            "DATASETS.ROOT_DIR", DATA_ROOT,   # critical override
        ]
        print("[B1][eval] Launch:", " ".join(cmd))
        ret = subprocess.call(cmd)
        if ret != 0:
            print(f"[B1][eval][warn] test.py failed for epoch {ep} (ret={ret})")

# ---------- Tag & path helpers (seed-aware) ----------
def _search_tag_base(t: float, w: float) -> str:
    return f"b1_supcon_T{_fmt(t)}_W{_fmt(w)}"

def _search_tag_with_seed(t: float, w: float) -> str:
    return f"{_search_tag_base(t, w)}_seed{SEARCH_SEED}"

def _search_run_dir(t: float, w: float) -> Path:
    return LOG_ROOT / f"veri776_{_search_tag_with_seed(t, w)}_deit_run"

def _search_test_dir(t: float, w: float) -> Path:
    return LOG_ROOT / f"veri776_{_search_tag_with_seed(t, w)}_deit_test"

def _full_tag_with_seed(best_T: float, best_W: float, seed: int) -> str:
    return f"b1_supcon_best_T{_fmt(best_T)}_W{_fmt(best_W)}_seed{seed}"

def _full_run_dir(best_T: float, best_W: float, seed: int) -> Path:
    return LOG_ROOT / f"veri776_{_full_tag_with_seed(best_T, best_W, seed)}_deit_run"

def _full_test_dir(best_T: float, best_W: float, seed: int) -> Path:
    return LOG_ROOT / f"veri776_{_full_tag_with_seed(best_T, best_W, seed)}_deit_test"

# ---------- Idempotent wrappers ----------
def ensure_search_run(t: float, w: float) -> Dict[str, Any]:
    """
    Ensure one (T,W) search run is completed with results.
    Behavior:
      - If tests already cover SEARCH_EPOCHS → skip training/testing; parse.
      - Else if checkpoints reached SEARCH_EPOCHS → reconstruct missing tests via test.py, then parse.
      - Else → (re)train with EVAL_PERIOD=1 & CHECKPOINT_PERIOD=1, then parse.
    """
    tag_base = _search_tag_base(t, w)
    tag_seed = _search_tag_with_seed(t, w)
    run_dir  = _search_run_dir(t, w)
    test_dir = _search_test_dir(t, w)

    tested_max = _max_test_epoch(test_dir)
    print(f"[B1][diagnose] SEARCH tested_max={tested_max} under {test_dir}")
    if tested_max >= SEARCH_EPOCHS:
        mAP, r1, r5, r10 = parse_metrics(test_dir)
        print(f"[B1] SEARCH ALREADY-TESTED tag={tag_seed} (tested_max={tested_max}≥{SEARCH_EPOCHS}) "
              f"mAP={mAP} R1={r1} R5={r5} R10={r10}")
        return {"tag": tag_base, "T": t, "W": w, "mAP": mAP, "R1": r1, "R5": r5, "R10": r10}

    trained_max = _max_epoch_from_checkpoints(run_dir)
    if trained_max >= SEARCH_EPOCHS:
        print(f"[B1] SEARCH trained_max={trained_max}≥{SEARCH_EPOCHS}; ensure tests exist for tag={tag_seed}")
        eval_missing_epochs_via_test_py(tag_seed, CONFIG, LOG_ROOT)
        mAP, r1, r5, r10 = parse_metrics(test_dir)
        print(f"[B1] SEARCH DONE (no retrain) tag={tag_seed} mAP={mAP} R1={r1} R5={r5} R10={r10}")
        return {"tag": tag_base, "T": t, "W": w, "mAP": mAP, "R1": r1, "R5": r5, "R10": r10}

    # Not done → (re)run training
    print(f"[B1] SEARCH RUN tag={tag_seed} (trained_max={trained_max}/{SEARCH_EPOCHS})")
    subprocess.check_call([
        sys.executable, "run_modelB_deit.py",
        "--config", CONFIG,
        "--opts",
        "LOSS.SUPCON.ENABLE", "True",
        "LOSS.SUPCON.T", str(t),
        "LOSS.SUPCON.W", str(w),
        "SOLVER.MAX_EPOCHS", str(SEARCH_EPOCHS),
        "SOLVER.SEED", str(SEARCH_SEED),
        "SOLVER.CHECKPOINT_PERIOD", "1",
        "SOLVER.EVAL_PERIOD", "1",
        "DATASETS.ROOT_DIR", DATA_ROOT,  # critical override for training/eval
        "OUTPUT_DIR", str(LOG_ROOT),
        "TAG", tag_base,  # seed suffix auto-added by the trainer
    ])
    eval_missing_epochs_via_test_py(tag_seed, CONFIG, LOG_ROOT)

    mAP, r1, r5, r10 = parse_metrics(test_dir)
    print(f"[B1] SEARCH RESULT tag={tag_seed} T={t} W={w} seed={SEARCH_SEED} "
          f"mAP={mAP} Rank-1={r1} Rank-5={r5} Rank-10={r10}")
    return {"tag": tag_base, "T": t, "W": w, "mAP": mAP, "R1": r1, "R5": r5, "R10": r10}

def ensure_full_run(best_T: float, best_W: float, seed: int, epochs: int) -> Optional[Dict[str, Any]]:
    """
    Ensure one full retrain run is completed with results.
    Behavior:
      - If tests already cover 'epochs' → skip training; pick best epoch.
      - Else if checkpoints reached 'epochs' → reconstruct missing tests via test.py, then pick best.
      - Else → (re)train (EVAL_PERIOD=1 & CHECKPOINT_PERIOD=1), reconstruct missing tests, then pick best.
    """
    tag_seed = _full_tag_with_seed(best_T, best_W, seed)
    run_dir  = _full_run_dir(best_T, best_W, seed)
    test_dir = _full_test_dir(best_T, best_W, seed)

    tested_max = _max_test_epoch(test_dir)
    print(f"[B1][diagnose] FULL tested_max={tested_max} under {test_dir}")
    if tested_max >= epochs:
        best_ep = pick_best_epoch_metrics(test_dir)
        if best_ep is None:
            print(f"[B1][warn] Already tested to {tested_max}, but cannot parse metrics under: {test_dir}")
            return None
        print(f"[B1] FULL ALREADY-TESTED tag={tag_seed} (tested_max={tested_max}≥{epochs}) best_ep={best_ep}")
        return best_ep

    trained_max = _max_epoch_from_checkpoints(run_dir)
    if trained_max >= epochs:
        print(f"[B1] FULL trained_max={trained_max}≥{epochs}; ensure tests exist for tag={tag_seed}")
        eval_missing_epochs_via_test_py(tag_seed, CONFIG, LOG_ROOT)
    else:
        print(f"[B1] FULL RUN (resume if exists): T={best_T}, W={best_W}, seed={seed} → tag={tag_seed} "
              f"(trained_max={trained_max}/{epochs})")
        subprocess.check_call([
            sys.executable, "run_modelB_deit.py",
            "--config", CONFIG,
            "--opts",
            "LOSS.SUPCON.ENABLE", "True",
            "LOSS.SUPCON.T", str(best_T),
            "LOSS.SUPCON.W", str(best_W),
            "SOLVER.MAX_EPOCHS", str(epochs),
            "SOLVER.SEED", str(seed),
            "SOLVER.CHECKPOINT_PERIOD", "1",
            "SOLVER.EVAL_PERIOD", "1",
            "DATASETS.ROOT_DIR", DATA_ROOT,  # critical override
            "OUTPUT_DIR", str(LOG_ROOT),
            "TAG", f"b1_supcon_best_T{_fmt(best_T)}_W{_fmt(best_W)}",  # launcher auto-adds _seed{seed}
        ])
        eval_missing_epochs_via_test_py(tag_seed, CONFIG, LOG_ROOT)

    best_ep = pick_best_epoch_metrics(test_dir)
    if best_ep is None:
        print(f"[B1][warn] No valid epoch metrics found under: {test_dir}")
        return None
    print(f"[B1] Seed {seed} best epoch: {best_ep}")
    return best_ep

# ---------- Main ----------
def main():
    # 1) Load grid
    T_list, W_list = load_grid_from_yaml(CONFIG)
    print(f"[B1] Search grid → T={T_list} ; W={W_list}")
    print(f"[B1] Search uses single seed: {SEARCH_SEED}")
    print(f"[B1] Full training seeds   : {FULL_SEEDS}")

    # 2) Sweep grid (resumable & test-aware)
    runs = [ensure_search_run(t, w) for t, w in itertools.product(T_list, W_list)]
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

    # 4) Full retrain for FULL_EPOCHS with 3 seeds (resumable & test-aware)
    seed_best_records: Dict[int, Dict[str, Any]] = {}
    for seed in FULL_SEEDS:
        best_ep = ensure_full_run(best["T"], best["W"], seed, FULL_EPOCHS)
        if best_ep is not None:
            seed_best_records[seed] = best_ep

    # 5) Aggregate mean/std across seeds
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
