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
#                            - Skip or resume per seed based on epoch_* count
#                            - If training finished but no epoch results, auto eval-only
#                            - Warm-start only when no progress exists (finished==0)
# =============================================================================
"""
B2: Use best (T,W) from B1 to run long training with seeds=[0,1,2],
pick best epoch per seed (by mAP, tie Rank-1), then aggregate mean/std.

Robustness / Seamless behavior:
- Auto-select LOG_ROOT (local/Colab)
- Per-seed run is idempotent:
  * If epoch_* >= FULL_EPOCHS → skip training; ensure test results (eval-only if missing)
  * If 0 < epoch_* < FULL_EPOCHS → resume training (same TAG/OUTPUT_DIR)
  * If epoch_* == 0 → (first run) warm-start from B1's best weight when available
- Test guard: if finished but no epoch summaries, try eval-only to produce results
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

# ---------- Metric parsing helpers ----------
def _re_pick(text: str, pat: str) -> float:
    m = re.search(pat, text, re.I)
    return float(m.group(1)) if m else -1.0

def _re_pick_last(text: str, pat: str) -> float:
    mm = list(re.finditer(pat, text, re.I))
    return float(mm[-1].group(1)) if mm else -1.0

def parse_metrics_from_epoch_dir(epoch_dir: Path) -> Tuple[float, float, float, float]:
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
    return len([p for p in test_dir.glob("epoch_*") if p.is_dir() and p.name.split("_")[-1].isdigit()])

def has_any_epoch_results(test_dir: Path) -> bool:
    for ep in test_dir.glob("epoch_*"):
        if not ep.is_dir():
            continue
        if (ep / "summary.json").exists():
            return True
        for name in ("test_summary.txt", "results.txt", "log.txt"):
            if (ep / name).exists():
                return True
    if (test_dir / "test_all.log").exists():
        return True
    return False

def try_eval_only(tag: str, log_root: Path) -> None:
    """Try several common eval-only switches to produce per-epoch test outputs."""
    test_dir = log_root / f"veri776_{tag}_deit_test"
    print(f"[B2] Trying eval-only to produce test results for tag={tag} ...")
    candidate_cmds = [
        ([sys.executable, "run_modelB_deit.py", "--config", B2_CONFIG, "--only-test",
          "--opts", "OUTPUT_DIR", str(log_root), "TAG", tag], "--only-test"),
        ([sys.executable, "run_modelB_deit.py", "--config", B2_CONFIG, "--eval-only",
          "--opts", "OUTPUT_DIR", str(log_root), "TAG", tag], "--eval-only"),
        ([sys.executable, "run_modelB_deit.py", "--config", B2_CONFIG,
          "--opts", "EVAL_ONLY", "True", "OUTPUT_DIR", str(log_root), "TAG", tag], "EVAL_ONLY True"),
        ([sys.executable, "run_modelB_deit.py", "--config", B2_CONFIG,
          "--opts", "TEST.ONLY", "True", "OUTPUT_DIR", str(log_root), "TAG", tag], "TEST.ONLY True"),
    ]
    for cmd, desc in candidate_cmds:
        try:
            print(f"[B2] Eval-only attempt: {desc}")
            subprocess.check_call(cmd)
            if has_any_epoch_results(test_dir):
                print(f"[B2] Eval-only success via {desc}")
                return
        except Exception as e:
            print(f"[B2] Eval-only attempt failed ({desc}): {e}")
    print("[B2][warn] Could not trigger eval-only with known switches. "
          "Adjust try_eval_only() if your trainer uses different flags.")

def has_any_checkpoint(tag: str, log_root: Path) -> bool:
    """Heuristic: if epoch_* exists or common checkpoint files are present under the run dir."""
    test_dir = log_root / f"veri776_{tag}_deit_test"
    if count_finished_epochs(test_dir) > 0:
        return True
    # Try some common checkpoint locations/names adjacent to test dir
    run_root = log_root / f"veri776_{tag}_deit"
    patterns = ["*.pth", "checkpoint*.pth", "model_*.pth", "last*.pth"]
    for pat in patterns:
        if any(run_root.glob(pat)):
            return True
    return False

# ---------- Idempotent per-seed runner ----------
def ensure_full_run_seed(T: float, W: float, seed: int, epochs: int,
                         log_root: Path, b1_weight: Optional[Path], use_warmstart_if_empty: bool = True
                         ) -> Optional[Dict[str, Any]]:
    tag = f"b2_with_b1best_T{_fmt(T)}_W{_fmt(W)}_seed{seed}"
    test_dir = log_root / f"veri776_{tag}_deit_test"
    finished = count_finished_epochs(test_dir)

    if finished >= epochs:
        if not has_any_epoch_results(test_dir):
            print(f"[B2] Finished but no epoch results. Triggering eval-only: tag={tag}")
            try_eval_only(tag, log_root)
    else:
        # Build base cmd
        cmd = [
            sys.executable, "run_modelB_deit.py",
            "--config", B2_CONFIG,
            "--opts",
            # SupCon from B1 best
            "LOSS.SUPCON.ENABLE", "True",
            "LOSS.SUPCON.T", str(T),
            "LOSS.SUPCON.W", str(W),
            # TripletX + training mode
            "LOSS.TRIPLETX.ENABLE", "True",
            "MODEL.TRAINING_MODE", "supervised",
            # Run settings
            "SOLVER.MAX_EPOCHS", str(epochs),
            "SOLVER.SEED", str(seed),
            "OUTPUT_DIR", str(log_root),
            "TAG", tag,
        ]

        # Decide warm-start vs resume:
        # - If no progress & b1_weight available & allowed → warm-start (non-resume)
        # - Else → rely on trainer's auto-resume by TAG/OUTPUT_DIR
        if finished == 0 and use_warmstart_if_empty and (b1_weight is not None) and (not has_any_checkpoint(tag, log_root)):
            print(f"[B2] First run (no progress). Warm-start from B1 best: {b1_weight}")
            cmd += [
                "MODEL.PRETRAIN_CHOICE", "self",
                "MODEL.PRETRAIN_PATH", str(b1_weight),
            ]
        else:
            print(f"[B2] Resume/continue run: tag={tag} (finished={finished}/{epochs})")
            # If your trainer requires explicit resume flag, uncomment the next line:
            # cmd += ["SOLVER.RESUME", "True"]

        print("[B2] Launch:", " ".join(cmd))
        subprocess.check_call(cmd)

    # Pick best epoch metrics
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

    # Load best (T,W) strictly from B1 output
    best_json = log_root / "b1_supcon_best.json"
    if not best_json.exists():
        raise SystemExit(f"[B2] Missing {best_json}. Run B1 search first.")
    try:
        obj = json.loads(best_json.read_text())
        T = float(obj["T"])
        W = float(obj["W"])
    except Exception as e:
        raise SystemExit(f"[B2] Failed to parse {best_json}: {e}")

    # Locate B1 best weight for optional warm-start
    b1_weight = None
    try:
        # Prefer explicit run_dir if present
        run_dir = Path(json.loads(best_json.read_text()).get("run_dir", ""))
        if run_dir:
            cand = run_dir / "transformer_12.pth"
            if cand.exists():
                b1_weight = cand
        if b1_weight is None:
            # Fallback: try to infer from source_tag directory
            src_tag = obj.get("source_tag")
            if src_tag:
                guess = log_root / f"veri776_{src_tag}_deit" / "transformer_12.pth"
                if guess.exists():
                    b1_weight = guess
        if b1_weight:
            print(f"[B2] Warm-start weight found: {b1_weight}")
        else:
            print("[B2] No warm-start weight found; will resume/continue if progress exists.")
    except Exception as e:
        print(f"[B2][warn] Failed to locate B1 weight: {e}")

    seed_best_records: Dict[int, Dict[str, Any]] = {}
    for seed in SEEDS:
        rec = ensure_full_run_seed(T, W, seed, FULL_EPOCHS, log_root, b1_weight)
        if rec is not None:
            seed_best_records[seed] = rec

    # Aggregate
    summary_path = log_root / "b2_supcon_best_summary.json"
    if seed_best_records:
        mAPs  = [rec["mAP"] for rec in seed_best_records.values()]
        R1s   = [rec["Rank-1"] for rec in seed_best_records.values()]
        R5s   = [rec["Rank-5"] for rec in seed_best_records.values()]
        R10s  = [rec["Rank-10"] for rec in seed_best_records.values()]
        summary = {
            "T": T, "W": W, "config": B2_CONFIG, "epochs": FULL_EPOCHS,
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
