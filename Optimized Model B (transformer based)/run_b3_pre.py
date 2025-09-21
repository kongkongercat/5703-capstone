import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import statistics as stats
import glob

# ====== User settings ======
CONFIG = "configs/VeRi/deit_transreid_stride_b3_ssl_pretrain.yml"
SEARCH_EPOCHS = 12        # 12–15 recommended for quick screening
FULL_EPOCHS   = 30        # long training for the best (T,W)
# Seeds policy
SEARCH_SEED = 0           # fast screening: single seed only
FULL_SEEDS  = [0, 1, 2]   # confirm best with 3 seeds

# Whether to force SSL-style overrides so the pipeline does NOT use labels
FORCE_SSL_OVERRIDES = True
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
BEST_JSON = LOG_ROOT / "b3_supcon_best.json"
BEST_SUMMARY_JSON = LOG_ROOT / "b3_supcon_best_summary.json"

def _fmt(v: float) -> str:
    return str(v).replace(".", "p")

# ---------- YAML grid loader ----------
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
    # Try YACS
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

    # Try PyYAML
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

    return [0.07], [0.25, 0.30, 0.35]

# ---------- Regex helper / stats ----------
def _re_pick(s: str, pattern: str, default: float = -1.0) -> float:
    m = re.search(pattern, s, flags=re.I)
    try:
        return float(m.group(1)) if m else default
    except Exception:
        return default

def safe_mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and x >= 0]
    return sum(xs) / len(xs) if xs else float("nan")

def safe_stdev(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and x >= 0]
    return stats.pstdev(xs) if len(xs) > 1 else 0.0

# ---------- Metrics parsing helpers ----------
def _latest_epoch_dir(out_test_dir: Path) -> Optional[Path]:
    epochs = sorted(
        [p for p in out_test_dir.glob("epoch_*") if p.is_dir()],
        key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else -1,
    )
    return epochs[-1] if epochs else None

def _resolve_test_dir(log_root: Path, tag: str) -> Path:
    """
    Try to resolve the per-run test directory robustly.
    Primary expectation (from run_modelB_deit.py):
       logs/veri776_{tag}_deit_test
    If not found, fallback to glob search.
    """
    expected = log_root / f"veri776_{tag}_deit_test"
    if expected.exists():
        return expected
    # Fallback: glob within 3 levels
    candidates = glob.glob(str(log_root / f"**/*{tag}*deit_test"), recursive=True)
    if candidates:
        # Choose the most recent modified
        candidates = sorted(candidates, key=lambda p: os.path.getmtime(p), reverse=True)
        return Path(candidates[0])
    # As a last resort, return the expected (may not exist; caller should handle)
    return expected

def parse_metrics(out_test_dir: Path) -> Tuple[float, float, float, float]:
    """
    Parse metrics from latest epoch dir under out_test_dir.
    Returns (mAP, Rank-1, Rank-5, Rank-10) or (-1,-1,-1,-1) if not found.
    """
    ep = _latest_epoch_dir(out_test_dir)
    if not ep:
        return -1.0, -1.0, -1.0, -1.0

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

    # Fallback: parse text logs
    for name in ("test_summary.txt", "results.txt", "log.txt"):
        p = ep / name
        if p.exists():
            s = p.read_text(encoding="utf-8", errors="ignore")
            mAP = _re_pick(s, r"mAP[^0-9]*([0-9]+(?:\.[0-9]+)?)")
            r1  = _re_pick(s, r"Rank[-\s]?1[^0-9]*([0-9]+(?:\.[0-9]+)?)")
            r5  = _re_pick(s, r"Rank[-\s]?5[^0-9]*([0-9]+(?:\.[0-9]+)?)")
            r10 = _re_pick(s, r"Rank[-\s]?10[^0-9]*([0-9]+(?:\.[0-9]+)?)")
            return mAP, r1, r5, r10

    return -1.0, -1.0, -1.0, -1.0

def pick_best_epoch_metrics(test_dir: Path):
    """Pick best epoch by (mAP, Rank-1). Return a record dict or None."""
    best = None
    for ep in sorted([p for p in test_dir.glob("epoch_*") if p.is_dir()],
                     key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else -1):
        sj = ep / "summary.json"
        if sj.exists():
            try:
                obj = json.loads(sj.read_text())
                rec = {
                    "epoch": int(ep.name.split("_")[-1]) if ep.name.split("_")[-1].isdigit() else -1,
                    "mAP": float(obj.get("mAP", -1)),
                    "Rank-1": float(obj.get("Rank-1", obj.get("Rank1", -1))),
                    "Rank-5": float(obj.get("Rank-5", obj.get("Rank5", -1))),
                    "Rank-10": float(obj.get("Rank-10", obj.get("Rank10", -1))),
                }
                if rec["mAP"] < 0:
                    continue
                if best is None or (rec["mAP"], rec["Rank-1"]) > (best["mAP"], best["Rank-1"]):
                    best = rec
            except Exception:
                pass
    return best

# ---------- One run (search, single-seed) ----------
def _ssl_overrides_for_opts(t: float, w: float) -> List[str]:
    """
    Build --opts overrides to ensure label-free SSL if FORCE_SSL_OVERRIDES is True.
    Requires model pipeline to support z_supcon & instance positive rule.
    """
    if not FORCE_SSL_OVERRIDES:
        # Still ensure SupCon on + set (T, W)
        return [
            "LOSS.SUPCON.ENABLE", "True",
            "LOSS.SUPCON.T", str(t),
            "LOSS.SUPCON.W", str(w),
        ]
    return [
        # Turn on SupCon and set hyper-params
        "LOSS.SUPCON.ENABLE", "True",
        "LOSS.SUPCON.T", str(t),
        "LOSS.SUPCON.W", str(w),
        "LOSS.SUPCON.POS_RULE", "instance",
        # Switch to SSL mode + remove label-dependent pieces
        "MODEL.TRAINING_MODE", "self_supervised",
        "DATALOADER.SAMPLER", "random",
        "DATALOADER.NUM_INSTANCE", "1",
        # Avoid label-based evaluation during pretrain
        "TEST.EVAL", "False",
    ]

def run_one_search(t: float, w: float):
    tag = f"b3_supcon_T{_fmt(t)}_W{_fmt(w)}"
    cmd = [
        sys.executable, "run_modelB_deit.py",
        "--config", CONFIG,
        "--opts",
        *(_ssl_overrides_for_opts(t, w)),
        "SOLVER.MAX_EPOCHS", str(SEARCH_EPOCHS),
        "SOLVER.SEED", str(SEARCH_SEED),
        "OUTPUT_DIR", str(LOG_ROOT),
        "TAG", tag,
    ]
    print(f"[B3] Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

    out_test_dir = _resolve_test_dir(LOG_ROOT, tag)
    if not out_test_dir.exists():
        raise RuntimeError(f"[B3] Test dir not found: {out_test_dir}")

    mAP, r1, r5, r10 = parse_metrics(out_test_dir)
    print(f"[B3] SEARCH RESULT tag={tag} T={t} W={w} seed={SEARCH_SEED} "
          f"mAP={mAP} Rank-1={r1} Rank-5={r5} Rank-10={r10}")

    # 强制确保拿到有效指标；否则让你第一时间发现问题
    if mAP < 0 or r1 < 0:
        raise RuntimeError(f"[B3] Invalid metrics (mAP={mAP}, R1={r1}). "
                           f"Check logs under {out_test_dir} and training config.")

    return {"tag": tag, "T": t, "W": w, "mAP": mAP, "R1": r1, "R5": r5, "R10": r10}

# ---------- Main ----------
def main():
    T_list, W_list = load_grid_from_yaml(CONFIG)
    print(f"[B3] Search grid → T={T_list} ; W={W_list}")
    print(f"[B3] Search uses single seed: {SEARCH_SEED}")
    print(f"[B3] Full training seeds   : {FULL_SEEDS}")
    print(f"[B3] FORCE_SSL_OVERRIDES   : {FORCE_SSL_OVERRIDES}")

    runs = [run_one_search(t, w) for t, w in itertools.product(T_list, W_list)]
    if not runs:
        raise SystemExit("[B3] No runs executed. Check your SEARCH grid or paths.")

    # choose best
    best = max(runs, key=lambda x: (x["mAP"], x["R1"]))
    BEST_JSON.write_text(json.dumps({
        "T": best["T"], "W": best["W"],
        "mAP": best["mAP"], "Rank-1": best["R1"],
        "Rank-5": best.get("R5", -1), "Rank-10": best.get("R10", -1),
        "source_tag": best["tag"],
        "note": f"Best SupCon params from B3 search (single-seed screening). "
                f"FORCE_SSL_OVERRIDES={FORCE_SSL_OVERRIDES}",
    }, indent=2))
    print(f"[B3] Saved best → {BEST_JSON}")

    # ---- Full retrain with multiple seeds
    seed_best_records: Dict[int, Dict[str, Any]] = {}
    for seed in FULL_SEEDS:
        full_tag = f"b3_supcon_best_T{_fmt(best['T'])}_W{_fmt(best['W'])}_seed{seed}"
        print(f"[B3] Full retrain: (T={best['T']}, W={best['W']}) seed={seed} → tag={full_tag}")

        cmd = [
            sys.executable, "run_modelB_deit.py",
            "--config", CONFIG,
            "--opts",
            *(_ssl_overrides_for_opts(best["T"], best["W"])),
            "SOLVER.MAX_EPOCHS", str(FULL_EPOCHS),
            "SOLVER.SEED", str(seed),
            "OUTPUT_DIR", str(LOG_ROOT),
            "TAG", full_tag,
        ]
        print(f"[B3] Running: {' '.join(cmd)}")
        subprocess.check_call(cmd)

        test_dir = _resolve_test_dir(LOG_ROOT, full_tag)
        if not test_dir.exists():
            print(f"[B3][warn] Test dir not found: {test_dir}")
            continue
        best_ep = pick_best_epoch_metrics(test_dir)
        if best_ep is None:
            print(f"[B3][warn] No valid epoch metrics found under: {test_dir}")
        else:
            seed_best_records[seed] = best_ep
            print(f"[B3] Seed {seed} best epoch: {best_ep}")

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
                    f"FORCE_SSL_OVERRIDES={FORCE_SSL_OVERRIDES}.",
        }
        BEST_SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
        print(f"[B3] Wrote summary → {BEST_SUMMARY_JSON}")
    else:
        print("[B3][warn] No seed best records collected; skip summary.")

    print(f"[B3] Full retrains finished under {LOG_ROOT}")

if __name__ == "__main__":
    main()
