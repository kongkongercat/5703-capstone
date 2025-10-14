#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===========================================================
# File: run_b1_nogrid.py
# Purpose: B1 SupCon fixed-parameter training (no grid search).
#          - Read SupCon T/W from YAML (LOSS.SUPCON.T/W) or CLI
#          - Multi-seed full training with timestamped tags
#          - Robust resume and re-test to avoid duplicate work
#          - Seed-wise best-epoch selection and summary JSON
# Author: Hang Zhang (hzha0521)
# ===========================================================
# Change Log
# [2025-10-15 | Hang Zhang] First "no grid" version:
#                           - Removed grid search entirely
#                           - Added CLI --T/--W (override YAML)
#                           - Kept resume/re-test, timestamped TAG,
#                             and seed summary logic
# ===========================================================

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import statistics as stats
from datetime import datetime

# ================= Defaults (overridable by CLI) =================
DEFAULT_CONFIG      = "configs/VeRi/deit_transreid_stride_b1_supcon.yml"
DEFAULT_EPOCHS      = 120
DEFAULT_SAVE_EVERY  = 1
DEFAULT_SEEDS       = "0,1,2"
DEFAULT_DSPREFIX    = "veri776"
DEFAULT_T           = 0.07
DEFAULT_W           = 0.30
# =================================================================

# ---------- Test completion markers ----------
TEST_MARKER_FILES_DEFAULT = (
    "summary.json", "test_summary.txt", "results.txt", "log.txt",
    "test_log.txt",
    "dist_mat.npy"
)

# ================= Env / Paths =================
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

LOG_ROOT = pick_log_root()
LOG_ROOT.mkdir(parents=True, exist_ok=True)
SUMMARY_JSON = LOG_ROOT / "b1_supcon_nogrid_summary.json"

def _fmt(v: float) -> str:
    return str(v).replace(".", "p")

def detect_data_root() -> str:
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

# ================= CLI =================
def build_cli() -> argparse.Namespace:
    """
    Example:
      python run_b1_nogrid.py --epochs 120 --save-every 1 --seeds "0,1,2" \
        --config configs/VeRi/deit_transreid_stride_b1_supcon.yml \
        --output-root "/content/drive/.../logs" --ds-prefix veri776 \
        --T 0.07 --W 0.30
    """
    p = argparse.ArgumentParser(description="Run B1 SupCon (no grid) with fixed T/W.")
    # On/off test phase (kept for parity)
    p.add_argument("--test", dest="test", action="store_true")
    p.add_argument("--no-test", dest="test", action="store_false")
    p.set_defaults(test=True)

    # Epochs & frequency
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                   help="Total epochs for full training.")
    p.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY,
                   help="Checkpointing/testing interval in epochs.")

    # Seeds (comma or space separated)
    p.add_argument("--seeds", type=str, default=DEFAULT_SEEDS,
                   help="Comma/space-separated seeds, e.g. '0' or '0,1,2'.")

    # Paths / config
    p.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                   help="Path to config YAML for run_modelB_deit.py.")
    p.add_argument("--output-root", type=str, default=None,
                   help="Override output/log directory (default: auto).")
    p.add_argument("--data-root", type=str, default=None,
                   help="Override dataset root directory (default: auto).")
    p.add_argument("--ds-prefix", type=str, default=DEFAULT_DSPREFIX,
                   help="Dataset prefix for run/test dir naming (default: veri776).")

    # Tag & passthrough
    p.add_argument("--tag", type=str, default=None,
                   help="Custom base tag; auto-generated if not set.")
    p.add_argument("--opts", nargs=argparse.REMAINDER,
                   help="Extra YACS-style options passed to run_modelB_deit.py.")

    # Fixed SupCon hyperparams (override YAML LOSS.SUPCON.T/W if provided)
    p.add_argument("--T", type=float, default=None, help="SupCon temperature.")
    p.add_argument("--W", type=float, default=None, help="SupCon loss weight.")
    return p.parse_args()

# ================= YAML readers =================
def _to_float(x, default: Optional[float]) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return default

def load_TW_from_yaml(cfg_path: str) -> Tuple[Optional[float], Optional[float]]:
    # Try yacs first
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
        if supc is not None:
            t = _to_float(getattr(supc, "T", None), None)
            w = _to_float(getattr(supc, "W", None), None)
            if t is not None or w is not None:
                return t, w
    except Exception:
        pass

    # Fallback to raw YAML
    try:
        import yaml  # type: ignore
        with open(cfg_path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        loss = y.get("LOSS", {}) or {}
        supc = loss.get("SUPCON", {}) or {}
        t = _to_float(supc.get("T", None), None)
        w = _to_float(supc.get("W", None), None)
        if t is not None or w is not None:
            return t, w
    except Exception:
        pass

    return None, None

# ================= Metric parsing =================
def _latest_epoch_dir(out_test_dir: Path) -> Optional[Path]:
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
    for name in ("test_summary.txt", "results.txt", "log.txt", "test_log.txt"):
        p = epoch_dir / name
        if p.exists():
            s = p.read_text(encoding="utf-8", errors="ignore")
            mAP = _re_pick(s, r"mAP[^0-9]*([0-9]+(?:\.[0-9]+)?)")
            r1  = _re_pick(s, r"Rank[-\s]?1[^0-9]*([0-9]+(?:\.[0-9]+)?)")
            r5  = _re_pick(s, r"Rank[-\\s]?5[^0-9]*([0-9]+(?:\.[0-9]+)?)")
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

# ================= Progress & path helpers =================
def _max_test_epoch(test_dir: Path) -> int:
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

def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")  # minute precision

def _base_tag_fixed(t: float, w: float) -> str:
    return f"b1_supcon_fixed_T{_fmt(t)}_W{_fmt(w)}"

def _tag_with_seed(base: str, seed: int) -> str:
    return f"{base}_seed{seed}"

def _glob_latest_tag(base_with_seed: str, ds_prefix: str, log_root: Path) -> Optional[str]:
    """
    Find latest timestamped tag for a given base_with_seed.
    Matches: {ds_prefix}_{base_with_seed}_YYYYMMDD_HHMM_deit_run
    Returns the tag WITHOUT ds_prefix/suffix (just the TAG part).
    """
    pattern = f"{ds_prefix}_{base_with_seed}_????????_????_deit_run"
    cands = sorted(log_root.glob(pattern), key=lambda p: p.stat().st_mtime)
    if cands:
        name = cands[-1].name
        prefix = f"{ds_prefix}_"
        suffix = "_deit_run"
        if name.startswith(prefix) and name.endswith(suffix):
            return name[len(prefix):-len(suffix)]
    return None

def _run_dir_of_tag(tag: str, ds_prefix: str, log_root: Path) -> Path:
    return log_root / f"{ds_prefix}_{tag}_deit_run"

def _test_dir_of_tag(tag: str, ds_prefix: str, log_root: Path) -> Path:
    return log_root / f"{ds_prefix}_{tag}_deit_test"

# ================= Robust re-test =================
def eval_missing_epochs_via_test_py(full_tag: str, config_path: str, log_root: Path, ds_prefix: str, data_root: str) -> None:
    run_dir  = _run_dir_of_tag(full_tag, ds_prefix, log_root)
    test_dir = _test_dir_of_tag(full_tag, ds_prefix, log_root)
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
            "DATASETS.ROOT_DIR", data_root,
        ]
        print("[B1][eval] Launch:", " ".join(cmd))
        ret = subprocess.call(cmd)
        if ret != 0:
            print(f"[B1][eval][warn] test.py failed for epoch {ep} (ret={ret})")

# ================= Train launcher (timestamped TAG) =================
def _launch_training(config_path: str,
                     tag_base: str,
                     seed: int,
                     target_epochs: int,
                     log_root: Path,
                     ds_prefix: str,
                     data_root: str,
                     ckpt_period: int,
                     eval_period: int,
                     t_supcon: float,
                     w_supcon: float,
                     extra_opts: Optional[List[str]] = None) -> str:
    """
    Launch training with a timestamped TAG; returns the full TAG used (with timestamp).
    """
    timestamp = _now_stamp()
    full_tag = f"{_tag_with_seed(tag_base, seed)}_{timestamp}"

    run_dir  = _run_dir_of_tag(full_tag, ds_prefix, log_root)
    test_dir = _test_dir_of_tag(full_tag, ds_prefix, log_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    trained_max = _max_epoch_from_checkpoints(run_dir)

    opts = [
        "LOSS.SUPCON.ENABLE", "True",
        "LOSS.SUPCON.T", str(t_supcon),
        "LOSS.SUPCON.W", str(w_supcon),
        "SOLVER.MAX_EPOCHS", str(target_epochs),
        "SOLVER.SEED", str(seed),
        "SOLVER.CHECKPOINT_PERIOD", str(ckpt_period),
        "SOLVER.EVAL_PERIOD", str(eval_period),
        "DATASETS.ROOT_DIR", data_root,
        "OUTPUT_DIR", str(log_root),
        "TAG", full_tag,
    ]
    if extra_opts:
        opts += extra_opts

    if trained_max > 0:
        last_ckpt = run_dir / f"transformer_{trained_max}.pth"
        if last_ckpt.exists():
            print(f"[B1][resume] Found {last_ckpt}. Resume with explicit PRETRAIN_PATH. "
                  f"Override MAX_EPOCHS={target_epochs}.")
            opts += [
                "MODEL.PRETRAIN_CHOICE", "resume",
                "MODEL.PRETRAIN_PATH", str(last_ckpt),
            ]
        else:
            print(f"[B1][resume] Found progress markers but last ckpt missing; proceeding without explicit path.")

    cmd = [sys.executable, "run_modelB_deit.py", "--config", config_path, "--opts"] + opts
    print(f"[B1] Launch tag={full_tag} @ {timestamp}")
    print("[B1] CMD:", " ".join(cmd))
    subprocess.check_call(cmd)
    return full_tag

# ================= Full-run wrapper =================
def ensure_full_run_fixed(t_supcon: float, w_supcon: float, seed: int, epochs: int,
                          config: str, log_root: Path, ds_prefix: str, data_root: str,
                          ckpt_period: int, eval_period: int, extra_opts: Optional[List[str]]) -> Optional[Dict[str, Any]]:
    tag_base = _base_tag_fixed(t_supcon, w_supcon)
    base_with_seed = _tag_with_seed(tag_base, seed)

    latest_tag = _glob_latest_tag(base_with_seed, ds_prefix, log_root)
    if latest_tag:
        run_dir  = _run_dir_of_tag(latest_tag, ds_prefix, log_root)
        test_dir = _test_dir_of_tag(latest_tag, ds_prefix, log_root)

        tested_max = _max_test_epoch(test_dir)
        print(f"[B1][diagnose] FULL tested_max={tested_max} under {test_dir}")
        if tested_max >= epochs:
            best_ep = pick_best_epoch_metrics(test_dir)
            if best_ep is None:
                print(f"[B1][warn] Already tested to {tested_max}, but cannot parse metrics under: {test_dir}")
                return None
            print(f"[B1] FULL ALREADY-TESTED tag={latest_tag} (tested_max={tested_max}≥{epochs}) best_ep={best_ep}")
            return best_ep

        trained_max = _max_epoch_from_checkpoints(run_dir)
        if trained_max >= epochs:
            print(f"[B1] FULL trained_max={trained_max}≥{epochs}; ensure tests exist for tag={latest_tag}")
            eval_missing_epochs_via_test_py(latest_tag, config, log_root, ds_prefix, data_root)
        else:
            print(f"[B1] FULL RUN (trained_max={trained_max}/{epochs}) base={base_with_seed}")
            latest_tag = _launch_training(
                config, tag_base, seed, epochs, log_root, ds_prefix, data_root,
                ckpt_period, eval_period, t_supcon, w_supcon, extra_opts
            )
            eval_missing_epochs_via_test_py(latest_tag, config, log_root, ds_prefix, data_root)
    else:
        print(f"[B1] FULL RUN (fresh) base={base_with_seed}")
        latest_tag = _launch_training(
            config, tag_base, seed, epochs, log_root, ds_prefix, data_root,
            ckpt_period, eval_period, t_supcon, w_supcon, extra_opts
        )
        eval_missing_epochs_via_test_py(latest_tag, config, log_root, ds_prefix, data_root)

    test_dir = _test_dir_of_tag(latest_tag, ds_prefix, log_root)
    best_ep = pick_best_epoch_metrics(test_dir)
    if best_ep is None:
        print(f"[B1][warn] No valid epoch metrics found under: {test_dir}")
        return None
    print(f"[B1] Seed {seed} best epoch: {best_ep}")
    return best_ep

# ================= Main =================
def main():
    args = build_cli()

    # Resolve roots
    global LOG_ROOT, DATA_ROOT
    if args.output_root:
        LOG_ROOT = Path(args.output_root)
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
    if args.data_root:
        DATA_ROOT = args.data_root

    # Parse seeds
    seed_list = [int(s) for s in re.split(r"[,\s]+", args.seeds) if s.strip() != ""]
    if not seed_list:
        raise SystemExit("[B1] No valid seeds given.")

    # Shared settings
    EPOCHS      = int(args.epochs)
    CKPT_PERIOD = int(args.save_every)
    EVAL_PERIOD = int(args.save_every)
    CONFIG      = args.config
    DS_PREFIX   = args.ds_prefix

    # Determine SupCon T/W: CLI > YAML > default
    cli_T = args.T
    cli_W = args.W
    yaml_T, yaml_W = load_TW_from_yaml(CONFIG)
    T_use = float(cli_T if cli_T is not None else (yaml_T if yaml_T is not None else DEFAULT_T))
    W_use = float(cli_W if cli_W is not None else (yaml_W if yaml_W is not None else DEFAULT_W))

    print(f"[B1-CLI] CONFIG={CONFIG}")
    print(f"[B1-CLI] EPOCHS={EPOCHS} SAVE_EVERY={CKPT_PERIOD}")
    print(f"[B1-CLI] SEEDS={seed_list}")
    print(f"[B1-CLI] OUTPUT_ROOT={LOG_ROOT}")
    print(f"[B1-CLI] DATA_ROOT={DATA_ROOT}")
    print(f"[B1-CLI] DS_PREFIX={DS_PREFIX}")
    print(f"[B1-CLI] SupCon T={T_use} W={W_use}")

    # Extra YACS opts passthrough (LOSS.SUPCON.ENABLE/T/W are injected by launcher)
    extra_opts = (args.opts or [])

    # Full training for each seed with fixed T/W
    seed_best_records: Dict[int, Dict[str, Any]] = {}
    for seed in seed_list:
        best_ep = ensure_full_run_fixed(
            T_use, W_use, seed, EPOCHS,
            config=CONFIG, log_root=LOG_ROOT, ds_prefix=DS_PREFIX, data_root=DATA_ROOT,
            ckpt_period=CKPT_PERIOD, eval_period=EVAL_PERIOD, extra_opts=extra_opts
        )
        if best_ep is not None:
            seed_best_records[seed] = best_ep

    # Summarize seeds
    if seed_best_records:
        mAPs  = [rec["mAP"] for rec in seed_best_records.values()]
        R1s   = [rec["Rank-1"] for rec in seed_best_records.values()]
        R5s   = [rec["Rank-5"] for rec in seed_best_records.values()]
        R10s  = [rec["Rank-10"] for rec in seed_best_records.values()]
        summary = {
            "T": T_use,
            "W": W_use,
            "seeds": {str(k): v for k, v in seed_best_records.items()},
            "mean": {
                "mAP":   round(safe_mean(mAPs), 4),
                "Rank-1": round(safe_mean(R1s), 4),
                "Rank-5": round(safe_mean(R5s), 4),
                "Rank-10": round(safe_mean(R10s), 4),
            },
            "std": {
                "mAP":   round(safe_stdev(mAPs), 4),
                "Rank-1": round(safe_stdev(R1s), 4),
                "Rank-5": round(safe_stdev(R5s), 4),
                "Rank-10": round(safe_stdev(R10s), 4),
            },
            "note": "Each seed uses its best epoch by mAP (tie-breaker Rank-1). "
                    "Means/std computed over seeds with valid metrics.",
        }
        SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
        print(f"[B1] Wrote summary → {SUMMARY_JSON}")
    else:
        print("[B1][warn] No seed best records collected; skip summary.")

    print(f"[B1] Fixed-parameter full runs finished under {LOG_ROOT}")

if __name__ == "__main__":
    main()
