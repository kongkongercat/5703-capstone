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
#                             and seed summary logic.
# [2025-10-15 | Hang Zhang] Tag format update:
#                           - Use TAG base "b1_supcon" (no "_fixed" nor "_T/_W").
#                           - Final dirs: {DS_PREFIX}_b1_supcon_seed{S}_{YYYYMMDD_HHMM}_{...}
# [2025-10-17 | Hang Zhang]  Auto naming enhancement:
#                           - Output folder name includes FEAT_SRC (CE/TRI/SUP)
#                           - Pattern:
#                             {DS_PREFIX}_b1_supcon_CE-{ce}_TRI-{tri}_SUP-{sup}_seed{S}_{YYYYMMDD_HHMM}_deit_run
#                           - Added parsing of --opts for FEAT_SRC values.
#                           - Added [B1][naming] print confirmation.
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

# ================= Defaults =================
DEFAULT_CONFIG      = "configs/VeRi/deit_transreid_stride_b1_supcon.yml"
DEFAULT_EPOCHS      = 120
DEFAULT_SAVE_EVERY  = 1
DEFAULT_SEEDS       = "0,1,2"
DEFAULT_DSPREFIX    = "veri776"
DEFAULT_T           = 0.07
DEFAULT_W           = 0.30

TEST_MARKER_FILES_DEFAULT = (
    "summary.json", "test_summary.txt", "results.txt", "log.txt",
    "test_log.txt", "dist_mat.npy"
)

# ================= Environment =================
def _in_colab() -> bool:
    try:
        import google.colab  # noqa
        return True
    except Exception:
        return False

def pick_log_root() -> Path:
    env = os.getenv("OUTPUT_ROOT")
    if env:
        return Path(env)
    if _in_colab():
        try:
            from google.colab import drive
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

def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")

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
    p = argparse.ArgumentParser(description="Run B1 SupCon (no grid) with fixed T/W.")
    p.add_argument("--test", dest="test", action="store_true")
    p.add_argument("--no-test", dest="test", action="store_false")
    p.set_defaults(test=True)

    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    p.add_argument("--seeds", type=str, default=DEFAULT_SEEDS)
    p.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    p.add_argument("--output-root", type=str, default=None)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--ds-prefix", type=str, default=DEFAULT_DSPREFIX)
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--opts", nargs=argparse.REMAINDER)
    p.add_argument("--T", type=float, default=None)
    p.add_argument("--W", type=float, default=None)
    return p.parse_args()

# ================= YAML Readers =================
def _to_float(x, default: Optional[float]) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return default

def load_TW_from_yaml(cfg_path: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        import yaml
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

# ================= Metric Parsing =================
def _latest_epoch_dir(out_test_dir: Path) -> Optional[Path]:
    epochs = sorted([p for p in out_test_dir.glob("epoch_*") if p.is_dir()],
                    key=lambda p: int(p.name.split("_")[-1]))
    return epochs[-1] if epochs else None

def parse_metrics(out_test_dir: Path) -> Tuple[float, float, float, float]:
    ep = _latest_epoch_dir(out_test_dir)
    if not ep:
        return -1.0, -1.0, -1.0, -1.0
    sj = ep / "summary.json"
    if sj.exists():
        try:
            obj = json.loads(sj.read_text())
            return (
                float(obj.get("mAP", -1)),
                float(obj.get("Rank-1", obj.get("Rank1", -1))),
                float(obj.get("Rank-5", obj.get("Rank5", -1))),
                float(obj.get("Rank-10", obj.get("Rank10", -1)))
            )
        except Exception:
            pass
    return -1.0, -1.0, -1.0, -1.0

def safe_mean(values: List[float]) -> float:
    vals = [v for v in values if v >= 0]
    return round(stats.mean(vals), 4) if vals else -1.0

def safe_stdev(values: List[float]) -> float:
    vals = [v for v in values if v >= 0]
    return round(stats.stdev(vals), 4) if len(vals) >= 2 else 0.0

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
    for p in run_dir.glob("transformer_*.pth"):
        m = re.search(r"transformer_(\d+)\.pth", p.name)
        if m:
            max_ep = max(max_ep, int(m.group(1)))
    return max_ep

# ================= Train launcher =================
def _launch_training(config_path: str,
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
    """Launch training with timestamped TAG including FEAT_SRC."""
    timestamp = _now_stamp()

    def _pick_src(opt_list, key, default="bnneck"):
        if not opt_list:
            return default
        for i, v in enumerate(opt_list[:-1]):
            if v.strip() == key:
                return opt_list[i + 1]
        return default

    ce_src  = _pick_src(extra_opts, "LOSS.CE.FEAT_SRC", "bnneck")
    tri_src = _pick_src(extra_opts, "LOSS.TRIPLET.FEAT_SRC", "pre_bn")
    sup_src = _pick_src(extra_opts, "LOSS.SUPCON.FEAT_SRC", "bnneck")

    full_tag = f"b1_supcon_CE-{ce_src}_TRI-{tri_src}_SUP-{sup_src}_seed{seed}_{timestamp}"
    print(f"[B1][naming] full_tag={full_tag}")

    run_dir  = log_root / f"{ds_prefix}_{full_tag}_deit_run"
    test_dir = log_root / f"{ds_prefix}_{full_tag}_deit_test"
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
            print(f"[B1][resume] Found {last_ckpt}. Resume training.")
            opts += [
                "MODEL.PRETRAIN_CHOICE", "resume",
                "MODEL.PRETRAIN_PATH", str(last_ckpt),
            ]

    cmd = [sys.executable, "run_modelB_deit.py", "--config", config_path, "--opts"] + opts
    print(f"[B1] Launch tag={full_tag} @ {timestamp}")
    print("[B1] CMD:", " ".join(cmd))
    subprocess.check_call(cmd)
    return full_tag

# ================= Wrapper =================
def ensure_full_run_fixed(t_supcon: float, w_supcon: float, seed: int, epochs: int,
                          config: str, log_root: Path, ds_prefix: str, data_root: str,
                          ckpt_period: int, eval_period: int, extra_opts: Optional[List[str]]) -> None:
    print(f"[B1] Full run seed={seed}, SupCon T={t_supcon}, W={w_supcon}")
    return _launch_training(
        config, seed, epochs, log_root, ds_prefix, data_root,
        ckpt_period, eval_period, t_supcon, w_supcon, extra_opts
    )

# ================= Main =================
def main():
    args = build_cli()

    global LOG_ROOT, DATA_ROOT
    if args.output_root:
        LOG_ROOT = Path(args.output_root)
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
    if args.data_root:
        DATA_ROOT = args.data_root

    seed_list = [int(s) for s in re.split(r"[,\s]+", args.seeds) if s.strip()]
    if not seed_list:
        raise SystemExit("[B1] No valid seeds given.")

    EPOCHS      = int(args.epochs)
    CKPT_PERIOD = int(args.save_every)
    EVAL_PERIOD = int(args.save_every)
    CONFIG      = args.config
    DS_PREFIX   = args.ds_prefix
    cli_T, cli_W = args.T, args.W
    yaml_T, yaml_W = load_TW_from_yaml(CONFIG)
    T_use = float(cli_T if cli_T is not None else (yaml_T if yaml_T is not None else DEFAULT_T))
    W_use = float(cli_W if cli_W is not None else (yaml_W if yaml_W is not None else DEFAULT_W))

    print(f"[B1-CLI] CONFIG={CONFIG}")
    print(f"[B1-CLI] EPOCHS={EPOCHS}")
    print(f"[B1-CLI] SEEDS={seed_list}")
    print(f"[B1-CLI] OUTPUT_ROOT={LOG_ROOT}")
    print(f"[B1-CLI] DATA_ROOT={DATA_ROOT}")
    print(f"[B1-CLI] DS_PREFIX={DS_PREFIX}")
    print(f"[B1-CLI] SupCon T={T_use} W={W_use}")

    extra_opts = (args.opts or [])
    for seed in seed_list:
        ensure_full_run_fixed(
            T_use, W_use, seed, EPOCHS,
            config=CONFIG, log_root=LOG_ROOT, ds_prefix=DS_PREFIX,
            data_root=DATA_ROOT, ckpt_period=CKPT_PERIOD,
            eval_period=EVAL_PERIOD, extra_opts=extra_opts
        )

    print(f"[B1] Finished all seeds under {LOG_ROOT}")

if __name__ == "__main__":
    main()
