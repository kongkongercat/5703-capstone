#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===========================================================
# File: run_ce_triplet.py
# Purpose: Baseline (CrossEntropy + Triplet) training launcher
#          with unified auto-naming and FEAT_SRC layer info.
# Author: Hang Zhang (hzha0521)
# ===========================================================
# Change Log
# [2025-10-17 | Hang Zhang] Unified structure with run_b1_nogrid.py:
#   - CLI for epochs/save-every/seeds/config/output-root/data-root/ds-prefix/opts
#   - Auto-tag includes FEAT_SRC for CE/TRI:
#       {ds_prefix}_ce_triplet_CE-{ce}_TRI-{tri}_seed{S}_{YYYYMMDD_HHMM}_{run|test}
#   - Resume-safe (detect latest transformer_*.pth)
#   - Create run/test folders (no auto-test by default)
# ===========================================================

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# ================= Defaults =================
DEFAULT_CONFIG      = "configs/VeRi/deit_transreid_stride_b0_baseline.yml"
DEFAULT_EPOCHS      = 120
DEFAULT_SAVE_EVERY  = 3
DEFAULT_SEEDS       = "0,1,2"
DEFAULT_DSPREFIX    = "veri776"

# ================= Env / paths =================
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
    return str(Path.cwd() / "datasets")

DATA_ROOT = detect_data_root()

# ================= CLI =================
def build_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CE + Triplet baseline with unified auto-naming (layers included).")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    p.add_argument("--seeds", type=str, default=DEFAULT_SEEDS, help="comma/space separated, e.g. '0,1,2'")
    p.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    p.add_argument("--output-root", type=str, default=None)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--ds-prefix", type=str, default=DEFAULT_DSPREFIX)
    p.add_argument("--opts", nargs=argparse.REMAINDER, help="extra YACS options passed to run_modelB_deit.py")
    return p.parse_args()

# ================= Helpers =================
def _max_epoch_from_checkpoints(run_dir: Path) -> int:
    max_ep = 0
    if not run_dir.exists():
        return 0
    for p in run_dir.glob("transformer_*.pth"):
        m = re.search(r"transformer_(\d+)\.pth", p.name)
        if m:
            max_ep = max(max_ep, int(m.group(1)))
    return max_ep

def _pick_src(opt_list, key, default):
    """Scan --opts list to pick the value after 'key'."""
    if not opt_list:
        return default
    for i, v in enumerate(opt_list[:-1]):
        if str(v).strip() == key:
            return str(opt_list[i + 1]).strip()
    return default

# ================= Train launcher =================
def _launch_training(config_path: str,
                     seed: int,
                     target_epochs: int,
                     log_root: Path,
                     ds_prefix: str,
                     data_root: str,
                     ckpt_period: int,
                     eval_period: int,
                     extra_opts):
    timestamp = _now_stamp()

    # Parse FEAT_SRC from --opts for naming
    ce_src  = _pick_src(extra_opts, "LOSS.CE.FEAT_SRC", "bnneck")
    tri_src = _pick_src(extra_opts, "LOSS.TRIPLET.FEAT_SRC", "pre_bn")

    full_tag = f"ce_triplet_CE-{ce_src}_TRI-{tri_src}_seed{seed}_{timestamp}"
    print(f"[CE+Triplet][naming] full_tag={full_tag}")

    run_dir  = log_root / f"{ds_prefix}_{full_tag}_deit_run"
    test_dir = log_root / f"{ds_prefix}_{full_tag}_deit_test"
    run_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    trained_max = _max_epoch_from_checkpoints(run_dir)

    opts = [
        "SOLVER.MAX_EPOCHS", str(target_epochs),
        "SOLVER.SEED", str(seed),
        "SOLVER.CHECKPOINT_PERIOD", str(ckpt_period),
        "SOLVER.EVAL_PERIOD", str(eval_period),   # 与 B1 对齐；若不想边训边测可改成 "0"
        "DATASETS.ROOT_DIR", data_root,
        "OUTPUT_DIR", str(log_root),
        "TAG", full_tag,
    ]
    if extra_opts:
        opts += extra_opts

    if trained_max > 0:
        last_ckpt = run_dir / f"transformer_{trained_max}.pth"
        if last_ckpt.exists():
            print(f"[CE+Triplet][resume] Found {last_ckpt}. Resume training.")
            opts += [
                "MODEL.PRETRAIN_CHOICE", "resume",
                "MODEL.PRETRAIN_PATH", str(last_ckpt),
            ]

    cmd = [sys.executable, "run_modelB_deit.py", "--config", config_path, "--opts"] + opts
    print(f"[CE+Triplet] Launch tag={full_tag} @ {timestamp}")
    print("[CE+Triplet] CMD:", " ".join(cmd))
    subprocess.check_call(cmd)
    return full_tag

# ================= Wrapper =================
def ensure_full_run(seed: int, epochs: int, config: str, log_root: Path,
                    ds_prefix: str, data_root: str, ckpt_period: int,
                    eval_period: int, extra_opts):
    return _launch_training(config, seed, epochs, log_root, ds_prefix, data_root, ckpt_period, eval_period, extra_opts)

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
        raise SystemExit("[CE+Triplet] No valid seeds given.")

    print(f"[CE+Triplet-CLI] CONFIG={args.config}")
    print(f"[CE+Triplet-CLI] EPOCHS={args.epochs}")
    print(f"[CE+Triplet-CLI] SEEDS={seed_list}")
    print(f"[CE+Triplet-CLI] OUTPUT_ROOT={LOG_ROOT}")
    print(f"[CE+Triplet-CLI] DATA_ROOT={DATA_ROOT}")
    print(f"[CE+Triplet-CLI] DS_PREFIX={args.ds_prefix}")

    extra_opts = (args.opts or [])

    for seed in seed_list:
        ensure_full_run(
            seed, args.epochs, args.config,
            log_root=LOG_ROOT,
            ds_prefix=args.ds_prefix,
            data_root=DATA_ROOT,
            ckpt_period=args.save_every,
            eval_period=args.save_every,
            extra_opts=extra_opts
        )

    print(f"[CE+Triplet] Finished all seeds under {LOG_ROOT}")

if __name__ == "__main__":
    main()
