#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Hang Zhang (hzha0521)
# File: run_tripletx_only.py
# Desc: Run TripletX-only on VeRi (DeiT-TransReID). SupCon disabled; TripletX enabled.
# Date: 2025-10-11

"""
Run TripletX-only on VeRi (DeiT-TransReID)
- SupCon disabled
- TripletX enabled
- Save/Eval every epoch
- Sydney time in folder name
- Writes logs/checkpoints to Google Drive by default
"""
import argparse, os, sys, subprocess
from pathlib import Path
from datetime import datetime
import pytz

# -------- defaults --------
CFG = "configs/VeRi/deit_transreid_stride_b2_supcon_tripletx.yml"
DRIVE_ROOT = Path("/content/drive/MyDrive/5703(hzha0521)/Optimized Model B (transformer based)")
DATASET_CANDIDATES = [
    DRIVE_ROOT / "datasets",
    Path("/content/datasets"),
    Path("/workspace/datasets"),
]

def detect_data_root() -> Path:
    for d in DATASET_CANDIDATES:
        if (d / "VeRi").exists():
            return d
    return Path.cwd() / "datasets"

def sydney_stamp(fmt="%Y%m%d_%H%M"):
    tz = pytz.timezone("Australia/Sydney")
    return datetime.now(tz).strftime(fmt)

def build_cli():
    p = argparse.ArgumentParser("TripletX-only launcher")
    p.add_argument("--config", default=CFG)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=4, help="DATALOADER.NUM_INSTANCE (K)")
    p.add_argument("--output-root", default=str(DRIVE_ROOT / "logs"))
    p.add_argument("--tag", default=None, help="custom tag for folder names")
    p.add_argument("--every-epoch-save", action="store_true", default=True)
    return p.parse_args()

def main():
    args = build_cli()

    # resolve paths
    repo = Path.cwd()
    config = Path(args.config)
    if not config.exists():
        raise SystemExit(f"[X] Config not found: {config}")

    data_root = detect_data_root()
    if not (data_root / "VeRi").exists():
        raise SystemExit(f"[X] Dataset not found under: {data_root/'VeRi'}")

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    tag = args.tag or f"tripletx_only_seed{args.seed}_{sydney_stamp()}"
    run_dir = out_root / f"veri776_{tag}_deit_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    runner = "run_modelB_deit.py"
    if not (repo / runner).exists():
        runner = "run_b2.py"
        if not (repo / runner).exists():
            raise SystemExit("[X] Neither run_modelB_deit.py nor run_b2.py found in repo root.")

    cmd = [
        sys.executable, runner,
        "--config", str(config),
        "--opts",
        "LOSS.SUPCON.ENABLE","False",
        "LOSS.SUPCON.W","0.0",
        "LOSS.TRIPLETX.ENABLE","True",
        "MODEL.TRAINING_MODE","supervised",
        "DATALOADER.NUM_INSTANCE", str(args.k),
        "SOLVER.SEED", str(args.seed),
        "SOLVER.MAX_EPOCHS", str(args.epochs),
        "SOLVER.CHECKPOINT_PERIOD", "1",
        "SOLVER.EVAL_PERIOD", "1",
        "TEST.RE_RANKING", "False",
        "DATASETS.ROOT_DIR", str(data_root),
        "OUTPUT_DIR", str(out_root),
        "TAG", f"{tag}",
    ]

    print("[RUN] CWD       :", repo)
    print("[RUN] runner    :", runner)
    print("[RUN] data_root :", data_root)
    print("[RUN] output    :", run_dir)
    print("[RUN] CMD       :", " ".join(cmd))
    subprocess.check_call(cmd)

if __name__ == "__main__":
    main()
