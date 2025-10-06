#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================
# File: run_b4_fine.py
# Purpose: Fully independent B4 pipeline (Self-Supervised + Fine-tune)
# Author: Fanyi Meng (2025-10-06)
# ==========================================
# Change Log
# [2025-10-06 | Fanyi Meng] Independent two-stage version (no external JSON).
# ==========================================

import subprocess
import sys
import time
from pathlib import Path
import os

# ==========================
# BASIC SETTINGS
# ==========================
CONFIG_SSL = "configs/VeRi/deit_transreid_stride_b3_ssl_pretrain.yml"
CONFIG_FINE = "configs/VeRi/deit_transreid_stride_b4_ssl_finetune.yml"

EPOCHS_SSL = 10          # SupCon pretrain
EPOCHS_FINE = 10         # Fine-tune
BATCH = 64
SEED = 0

# ==========================
# PATHS (adaptable to Colab)
# ==========================
def pick_root():
    if Path("/content/drive/MyDrive").exists():
        root = Path("/content/drive/MyDrive/5703(hzha0521)/Optimized Model B (transformer based)")
    else:
        root = Path(".")
    return root.resolve()

ROOT = pick_root()
LOG_ROOT = ROOT / "logs"
PRETRAINED_PATH = ROOT / "pretrained/deit_base_distilled_patch16_224-df68dfff.pth"
DATASET_ROOT = ROOT / "datasets"
LOG_ROOT.mkdir(parents=True, exist_ok=True)

# ==========================
# HELPER
# ==========================
def run_stage(cmd, stage_name):
    print(f"\n===== [B4] Starting {stage_name} =====\n")
    print(" ".join(cmd))
    subprocess.check_call(cmd)
    print(f"\n===== [B4] Finished {stage_name} =====\n")


# ==========================
# STAGE 1: SSL PRETRAIN
# ==========================
def stage1_ssl_pretrain():
    tag = f"b4_stage1_ssl_{time.strftime('%m%d-%H%M')}"
    run_cmd = [
        sys.executable, "run_modelB_deit.py",
        "--config", CONFIG_SSL,
        "--epochs", str(EPOCHS_SSL),
        "--batch", str(BATCH),
        "--data-root", str(DATASET_ROOT),
        "--pretrained", str(PRETRAINED_PATH),
        "--tag", tag,
        "--opts",
        # SSL mode
        "MODEL.PRETRAIN_CHOICE", "imagenet",
        "MODEL.TRAINING_MODE", "self_supervised",
        "LOSS.SUPCON.ENABLE", "True",
        "LOSS.CE.ENABLE", "False",
        "LOSS.TRIPLETX.ENABLE", "False",
        "SOLVER.SEED", str(SEED),
        "OUTPUT_DIR", str(LOG_ROOT),
    ]
    run_stage(run_cmd, "Stage 1: Self-Supervised Pretraining")

    ckpt_dir = LOG_ROOT / f"veri776_{tag}_deit_run"
    ckpts = sorted(ckpt_dir.glob("transformer_*.pth"))
    if not ckpts:
        raise SystemExit(f"[B4] No SSL checkpoint found in {ckpt_dir}")
    return ckpts[-1]


# ==========================
# STAGE 2: FINE-TUNE
# ==========================
def stage2_finetune(ssl_ckpt):
    tag = f"b4_stage2_finetune_{time.strftime('%m%d-%H%M')}"
    run_cmd = [
        sys.executable, "run_modelB_deit.py",
        "--config", CONFIG_FINE,
        "--epochs", str(EPOCHS_FINE),
        "--batch", str(BATCH),
        "--data-root", str(DATASET_ROOT),
        "--pretrained", str(PRETRAINED_PATH),
        "--tag", tag,
        "--opts",
        # Fine-tune mode
        "MODEL.PRETRAIN_CHOICE", "finetune",
        "MODEL.PRETRAIN_PATH", str(ssl_ckpt),
        "MODEL.TRAINING_MODE", "supervised",
        "LOSS.SUPCON.ENABLE", "False",
        "LOSS.CE.ENABLE", "True",
        "LOSS.TRIPLETX.ENABLE", "True",
        "SOLVER.SEED", str(SEED),
        "OUTPUT_DIR", str(LOG_ROOT),
    ]
    run_stage(run_cmd, "Stage 2: Supervised Fine-tuning")


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    print("[B4] ========================================")
    print("[B4] Independent Two-Stage Training Pipeline")
    print("[B4] Stage 1: Self-supervised (SupCon)")
    print("[B4] Stage 2: Fine-tuning (CE + TripletX)")
    print("[B4] ========================================\n")

    ssl_ckpt = stage1_ssl_pretrain()
    print(f"[B4] ✅ SSL Pretraining Done. Using checkpoint: {ssl_ckpt}")

    stage2_finetune(ssl_ckpt)
    print("[B4] ✅ All Done. Check logs under:", LOG_ROOT)
