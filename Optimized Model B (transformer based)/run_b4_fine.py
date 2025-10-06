#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================
# File: run_b4_fine.py
# Purpose: Independent two-stage pipeline:
#   Stage 1: SSL pretraining with SupCon ONLY
#   Stage 2: Joint fine-tuning with CE + TripletX + SupCon
# Notes:
#   - No grid search; fixed hyper-params.
#   - Works in Colab + local. Uses run_modelB_deit.py as the launcher.
# ==========================================

import subprocess
import sys
import time
from pathlib import Path

# ---------- User knobs (feel free to tweak) ----------
# Epochs
EPOCHS_SSL  = 10   # Stage 1: SupCon-only SSL
EPOCHS_FINE = 10   # Stage 2: Joint fine-tuning

# Batches / workers (reduce if OOM)
BATCH = 64
NUM_WORKERS = 8

# Seeds
SEED = 0

# SupCon hyper-params
SSL_T = 0.07
SSL_W = 0.30          # Stage 1
FT_T  = 0.07
FT_W  = 0.20          # Stage 2 (smaller weight for joint training)

# Paths
CONFIG_SSL  = "configs/VeRi/deit_transreid_stride_b3_ssl_pretrain.yml"
CONFIG_FINE = "configs/VeiRi/deit_transreid_stride_b4_ssl_finetune.yml".replace("VeiRi","VeRi")

def _in_colab() -> bool:
    try:
        import google.colab  # noqa
        return True
    except Exception:
        return False

def pick_root() -> Path:
    if _in_colab() and Path("/content/drive/MyDrive").exists():
        return Path("/content/drive/MyDrive/5703(hzha0521)/Optimized Model B (transformer based)")
    return Path(".").resolve()

ROOT = pick_root()
LOG_ROOT = ROOT / "logs"
DATASET_ROOT = ROOT / "datasets"
PRETRAINED_PATH = ROOT / "pretrained/deit_base_distilled_patch16_224-df68dfff.pth"
LOG_ROOT.mkdir(parents=True, exist_ok=True)

def run_and_print(cmd: list[str], title: str):
    print(f"\n===== [B4] {title} =====")
    print(" ".join(cmd))
    ret = subprocess.call(cmd)
    if ret != 0:
        raise SystemExit(f"[B4] ERROR: command failed with code {ret}")

def stage1_ssl() -> Path:
    """Stage 1: SupCon-only self-supervised pretraining."""
    tag = f"b4_sslonly_T{str(SSL_T).replace('.','p')}_W{str(SSL_W).replace('.','p')}_{time.strftime('%m%d-%H%M')}"
    cmd = [
        sys.executable, "run_modelB_deit.py",
        "--config", CONFIG_SSL,
        "--epochs", str(EPOCHS_SSL),
        "--batch", str(BATCH),
        "--num-workers", str(NUM_WORKERS),
        "--data-root", str(DATASET_ROOT),
        "--pretrained", str(PRETRAINED_PATH),
        "--tag", tag,
        "--opts",
        # enforce SSL-only training
        "MODEL.PRETRAIN_CHOICE", "imagenet",
        "MODEL.TRAINING_MODE", "self_supervised",

        "LOSS.SUPCON.ENABLE", "True",
        "LOSS.SUPCON.T", str(SSL_T),
        "LOSS.SUPCON.W", str(SSL_W),

        # make sure supervised losses are off
        "LOSS.CE.ENABLE", "False",
        "LOSS.TRIPLETX.ENABLE", "False",
        "MODEL.ID_LOSS_TYPE", "none",
        "MODEL.ID_LOSS_WEIGHT", "0.0",
        "MODEL.METRIC_LOSS_TYPE", "none",
        "MODEL.TRIPLET_LOSS_WEIGHT", "0.0",

        # logging
        "SOLVER.SEED", str(SEED),
        "OUTPUT_DIR", str(LOG_ROOT),
    ]
    run_and_print(cmd, "Stage 1: SSL (SupCon-only)")
    run_dir = LOG_ROOT / f"veri776_{tag}_deit_run"
    ckpts = sorted(run_dir.glob("transformer_*.pth"))
    if not ckpts:
        raise SystemExit(f"[B4] No checkpoint found under: {run_dir}")
    return ckpts[-1]

def stage2_joint_finetune(ssl_ckpt: Path):
    """Stage 2: joint fine-tune with CE + TripletX + SupCon."""
    tag = f"b4_jointft_T{str(FT_T).replace('.','p')}_W{str(FT_W).replace('.','p')}_{time.strftime('%m%d-%H%M')}"
    cmd = [
        sys.executable, "run_modelB_deit.py",
        "--config", CONFIG_FINE,
        "--epochs", str(EPOCHS_FINE),
        "--batch", str(BATCH),
        "--num-workers", str(NUM_WORKERS),
        "--data-root", str(DATASET_ROOT),
        "--pretrained", str(PRETRAINED_PATH),
        "--tag", tag,
        "--opts",
        # load from SSL checkpoint and run supervised + joint losses
        "MODEL.PRETRAIN_CHOICE", "finetune",
        "MODEL.PRETRAIN_PATH", str(ssl_ckpt),
        "MODEL.TRAINING_MODE", "supervised",

        # CE + TripletX + SupCon (joint)
        "LOSS.CE.ENABLE", "True",
        "LOSS.TRIPLETX.ENABLE", "True",
        "LOSS.SUPCON.ENABLE", "True",
        "LOSS.SUPCON.T", str(FT_T),
        "LOSS.SUPCON.W", str(FT_W),

        # logging
        "SOLVER.SEED", str(SEED),
        "OUTPUT_DIR", str(LOG_ROOT),
    ]
    run_and_print(cmd, "Stage 2: Joint fine-tune (CE + TripletX + SupCon)")

def main():
    print("[B4] ==========================================")
    print("[B4] Two-stage pipeline: SSL --> Joint Fine-tune")
    print("[B4]  - Stage 1: SupCon-only")
    print("[B4]  - Stage 2: CE + TripletX + SupCon")
    print("[B4] Logs:", LOG_ROOT)
    print("[B4] ==========================================")

    ssl_ckpt = stage1_ssl()
    print(f"[B4] Using SSL checkpoint: {ssl_ckpt}")

    stage2_joint_finetune(ssl_ckpt)
    print("[B4] Done. Check runs in:", LOG_ROOT)

if __name__ == "__main__":
    main()
