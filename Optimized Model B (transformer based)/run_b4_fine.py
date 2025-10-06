#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================
# File: run_b4_fine.py
# Purpose: Two-stage pipeline using B4 config
#   Stage 1: SupCon-only SSL pretraining
#   Stage 2: Joint fine-tuning (CE + TripletX + SupCon)
# ==========================================

import subprocess
import sys
import time
import re
from pathlib import Path


# ---------- Basic Settings ----------
EPOCHS_SSL = 10     # Stage 1: self-supervised
EPOCHS_FINE = 10    # Stage 2: joint fine-tune
BATCH = 64
NUM_WORKERS = 8
SEED = 0

# SupCon hyper-parameters
SSL_T, SSL_W = 0.07, 0.30
FT_T, FT_W = 0.07, 0.20

# ---------- Paths ----------
def _in_colab():
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
CONFIG_B4 = "configs/VeRi/deit_transreid_stride_b4_ssl_finetune.yml"
LOG_ROOT.mkdir(parents=True, exist_ok=True)


# ---------- Utility ----------
def run_and_log(cmd: list[str], title: str):
    print(f"\n===== [B4] {title} =====")
    print(" ".join(cmd))
    ret = subprocess.call(cmd)
    if ret != 0:
        raise SystemExit(f"[B4] ERROR: command failed with code {ret}")


def find_best_checkpoint(run_dir: Path):
    """Parse log.txt to find checkpoint with best mAP."""
    log_file = run_dir / "log.txt"
    if not log_file.exists():
        print("[B4] ⚠️ No log file found, fallback to last checkpoint.")
        ckpts = sorted(run_dir.glob("transformer_*.pth"))
        return ckpts[-1] if ckpts else None

    best_epoch, best_map = None, -1.0
    for line in log_file.read_text().splitlines():
        match = re.search(r"Epoch\[(\d+)\].*mAP:\s*([\d.]+)%", line)
        if match:
            epoch, cur_map = int(match.group(1)), float(match.group(2))
            if cur_map > best_map:
                best_epoch, best_map = epoch, cur_map

    if best_epoch:
        best_ckpt = run_dir / f"transformer_{best_epoch}.pth"
        if best_ckpt.exists():
            print(f"[B4] ✅ Best checkpoint: epoch {best_epoch} (mAP={best_map:.2f}%)")
            return best_ckpt
    print("[B4] ⚠️ No valid checkpoint found, using last one.")
    ckpts = sorted(run_dir.glob("transformer_*.pth"))
    return ckpts[-1] if ckpts else None


# ---------- Stage 1: SupCon-only ----------
def run_stage1_ssl():
    tag = f"b4_sslonly_T{str(SSL_T).replace('.', 'p')}_W{str(SSL_W).replace('.', 'p')}_{time.strftime('%m%d-%H%M')}"
    cmd = [
        sys.executable, "run_modelB_deit.py",
        "--config", CONFIG_B4,
        "--epochs", str(EPOCHS_SSL),
        "--batch", str(BATCH),
        "--num-workers", str(NUM_WORKERS),
        "--data-root", str(DATASET_ROOT),
        "--pretrained", str(PRETRAINED_PATH),
        "--tag", tag,
        "--opts",
        "MODEL.PRETRAIN_CHOICE", "imagenet",
        "MODEL.TRAINING_MODE", "self_supervised",
        "LOSS.SUPCON.ENABLE", "True",
        "LOSS.SUPCON.T", str(SSL_T),
        "LOSS.SUPCON.W", str(SSL_W),
        "LOSS.TRIPLETX.ENABLE", "False",
        "LOSS.CE.ENABLE", "False",
        "MODEL.METRIC_LOSS_TYPE", "none",
        "MODEL.ID_LOSS_TYPE", "none",
        "MODEL.ID_LOSS_WEIGHT", "0.0",
        "MODEL.TRIPLET_LOSS_WEIGHT", "0.0",
        "SOLVER.SEED", str(SEED),
        "OUTPUT_DIR", str(LOG_ROOT),
    ]
    run_and_log(cmd, "Stage 1: SupCon-only SSL")
    run_dir = LOG_ROOT / f"veri776_{tag}_deit_run"
    ckpts = sorted(run_dir.glob("transformer_*.pth"))
    return run_dir if ckpts else None


# ---------- Stage 2: Joint fine-tuning ----------
def run_stage2_joint(best_ckpt: Path):
    tag = f"b4_jointft_T{str(FT_T).replace('.', 'p')}_W{str(FT_W).replace('.', 'p')}_{time.strftime('%m%d-%H%M')}"
    cmd = [
        sys.executable, "run_modelB_deit.py",
        "--config", CONFIG_B4,
        "--epochs", str(EPOCHS_FINE),
        "--batch", str(BATCH),
        "--num-workers", str(NUM_WORKERS),
        "--data-root", str(DATASET_ROOT),
        "--pretrained", str(best_ckpt),
        "--tag", tag,
        "--opts",
        "MODEL.PRETRAIN_CHOICE", "finetune",
        "MODEL.TRAINING_MODE", "supervised",
        "LOSS.SUPCON.ENABLE", "True",
        "LOSS.SUPCON.T", str(FT_T),
        "LOSS.SUPCON.W", str(FT_W),
        "LOSS.TRIPLETX.ENABLE", "True",
        "LOSS.CE.ENABLE", "True",
        "SOLVER.SEED", str(SEED),
        "OUTPUT_DIR", str(LOG_ROOT),
    ]
    run_and_log(cmd, "Stage 2: Joint fine-tune (CE + TripletX + SupCon)")


# ---------- Main ----------
def main():
    print("[B4] ==========================================")
    print("[B4] Two-stage pipeline: SSL --> Joint Fine-tune")
    print("[B4]  - Stage 1: SupCon-only")
    print("[B4]  - Stage 2: CE + TripletX + SupCon")
    print("[B4] Config:", CONFIG_B4)
    print("[B4] Logs:", LOG_ROOT)
    print("[B4] ==========================================")

    run_dir = run_stage1_ssl()
    if not run_dir:
        raise SystemExit("[B4] ❌ Stage 1 failed: no checkpoints found.")

    best_ckpt = find_best_checkpoint(run_dir)
    if not best_ckpt:
        raise SystemExit("[B4] ❌ No checkpoint found for Stage 2.")

    run_stage2_joint(best_ckpt)
    print("[B4] ✅ Finished! All results saved in:", LOG_ROOT)


if __name__ == "__main__":
    main()
