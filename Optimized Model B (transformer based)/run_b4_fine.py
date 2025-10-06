#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================
# File: run_b4_fine.py
# Purpose: Two-stage pipeline (B4 only)
#   Stage 1: SupCon-only SSL (SGD/LR=0.01 override)
#   Stage 2: CE + TripletX + SupCon joint fine-tune
# Notes:
#   - Uses ONLY the B4 YAML (no B3 dependency)
#   - Picks best epoch by mAP from test summaries; fallback to last
# ==========================================

import json
import subprocess
import sys
import time
from pathlib import Path

# ---------- Basic knobs ----------
EPOCHS_SSL  = 2       # Stage-1 epochs
EPOCHS_FINE = 2       # Stage-2 epochs
BATCH = 64
NUM_WORKERS = 8
SEED = 0

# SupCon hyper-params
SSL_T, SSL_W = 0.07, 0.30   # Stage-1
FT_T,  FT_W  = 0.07, 0.20   # Stage-2 joint training

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
LOG_ROOT      = ROOT / "logs"
DATASET_ROOT  = ROOT / "datasets"
PRETRAINED_IMAGENET = ROOT / "pretrained/deit_base_distilled_patch16_224-df68dfff.pth"
CONFIG_B4 = "configs/VeRi/deit_transreid_stride_b4_ssl_finetune.yml"
LOG_ROOT.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------
def run_and_log(cmd: list[str], title: str):
    print(f"\n===== [B4] {title} =====")
    print(" ".join(cmd))
    ret = subprocess.call(cmd)
    if ret != 0:
        raise SystemExit(f"[B4] ERROR: command failed with code {ret}")

def _tag(prefix: str) -> str:
    return f"{prefix}_{time.strftime('%m%d-%H%M')}"

def _run_dir_from_tag(tag: str) -> Path:
    """兼容含 seed 的保存目录"""
    p1 = LOG_ROOT / f"veri776_{tag}_seed{SEED}_deit_run"
    if p1.exists():
        return p1
    p2 = LOG_ROOT / f"veri776_{tag}_deit_run"
    if p2.exists():
        return p2
    cands = sorted(LOG_ROOT.glob(f"veri776_{tag}*_deit_run"))
    return cands[-1] if cands else p1

def _test_dir_from_tag(tag: str) -> Path:
    p1 = LOG_ROOT / f"veri776_{tag}_seed{SEED}_deit_test"
    if p1.exists():
        return p1
    p2 = LOG_ROOT / f"veri776_{tag}_deit_test"
    if p2.exists():
        return p2
    cands = sorted(LOG_ROOT.glob(f"veri776_{tag}*_deit_test"))
    return cands[-1] if cands else p1
def pick_best_ckpt_from_test(test_dir: Path, run_dir: Path) -> Path | None:
    """
    Prefer best by mAP from test summaries; fallback to log parsing or last ckpt.
    """
    best = None
    if test_dir.exists():
        for ep in sorted([p for p in test_dir.glob("epoch_*") if p.is_dir()],
                         key=lambda p: int(p.name.split("_")[-1])):
            # 优先读 summary.json
            sj = ep / "summary.json"
            log_path = ep / "log.txt"
            mAP, r1 = -1, -1
            if sj.exists():
                try:
                    obj = json.loads(sj.read_text())
                    mAP = float(obj.get("mAP", -1))
                    r1 = float(obj.get("Rank-1", obj.get("Rank1", -1)))
                except Exception:
                    pass
            elif log_path.exists():
                # 没有 summary.json 就从 log.txt 提取
                txt = log_path.read_text()
                import re
                m = re.search(r"mAP:\s*([\d\.]+)%", txt)
                n = re.search(r"Rank-1\s*:\s*([\d\.]+)%", txt)
                if m: mAP = float(m.group(1))
                if n: r1 = float(n.group(1))

            if mAP > 0:
                epoch = int(ep.name.split("_")[-1])
                if not best or mAP > best[0]:
                    best = (mAP, r1, epoch)

    if best:
        mAP, r1, ep = best
        ckpt = run_dir / f"transformer_{ep}.pth"
        print(f"[B4] ✅ Best epoch = {ep} | mAP = {mAP:.2f}% | Rank-1 = {r1:.2f}%")
        return ckpt

    # fallback
    ckpts = sorted(run_dir.glob("transformer_*.pth"))
    if ckpts:
        print(f"[B4] ⚠️ No summary found; fallback to last ckpt: {ckpts[-1].name}")
        return ckpts[-1]

    print("[B4] ❌ No checkpoints found.")
    return None

# ---------- Stage 1: SSL (SupCon-only) ----------
def stage1_ssl() -> Path:
    tag = _tag(f"b4_sslonly_T{str(SSL_T).replace('.','p')}_W{str(SSL_W).replace('.','p')}")
    cmd = [
        sys.executable, "run_modelB_deit.py",
        "--config", CONFIG_B4,
        "--epochs", str(EPOCHS_SSL),
        "--batch", str(BATCH),
        "--num-workers", str(NUM_WORKERS),
        "--data-root", str(DATASET_ROOT),
        "--pretrained", str(PRETRAINED_IMAGENET),
        "--tag", tag,
        "--opts",
        "MODEL.PRETRAIN_CHOICE", "imagenet",
        "MODEL.TRAINING_MODE", "self_supervised",
        "LOSS.SUPCON.ENABLE", "True",
        "LOSS.SUPCON.T", str(SSL_T),
        "LOSS.SUPCON.W", str(SSL_W),
        "LOSS.TRIPLETX.ENABLE", "False",
        "MODEL.METRIC_LOSS_TYPE", "none",
        "MODEL.ID_LOSS_TYPE", "none",
        "MODEL.ID_LOSS_WEIGHT", "0.0",
        "MODEL.TRIPLET_LOSS_WEIGHT", "0.0",
        "SOLVER.OPTIMIZER_NAME", "SGD",
        "SOLVER.BASE_LR", "0.01",
        "SOLVER.SEED", str(SEED),
        "OUTPUT_DIR", str(LOG_ROOT),
    ]
    run_and_log(cmd, "Stage 1: SupCon-only SSL")

    run_dir  = _run_dir_from_tag(tag)
    test_dir = _test_dir_from_tag(tag)
    if not run_dir.exists():
        raise SystemExit(f"[B4] ❌ Stage-1 run dir not found: {run_dir}")

    best_ckpt = pick_best_ckpt_from_test(test_dir, run_dir)
    if not best_ckpt:
        raise SystemExit("[B4] ❌ Stage-1 has no checkpoint to continue.")
    return best_ckpt

# ---------- Stage 2: Joint Fine-tuning ----------
def stage2_joint_finetune(ssl_ckpt: Path):
    tag = _tag(f"b4_jointft_T{str(FT_T).replace('.','p')}_W{str(FT_W).replace('.','p')}")
    cmd = [
        sys.executable, "run_modelB_deit.py",
        "--config", CONFIG_B4,
        "--epochs", str(EPOCHS_FINE),
        "--batch", str(BATCH),
        "--num-workers", str(NUM_WORKERS),
        "--data-root", str(DATASET_ROOT),
        "--pretrained", str(ssl_ckpt),
        "--tag", tag,
        "--opts",
        "MODEL.PRETRAIN_CHOICE", "finetune",
        "MODEL.PRETRAIN_PATH",  str(ssl_ckpt),
        "MODEL.TRAINING_MODE", "supervised",
        "MODEL.ID_LOSS_TYPE", "softmax",
        "MODEL.ID_LOSS_WEIGHT", "1.0",
        "MODEL.METRIC_LOSS_TYPE", "triplet",
        "MODEL.TRIPLET_LOSS_WEIGHT", "1.0",
        "LOSS.TRIPLETX.ENABLE", "True",
        "LOSS.SUPCON.ENABLE", "True",
        "LOSS.SUPCON.T", str(FT_T),
        "LOSS.SUPCON.W", str(FT_W),
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

    ssl_best = stage1_ssl()
    print(f"[B4] Using best SSL checkpoint: {ssl_best}")
    stage2_joint_finetune(ssl_best)
    print("[B4] ✅ Finished! All results saved in:", LOG_ROOT)

if __name__ == "__main__":
    main()
