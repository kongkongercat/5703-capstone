#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================
# File: run_b3_self_supervised.py
# Purpose: SupCon 自监督网格搜索（T/W 来自 YAML），每轮评估与保存，
#          并在训练目录额外导出 "transformer_best.pth"
# Author: Zeyu Yang (Your Email or Info Here)
# ==========================================
# Change Log
# [2025-09-16] Initial version: self-supervised SupCon grid search.
# [2025-09-22] Use softmax_triplet sampler, disable ID/Triplet, eval on, unique TAGs.
# [2025-09-23] Fix YACS dtype mismatch: MODEL.*_WEIGHT as "0.0" strings.
# [2025-09-24] (This version) 固定轮次为 30，强制每轮评估/保存；新增保存最优轮次 best pth。
#              - EVAL_PERIOD=1 & CHECKPOINT_PERIOD=1
#              - 解析每轮结果并复制 transformer_{best}.pth -> transformer_best.pth
#              - 更鲁棒的 run/test 目录解析 + metrics 解析回退
# ==========================================

import itertools
import json
import os
import re
import subprocess
import sys
import time
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import statistics as stats
import glob

# ====== User settings ======
CONFIG = "configs/VeRi/deit_transreid_stride_b3_ssl_pretrain.yml"
SEARCH_EPOCHS = 30       # 固定为 30 轮
FULL_EPOCHS   = 30       # 如需 full retrain，可保留；本脚本主要用 SEARCH
# Seeds policy
SEARCH_SEED = 0
FULL_SEEDS  = [0, 1, 2]

# 是否应用自监督样式覆盖：只用 SupCon，评估开启，从 ImageNet 初始化
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

def _unique_tag(base: str) -> str:
    """Make a unique tag to avoid resuming from old checkpoints accidentally."""
    return f"{base}_{time.strftime('%m%d-%H%M%S')}"

# ---------- YAML grid loader ----------
def _to_float_list(x) -> List[float]:
    if isinstance(x, (int, float)):
        return [float(x)]
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    return []

def load_grid_from_yaml(cfg_path: str) -> Tuple[List[float], List[float]]:
    """
    Load T/W grid from YAML (LOSS.SUPCON.SEARCH). Fallback: T=[0.07], W=[0.25,0.30,0.35]
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

# ---------- Dir resolvers ----------
def _resolve_test_dir(log_root: Path, tag: str) -> Path:
    """
    Expect: logs/veri776_{tag}_deit_test ; fallback to glob
    """
    expected = log_root / f"veri776_{tag}_deit_test"
    if expected.exists():
        return expected
    # fallback glob
    candidates = glob.glob(str(log_root / f"**/*{tag}*deit_test"), recursive=True)
    if candidates:
        candidates = sorted(candidates, key=lambda p: os.path.getmtime(p), reverse=True)
        return Path(candidates[0])
    return expected

def _resolve_run_dir(log_root: Path, tag: str) -> Path:
    """
    Expect: logs/veri776_{tag}_deit_run ; fallback to glob
    """
    expected = log_root / f"veri776_{tag}_deit_run"
    if expected.exists():
        return expected
    candidates = glob.glob(str(log_root / f"**/*{tag}*deit_run"), recursive=True)
    if candidates:
        candidates = sorted(candidates, key=lambda p: os.path.getmtime(p), reverse=True)
        return Path(candidates[0])
    return expected

# ---------- Metrics parser ----------
def _latest_epoch_dir(out_test_dir: Path) -> Optional[Path]:
    epochs = sorted(
        [p for p in out_test_dir.glob("epoch_*") if p.is_dir()],
        key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else -1,
    )
    return epochs[-1] if epochs else None

def _parse_one_epoch_metrics(ep_dir: Path) -> Optional[Dict[str, float]]:
    """
    解析单个 epoch 目录的指标（优先 summary.json；回退文本日志）。
    返回 dict: {"epoch": int, "mAP": float, "Rank-1": float, "Rank-5": float, "Rank-10": float}
    """
    epoch_idx = int(ep_dir.name.split("_")[-1]) if ep_dir.name.split("_")[-1].isdigit() else -1
    sj = ep_dir / "summary.json"
    if sj.exists():
        try:
            obj = json.loads(sj.read_text())
            return {
                "epoch": epoch_idx,
                "mAP": float(obj.get("mAP", -1)),
                "Rank-1": float(obj.get("Rank-1", obj.get("Rank1", -1))),
                "Rank-5": float(obj.get("Rank-5", obj.get("Rank5", -1))),
                "Rank-10": float(obj.get("Rank-10", obj.get("Rank10", -1))),
            }
        except Exception:
            pass

    # fallback: parse text logs in this epoch dir
    for name in ("test_summary.txt", "results.txt", "log.txt"):
        p = ep_dir / name
        if p.exists():
            s = p.read_text(encoding="utf-8", errors="ignore")
            mAP = _re_pick(s, r"mAP[^0-9]*([0-9]+(?:\.[0-9]+)?)")
            r1  = _re_pick(s, r"Rank[-\s]?1[^0-9]*([0-9]+(?:\.[0-9]+)?)")
            r5  = _re_pick(s, r"Rank[-\s]?5[^0-9]*([0-9]+(?:\.[0-9]+)?)")
            r10 = _re_pick(s, r"Rank[-\s]?10[^0-9]*([0-9]+(?:\.[0-9]+)?)")
            return {"epoch": epoch_idx, "mAP": mAP, "Rank-1": r1, "Rank-5": r5, "Rank-10": r10}
    return None

def pick_best_epoch_metrics(test_dir: Path) -> Optional[Dict[str, float]]:
    """
    遍历 test_dir 下所有 epoch_*，按 (mAP, Rank-1) 选最优，返回该 epoch 的记录。
    """
    best = None
    for ep in sorted([p for p in test_dir.glob("epoch_*") if p.is_dir()],
                     key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else -1):
        rec = _parse_one_epoch_metrics(ep)
        if not rec or rec["mAP"] < 0:
            continue
        if (best is None) or ((rec["mAP"], rec["Rank-1"]) > (best["mAP"], best["Rank-1"])):
            best = rec
    return best

def parse_metrics(out_test_dir: Path) -> Tuple[float, float, float, float]:
    """
    读取“最新”epoch 的指标（用于快速打印）；若缺失则返回 -1
    """
    ep = _latest_epoch_dir(out_test_dir)
    if not ep:
        return -1.0, -1.0, -1.0, -1.0
    rec = _parse_one_epoch_metrics(ep)
    if not rec:
        return -1.0, -1.0, -1.0, -1.0
    return rec["mAP"], rec["Rank-1"], rec["Rank-5"], rec["Rank-10"]

# ---------- opts builder ----------
def _ssl_overrides_for_opts(t: float, w: float) -> List[str]:
    """
    自监督 SupCon 推荐覆盖：
      - 采样器 softmax_triplet，但将 ID/TRIPLET 权重置 0
      - 强制每轮评估与保存：EVAL_PERIOD=1, CHECKPOINT_PERIOD=1
      - 从 ImageNet 初始化，避免误恢复旧权重
    """
    if not FORCE_SSL_OVERRIDES:
        return [
            "LOSS.SUPCON.ENABLE", "True",
            "LOSS.SUPCON.T", str(t),
            "LOSS.SUPCON.W", str(w),
        ]
    return [
        # SupCon 开启 + 超参
        "LOSS.SUPCON.ENABLE", "True",
        "LOSS.SUPCON.T", str(t),
        "LOSS.SUPCON.W", str(w),
        "LOSS.SUPCON.POS_RULE", "class",

        # 监督采样器，但监督损失权重为 0（纯 SupCon 训练）
        "DATALOADER.SAMPLER", "softmax_triplet",
        "DATALOADER.NUM_INSTANCE", "4",

        # 注意：以字符串形式传给 YACS，避免 dtype mismatch
        "MODEL.ID_LOSS_WEIGHT", "0.0",
        "MODEL.TRIPLET_LOSS_WEIGHT", "0.0",

        # 训练模式与初始化
        "MODEL.TRAINING_MODE", "self_supervised",
        "MODEL.PRETRAIN_CHOICE", "imagenet",

        # 评估 & 保存频率：每轮一次
        "TEST.EVAL", "True",
        "SOLVER.EVAL_PERIOD", "1",
        "SOLVER.CHECKPOINT_PERIOD", "1",
    ]

# ---------- One run ----------
def _copy_best_weight_if_available(tag: str, log_root: Path, best_epoch: int) -> Optional[Path]:
    """
    在训练目录下把 transformer_{best_epoch}.pth 复制为 transformer_best.pth
    """
    run_dir = _resolve_run_dir(log_root, tag)
    src = run_dir / f"transformer_{best_epoch}.pth"
    dst = run_dir / "transformer_best.pth"
    if src.exists():
        try:
            shutil.copyfile(src, dst)
            print(f"[B3] Best checkpoint exported → {dst}")
            return dst
        except Exception as e:
            print(f"[B3][warn] Failed to copy best weight: {e}")
    else:
        print(f"[B3][warn] Best weight not found: {src}")
    return None

def run_one_search(t: float, w: float):
    tag = _unique_tag(f"b3_supcon_T{_fmt(t)}_W{_fmt(w)}")
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

    # 解析结果目录
    out_test_dir = _resolve_test_dir(LOG_ROOT, tag)
    if not out_test_dir.exists():
        raise RuntimeError(f"[B3] Test dir not found: {out_test_dir}")

    # 打印最新一轮指标（便于快速查看）
    mAP, r1, r5, r10 = parse_metrics(out_test_dir)
    print(f"[B3] LAST EPOCH RESULT tag={tag} T={t} W={w} seed={SEARCH_SEED} "
          f"mAP={mAP} Rank-1={r1} Rank-5={r5} Rank-10={r10}")

    # 选取最优 epoch（mAP 优先，Rank-1 次之），导出 best pth
    best_ep = pick_best_epoch_metrics(out_test_dir)
    if best_ep is None:
        raise RuntimeError(f"[B3] No valid epoch metrics found under: {out_test_dir}")

    best_weight_path = _copy_best_weight_if_available(tag, LOG_ROOT, best_ep["epoch"])

    print(f"[B3] BEST EPOCH tag={tag}: epoch={best_ep['epoch']}  "
          f"mAP={best_ep['mAP']}  R1={best_ep['Rank-1']}  "
          f"(best weight: {best_weight_path if best_weight_path else 'N/A'})")

    return {
        "tag": tag, "T": t, "W": w,
        "mAP": best_ep["mAP"], "R1": best_ep["Rank-1"],
        "R5": best_ep.get("Rank-5", -1), "R10": best_ep.get("Rank-10", -1),
        "best_epoch": best_ep["epoch"],
        "best_weight": str(best_weight_path) if best_weight_path else "",
    }

# ---------- Main ----------
def main():
    T_list, W_list = load_grid_from_yaml(CONFIG)
    print(f"[B3] Search grid → T={T_list} ; W={W_list}")
    print(f"[B3] Epochs per run        : {SEARCH_EPOCHS} (eval/save every epoch)")
    print(f"[B3] Seed (search)         : {SEARCH_SEED}")
    print(f"[B3] FORCE_SSL_OVERRIDES   : {FORCE_SSL_OVERRIDES}")

    runs = [run_one_search(t, w) for t, w in itertools.product(T_list, W_list)]
    if not runs:
        raise SystemExit("[B3] No runs executed. Check your SEARCH grid or paths.")

    # 选择整体最佳 (mAP, R1)
    best = max(runs, key=lambda x: (x["mAP"], x["R1"]))
    BEST_JSON.write_text(json.dumps({
        "T": best["T"], "W": best["W"],
        "mAP": best["mAP"], "Rank-1": best["R1"],
        "Rank-5": best.get("R5", -1), "Rank-10": best.get("R10", -1),
        "best_epoch": best.get("best_epoch", -1),
        "best_weight": best.get("best_weight", ""),
        "source_tag": best["tag"],
        "note": f"Best SupCon params from B3 search (single-seed screening). "
                f"EVAL_PERIOD=1, CHECKPOINT_PERIOD=1, EPOCHS={SEARCH_EPOCHS}",
    }, indent=2))
    print(f"[B3] Saved best summary → {BEST_JSON}")

    # 如需 full retrain，可在这里保留多 seed 逻辑（略）
    print(f"[B3] All runs finished under {LOG_ROOT}")

if __name__ == "__main__":
    main()

