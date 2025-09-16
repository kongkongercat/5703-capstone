#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================
# File: run_b3_fine.py   (based on run_b3_pre.py + fine-tune stage)
# Purpose: B3 SupCon grid search + full retrain + fine-tuning stage
# Author: Zeyu Yang (extended with fine-tuning by ChatGPT)
# ==========================================

import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import statistics as stats

# ====== User settings ======
CONFIG = "configs/VeRi/deit_transreid_stride_b3_ssl_pretrain.yml".replace("VeRi", "VeRi")
SEARCH_EPOCHS = 12
FULL_EPOCHS   = 30
SEARCH_SEED = 0
FULL_SEEDS  = [0, 1, 2]
# ===========================

# ---------- Env helpers ----------
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
BEST_JSON = LOG_ROOT / "b3_supcon_best.json"
BEST_SUMMARY_JSON = LOG_ROOT / "b3_supcon_best_summary.json"

def _fmt(v: float) -> str:
    return str(v).replace(".", "p")

# ---------- Helpers ----------
def _to_float_list(x) -> List[float]:
    if isinstance(x, (int, float)):
        return [float(x)]
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    return []

def load_grid_from_yaml(cfg_path: str) -> Tuple[List[float], List[float]]:
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
    try:
        import yaml
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

def _latest_epoch_dir(out_test_dir: Path) -> Optional[Path]:
    epochs = sorted(
        [p for p in out_test_dir.glob("epoch_*") if p.is_dir()],
        key=lambda p: int(p.name.split("_")[-1])
    )
    return epochs[-1] if epochs else None

def _re_pick(s: str, pat: str) -> float:
    m = re.search(pat, s)
    return float(m.group(1)) if m else -1.0

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
    for name in ("test_summary.txt", "results.txt", "log.txt"):
        if ep is None: break
        p = ep / name
        if p.exists():
            s = p.read_text(encoding="utf-8", errors="ignore")
            mAP = _re_pick(s, r"mAP[^0-9]*([0-9]+(?:\.[0-9]+)?)")
            r1  = _re_pick(s, r"Rank[-\s]?1[^0-9]*([0-9]+(?:\.[0-9]+)?)")
            r5  = _re_pick(s, r"Rank[-\s]?5[^0-9]*([0-9]+(?:\.[0-9]+)?)")
            r10 = _re_pick(s, r"Rank[-\s]?10[^0-9]*([0-9]+(?:\.[0-9]+)?)")
            return mAP, r1, r5, r10
    return -1.0, -1.0, -1.0, -1.0

def safe_mean(vals: List[float]) -> float:
    return sum(vals)/len(vals) if vals else -1.0

def safe_stdev(vals: List[float]) -> float:
    return stats.pstdev(vals) if len(vals) > 1 else 0.0

# ---------- One run (search, single-seed) ----------
def run_one_search(t: float, w: float):
    tag = f"b3_supcon_T{_fmt(t)}_W{_fmt(w)}"
    subprocess.check_call([
        sys.executable, "run_modelB_deit.py",
        "--config", CONFIG,
        "--opts",
        "LOSS.SUPCON.ENABLE", "True",
        "LOSS.SUPCON.T", str(t),
        "LOSS.SUPCON.W", str(w),
        "SOLVER.MAX_EPOCHS", str(SEARCH_EPOCHS),
        "SOLVER.SEED", str(SEARCH_SEED),
        "OUTPUT_DIR", str(LOG_ROOT),
        "TAG", tag,
    ])
    out_test_dir = LOG_ROOT / f"veri776_{tag}_deit_test"
    mAP, r1, r5, r10 = parse_metrics(out_test_dir)
    print(f"[B3] SEARCH RESULT tag={tag} T={t} W={w} seed={SEARCH_SEED} "
          f"mAP={mAP} Rank-1={r1} Rank-5={r5} Rank-10={r10}")
    return {"tag": tag, "T": t, "W": w, "mAP": mAP, "R1": r1, "R5": r5, "R10": r10}

# ---------- Fine-tune helpers ----------
def pick_best_ckpt_path(run_test_dir: Path) -> Optional[Path]:
    best_ep_dir = _latest_epoch_dir(run_test_dir)
    if best_ep_dir is None:
        return None
    train_dir = Path(str(run_test_dir).replace("_deit_test", "_deit_train"))
    for name in ["model_best.pth", "model.pth", "checkpoint.pth"]:
        p = train_dir / best_ep_dir.name / name
        if p.exists():
            return p
    return None

def finetune_from_pretrain(pre_ckpt: Path, seed: int, tag_prefix: str):
    FT_CONFIG = "configs/VeRi/deit_transreid_b3_finetune.yml"
    ft_tag = f"{tag_prefix}_ft_seed{seed}"
    print(f"[B3][FT] Start finetune: seed={seed}, ckpt={pre_ckpt}")
    subprocess.check_call([
        sys.executable, "run_modelB_deit.py",
        "--config", FT_CONFIG,
        "--opts",
        "MODEL.PRETRAIN_PATH", str(pre_ckpt),
        "SOLVER.SEED", str(seed),
        "OUTPUT_DIR", str(LOG_ROOT),
        "TAG", ft_tag,
    ])
    test_dir = LOG_ROOT / f"veri776_{ft_tag}_deit_test"
    mAP, r1, r5, r10 = parse_metrics(test_dir)
    print(f"[B3][FT] RESULT tag={ft_tag} mAP={mAP} R1={r1} R5={r5} R10={r10}")
    return {"tag": ft_tag, "mAP": mAP, "R1": r1, "R5": r5, "R10": r10}

# ---------- Main ----------
def main():
    T_list, W_list = load_grid_from_yaml(CONFIG)
    print(f"[B3] Search grid → T={T_list} ; W={W_list}")
    runs = [run_one_search(t, w) for t, w in itertools.product(T_list, W_list)]
    if not runs:
        raise SystemExit("[B3] No runs executed. Check your SEARCH grid or paths.")

    best = max(runs, key=lambda x: (x["mAP"], x["R1"]))
    BEST_JSON.write_text(json.dumps(best, indent=2))
    print(f"[B3] Saved best → {BEST_JSON}")

    seed_best_records: Dict[int, Dict[str, Any]] = {}
    for seed in FULL_SEEDS:
        full_tag = f"b3_supcon_best_T{_fmt(best['T'])}_W{_fmt(best['W'])}_seed{seed}"
        print(f"[B3] Full retrain: seed={seed} → {full_tag}")
        subprocess.check_call([
            sys.executable, "run_modelB_deit.py",
            "--config", CONFIG,
            "--opts",
            "LOSS.SUPCON.ENABLE", "True",
            "LOSS.SUPCON.T", str(best["T"]),
            "LOSS.SUPCON.W", str(best["W"]),
            "SOLVER.MAX_EPOCHS", str(FULL_EPOCHS),
            "SOLVER.SEED", str(seed),
            "OUTPUT_DIR", str(LOG_ROOT),
            "TAG", full_tag,
        ])
        test_dir = LOG_ROOT / f"veri776_{full_tag}_deit_test"
        mAP, r1, r5, r10 = parse_metrics(test_dir)
        seed_best_records[seed] = {"mAP": mAP, "Rank-1": r1, "Rank-5": r5, "Rank-10": r10}

    if seed_best_records:
        summary = {
            "T": best["T"], "W": best["W"],
            "seeds": seed_best_records,
            "mean": {
                "mAP":  round(safe_mean([v["mAP"] for v in seed_best_records.values()]), 4),
                "Rank-1": round(safe_mean([v["Rank-1"] for v in seed_best_records.values()]), 4),
                "Rank-5": round(safe_mean([v["Rank-5"] for v in seed_best_records.values()]), 4),
                "Rank-10": round(safe_mean([v["Rank-10"] for v in seed_best_records.values()]), 4),
            }
        }
        BEST_SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
        print(f"[B3] Wrote summary → {BEST_SUMMARY_JSON}")

    # -------- Fine-tuning stage --------
    print(f"[B3] Begin fine-tuning from best pretrain (T={best['T']}, W={best['W']}) ...")
    ft_records = {}
    for seed in FULL_SEEDS:
        pre_tag = f"b3_supcon_best_T{_fmt(best['T'])}_W{_fmt(best['W'])}_seed{seed}"
        pre_test_dir = LOG_ROOT / f"veri776_{pre_tag}_deit_test"
        ckpt = pick_best_ckpt_path(pre_test_dir)
        if ckpt is None:
            print(f"[B3][FT][warn] No checkpoint found for seed={seed}")
            continue
        ft_rec = finetune_from_pretrain(ckpt, seed, pre_tag)
        ft_records[seed] = ft_rec

    if ft_records:
        ft_summary = {
            "finetune_from": {"T": best["T"], "W": best["W"]},
            "seeds": ft_records,
            "mean": {
                "mAP":  round(safe_mean([r["mAP"] for r in ft_records.values()]), 4),
                "Rank-1": round(safe_mean([r["R1"] for r in ft_records.values()]), 4),
                "Rank-5": round(safe_mean([r["R5"] for r in ft_records.values()]), 4),
                "Rank-10": round(safe_mean([r["R10"] for r in ft_records.values()]), 4),
            }
        }
        (LOG_ROOT / "b3_supcon_finetune_summary.json").write_text(json.dumps(ft_summary, indent=2))
        print(f"[B3][FT] Wrote summary → {LOG_ROOT / 'b3_supcon_finetune_summary.json'}")

if __name__ == "__main__":
    main()
