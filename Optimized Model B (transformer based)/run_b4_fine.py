#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model B4 Runner: SSL pretraining + CE+SupCon+TripletX fine-tuning
Author: Fanyi Meng (2025-10-06)
Based on TransReID + DeiT baseline.
"""

import itertools, json, os, re, sys, subprocess, statistics as stats
from pathlib import Path
from typing import List, Dict, Any, Optional

# ================================================================
# Config paths
# ================================================================
SSL_CONFIG = "configs/VeRi/deit_transreid_stride_b3_ssl_pretrain.yml"
FT_CONFIG  = "configs/VeRi/deit_transreid_stride_b4_ssl_finetune.yml"

SEARCH_EPOCHS = 12
FULL_EPOCHS   = 30
FULL_SEEDS    = [0, 1, 2]

# ================================================================
# Utility functions
# ================================================================
def _fmt(v: float) -> str:
    return str(v).replace(".", "p")

def _to_float_list(x):
    if isinstance(x, (float, int)): return [float(x)]
    if isinstance(x, (list, tuple)): return [float(v) for v in x]
    return []

def pick_log_root() -> Path:
    """Resolve a writable log directory."""
    env = os.getenv("OUTPUT_ROOT")
    if env: return Path(env)
    drive = Path("/content/drive/MyDrive")
    if drive.exists(): return drive / "5703(hzha0521)/Optimized Model B (transformer based)/logs"
    return Path("logs")

LOG_ROOT = pick_log_root()
LOG_ROOT.mkdir(parents=True, exist_ok=True)

# ================================================================
# YAML grid loader (T/W search)
# ================================================================
def load_grid_from_yaml(cfg_path: str):
    """Read LOSS.SUPCON.SEARCH.{T,W} lists from YAML."""
    try:
        import yaml
        y = yaml.safe_load(open(cfg_path))
        s = y.get("LOSS", {}).get("SUPCON", {}).get("SEARCH", {})
        T, W = _to_float_list(s.get("T")), _to_float_list(s.get("W"))
        if T and W: return sorted(set(T)), sorted(set(W))
    except Exception:
        pass
    return [0.07], [0.25, 0.30, 0.35]

# ================================================================
# Metric parser
# ================================================================
def _latest_epoch_dir(out_test_dir: Path) -> Optional[Path]:
    eps = sorted([p for p in out_test_dir.glob("epoch_*") if p.is_dir()],
                 key=lambda p: int(p.name.split("_")[-1]))
    return eps[-1] if eps else None

def _re_pick(s: str, pat: str) -> float:
    m = re.search(pat, s)
    return float(m.group(1)) if m else -1.0

def parse_metrics(out_dir: Path):
    """Try to read mAP / Rank-1 / 5 / 10 from summary or log."""
    ep = _latest_epoch_dir(out_dir)
    if not ep: return (-1.0,)*4
    sj = ep / "summary.json"
    if sj.exists():
        try:
            js = json.load(open(sj))
            return (
                float(js.get("mAP", -1)),
                float(js.get("Rank-1", js.get("Rank1", -1))),
                float(js.get("Rank-5", js.get("Rank5", -1))),
                float(js.get("Rank-10", js.get("Rank10", -1))),
            )
        except Exception:
            pass
    for name in ("test_summary.txt", "results.txt", "log.txt"):
        p = ep / name
        if p.exists():
            s = p.read_text(errors="ignore")
            return (
                _re_pick(s, r"mAP[^0-9]*([0-9.]+)"),
                _re_pick(s, r"Rank[-\s]?1[^0-9]*([0-9.]+)"),
                _re_pick(s, r"Rank[-\s]?5[^0-9]*([0-9.]+)"),
                _re_pick(s, r"Rank[-\s]?10[^0-9]*([0-9.]+)")
            )
    return (-1.0,)*4

def safe_mean(vs): return sum(vs)/len(vs) if vs else -1
def safe_std(vs):  return stats.pstdev(vs) if len(vs)>1 else 0.0

# ================================================================
# Stage 1 – SSL search + pretraining
# ================================================================
def run_ssl_search(T: float, W: float, seed: int):
    tag = f"b4_ssl_T{_fmt(T)}_W{_fmt(W)}_seed{seed}"
    subprocess.check_call([
        sys.executable, "run_modelB_deit.py",
        "--config", SSL_CONFIG,
        "--opts",
        "LOSS.SUPCON.ENABLE","True",
        "LOSS.SUPCON.T",str(T),
        "LOSS.SUPCON.W",str(W),
        "SOLVER.MAX_EPOCHS",str(SEARCH_EPOCHS),
        "SOLVER.SEED",str(seed),
        "OUTPUT_DIR",str(LOG_ROOT),
        "TAG",tag,
    ])
    mAP,r1,r5,r10 = parse_metrics(LOG_ROOT/f"veri776_{tag}_deit_test")
    print(f"[SSL] T={T},W={W},seed={seed} → mAP={mAP},R1={r1}")
    return dict(tag=tag,T=T,W=W,mAP=mAP,R1=r1,R5=r5,R10=r10)

# ================================================================
# Stage 2 – Fine-tuning (CE+SupCon+TripletX)
# ================================================================
def pick_best_ckpt(test_dir: Path)->Optional[Path]:
    ep = _latest_epoch_dir(test_dir)
    if not ep: return None
    run_dir = Path(str(test_dir).replace("_deit_test","_deit_run"))
    for n in ["model_best.pth","model.pth"]+ [f"transformer_{i}.pth" for i in range(200,0,-1)]:
        p = run_dir/n
        if p.exists(): return p
    return None

def run_finetune(ckpt: Path, seed:int, T:float, W:float):
    tag = f"b4_ft_T{_fmt(T)}_W{_fmt(W)}_seed{seed}"
    subprocess.check_call([
        sys.executable,"run_modelB_deit.py",
        "--config",FT_CONFIG,
        "--opts",
        "MODEL.PRETRAIN_CHOICE","finetune",
        "MODEL.PRETRAIN_PATH",str(ckpt),
        "SOLVER.SEED",str(seed),
        "LOSS.CE.ENABLE","True",
        "LOSS.SUPCON.ENABLE","True",
        "LOSS.SUPCON.W",str(W),
        "LOSS.SUPCON.T",str(T),
        "LOSS.TRIPLETX.ENABLE","True",
        "OUTPUT_DIR",str(LOG_ROOT),
        "TAG",tag,
    ])
    mAP,r1,r5,r10 = parse_metrics(LOG_ROOT/f"veri776_{tag}_deit_test")
    print(f"[B4-FT] seed={seed} → mAP={mAP},R1={r1}")
    return dict(tag=tag,mAP=mAP,R1=r1,R5=r5,R10=r10)

# ================================================================
# Main controller
# ================================================================
def main():
    LOG_ROOT.mkdir(exist_ok=True,parents=True)
    Tlist,Wlist = load_grid_from_yaml(SSL_CONFIG)
    print(f"[B4] Search grid: T={Tlist}, W={Wlist}")

    # ---------- SSL grid search ----------
    results=[]
    for T,W in itertools.product(Tlist,Wlist):
        res = run_ssl_search(T,W,seed=0)
        results.append(res)

    if not results:
        print("[Error] No SSL run executed.")
        return
    best=max(results,key=lambda x:(x["mAP"],x["R1"]))
    json.dump(best,open(LOG_ROOT/"b4_best_ssl.json","w"),indent=2)
    print(f"[B4] Best SSL combo: T={best['T']} W={best['W']} (mAP={best['mAP']})")

    # ---------- Fine-tuning ----------
    ft_records={}
    for seed in FULL_SEEDS:
        ssl_tag=f"b4_ssl_T{_fmt(best['T'])}_W{_fmt(best['W'])}_seed{0}"
        ckpt=pick_best_ckpt(LOG_ROOT/f"veri776_{ssl_tag}_deit_test")
        if not ckpt:
            print(f"[Warn] No ckpt found for seed={seed}")
            continue
        rec=run_finetune(ckpt,seed,best["T"],best["W"])
        ft_records[seed]=rec

    if ft_records:
        summary={
            "finetune_from":best,
            "seeds":ft_records,
            "mean":{
                "mAP":round(safe_mean([r["mAP"] for r in ft_records.values()]),4),
                "Rank-1":round(safe_mean([r["R1"] for r in ft_records.values()]),4)
            }
        }
        json.dump(summary,open(LOG_ROOT/"b4_finetune_summary.json","w"),indent=2)
        print(f"[B4] Summary written → {LOG_ROOT/'b4_finetune_summary.json'}")

if __name__=="__main__":
    main()
