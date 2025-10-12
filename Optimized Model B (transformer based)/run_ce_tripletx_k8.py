#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===========================================================
# File: run_ce_tripletx_k8.py
# Purpose: TripletX-only runner (CE + TripletX), multi-seed,
#          with auto-resume, robust re-test, and summary.
# Author: Hang Zhang (hzha0521)
# Date: 2025-10-10
# ===========================================================

from __future__ import annotations
import json, os, re, subprocess, sys, statistics as stats, argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# =============== Defaults ===============
CONFIG            = "configs/VeRi/deit_transreid_stride_b2_supcon_tripletx.yml"
FULL_EPOCHS       = 120
CHECKPOINT_PERIOD = 3
EVAL_PERIOD       = 3
SEEDS_DEFAULT     = [0]
PK_DEFAULT        = 4   # DATALOADER.NUM_INSTANCE (PK sampler K)
# =======================================

TEST_MARKER_FILES = (
    "summary.json", "test_summary.txt", "results.txt",
    "log.txt", "test_log.txt", "dist_mat.npy",
)

# --------- Colab / Log root ---------
def _in_colab() -> bool:
    try:
        import google.colab  # noqa
        return True
    except Exception:
        return False

def pick_log_root(cli_output_root: Optional[str]) -> Path:
    if cli_output_root:
        root = Path(cli_output_root)
    else:
        env = os.getenv("OUTPUT_ROOT")
        if env:
            root = Path(env)
        elif _in_colab():
            root = Path(os.getenv(
                "DRIVE_LOG_ROOT",
                "/content/drive/MyDrive/5703(hzha0521)/Optimized Model B (transformer based)/logs",
            ))
        else:
            root = Path("logs")
    root.mkdir(parents=True, exist_ok=True)
    return root

# --------- Dataset root detection ---------
def detect_data_root() -> str:
    env = os.getenv("DATASETS.ROOT_DIR") or os.getenv("DATASETS_ROOT")
    if env and (Path(env) / "VeRi").exists():
        return env
    for c in [
        "/content/5703-capstone/Optimized Model B (transformer based)/datasets",
        "/content/drive/MyDrive/5703(hzha0521)/Optimized Model B (transformer based)/datasets",
        "/content/drive/MyDrive/datasets",
        "/content/datasets",
        "/workspace/datasets",
        str(Path.cwd() / "datasets"),
    ]:
        if (Path(c) / "VeRi").exists():
            return c
    return str(Path.cwd() / "datasets")

DATA_ROOT = detect_data_root()
print(f"[TX] Using DATASETS.ROOT_DIR={DATA_ROOT}")

# --------- Helpers ---------
def _re_pick(s: str, pat: str) -> float:
    m = re.search(pat, s, re.I)
    return float(m.group(1)) if m else -1.0

def parse_metrics_from_epoch_dir(epoch_dir: Path) -> Tuple[float, float, float, float]:
    sj = epoch_dir / "summary.json"
    if sj.exists():
        try:
            o = json.loads(sj.read_text())
            return float(o.get("mAP", -1)), float(o.get("Rank-1", -1)), \
                   float(o.get("Rank-5", -1)), float(o.get("Rank-10", -1))
        except Exception:
            pass
    for name in ("test_summary.txt", "results.txt", "log.txt", "test_log.txt"):
        p = epoch_dir / name
        if p.exists():
            s = p.read_text(encoding="utf-8", errors="ignore")
            mAP = _re_pick(s, r"mAP[^0-9]*([0-9]+(?:\.[0-9]+)?)")
            r1  = _re_pick(s, r"Rank[-\s]?1[^0-9]*([0-9]+(?:\.[0-9]+)?)")
            r5  = _re_pick(s, r"Rank[-\s]?5[^0-9]*([0-9]+(?:\.[0-9]+)?)")
            r10 = _re_pick(s, r"Rank[-\s]?10[^0-9]*([0-9]+(?:\.[0-9]+)?)")
            return mAP, r1, r5, r10
    return -1.0, -1.0, -1.0, -1.0

def pick_best_epoch_metrics(test_dir: Path) -> Optional[Dict[str, Any]]:
    epochs = sorted([p for p in test_dir.glob("epoch_*") if p.is_dir()],
                    key=lambda p: int(p.name.split("_")[-1]))
    best, best_key = None, None
    for ep in epochs:
        idx = int(ep.name.split("_")[-1])
        mAP, r1, r5, r10 = parse_metrics_from_epoch_dir(ep)
        if mAP < 0:
            continue
        key = (mAP, r1)
        if (best_key is None) or (key > best_key):
            best_key = key
            best = {"epoch": idx, "mAP": mAP, "Rank-1": r1, "Rank-5": r5, "Rank-10": r10}
    return best

def safe_mean(v: List[float]) -> float:
    vals = [x for x in v if x >= 0]
    return round(stats.mean(vals), 4) if vals else -1.0

def safe_stdev(v: List[float]) -> float:
    vals = [x for x in v if x >= 0]
    return round(stats.stdev(vals), 4) if len(vals) >= 2 else 0.0

def pick_eval_device() -> str:
    env = os.getenv("EVAL_DEVICE")
    if env:
        return env
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore
            return "mps"
    except Exception:
        pass
    return "cpu"

# --------- Robust re-test ---------
def eval_missing_epochs_via_test_py(tag: str, config_path: str, log_root: Path):
    run_dir  = log_root / f"veri776_{tag}_deit_run"
    test_dir = log_root / f"veri776_{tag}_deit_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    ckpts = sorted(run_dir.glob("transformer_*.pth"),
                   key=lambda p: int(re.search(r"(\d+)", p.name).group(1)))
    device = pick_eval_device()

    for ck in ckpts:
        ep = int(re.search(r"(\d+)", ck.name).group(1))
        out_ep = test_dir / f"epoch_{ep}"
        if any((out_ep / f).exists() for f in TEST_MARKER_FILES):
            continue
        out_ep.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "test.py", "--config_file", str(config_path),
            "MODEL.DEVICE", device,
            "TEST.WEIGHT", str(ck),
            "OUTPUT_DIR", str(out_ep),
            "DATASETS.ROOT_DIR", DATA_ROOT,
        ]
        print("[TX][eval] Launch:", " ".join(cmd))
        subprocess.call(cmd)

# --------- One-seed run ---------
def ensure_full_run(seed: int, epochs: int, ckp: int, eva: int,
                    pk: int, log_root: Path, train_only: bool = False
                    ) -> Optional[Dict[str, Any]]:
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    # Naming exactly like your screenshot:
    # veri776_ce_tripletx_k{pk}_{YYYYMMDD_HHMM}_seed{seed}_deit_run/test
    tag = f"ce_tripletx_k{pk}_{ts}_seed{seed}"
    run_dir  = log_root / f"veri776_{tag}_deit_run"
    test_dir = log_root / f"veri776_{tag}_deit_test"

    trained_max = max([int(re.search(r"(\d+)", p.name).group(1)) for p in run_dir.glob("transformer_*.pth")], default=0)

    if trained_max >= epochs:
        print(f"[TX] Trained to {trained_max}, reconstruct tests (seed={seed})")
        if not train_only:
            eval_missing_epochs_via_test_py(tag, CONFIG, log_root)
    else:
        print(f"[TX] Training seed={seed} (resume {trained_max}/{epochs}), PK={pk}")
        ckpt_path = run_dir / f"transformer_{trained_max}.pth"
        cmd = [
            sys.executable, "run_modelB_deit.py",
            "--config", CONFIG,
            "--opts",
            "SOLVER.MAX_EPOCHS", str(epochs),
            "SOLVER.SEED", str(seed),
            "SOLVER.CHECKPOINT_PERIOD", str(ckp),
            "SOLVER.EVAL_PERIOD", str(eva),
            "DATALOADER.NUM_INSTANCE", str(pk),   # <-- PK=4 by default
            "LOSS.SUPCON.ENABLE", "False",
            "LOSS.TRIPLETX.ENABLE", "True",
            "MODEL.TRAINING_MODE", "supervised",
            "DATASETS.ROOT_DIR", DATA_ROOT,
            "OUTPUT_DIR", str(log_root),
            "TAG", f"ce_tripletx_k{pk}_{ts}_seed{seed}",
        ]
        if ckpt_path.exists() and trained_max > 0:
            cmd += ["MODEL.PRETRAIN_CHOICE", "resume", "MODEL.PRETRAIN_PATH", str(ckpt_path)]
        print("[TX][train] Launch:", " ".join(cmd))
        subprocess.check_call(cmd)
        if not train_only:
            eval_missing_epochs_via_test_py(tag, CONFIG, log_root)

    if train_only:
        print(f"[TX] Seed {seed} training finished (test skipped).")
        return None

    best = pick_best_epoch_metrics(test_dir)
    if best:
        print(f"[TX] Seed {seed} best epoch: {best}")
        (log_root / f"{tag}_best.json").write_text(json.dumps(best, indent=2))
    return best

# --------- CLI & Main ---------
def build_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TripletX-only runner (PK sampler configurable, robust resume/test).")
    p.add_argument("--epochs", type=int, default=FULL_EPOCHS)
    p.add_argument("--ckp", type=int, default=CHECKPOINT_PERIOD, help="Checkpoint period.")
    p.add_argument("--eval", type=int, default=EVAL_PERIOD, help="Eval period. Set large to skip during train.")
    p.add_argument("--seeds", nargs="+", default=[str(s) for s in SEEDS_DEFAULT])
    p.add_argument("--output-root", type=str, default=None, help="Override output root directory.")
    p.add_argument("--train-only", action="store_true", help="Skip test phase.")
    p.add_argument("--pk", type=int, default=PK_DEFAULT, help="PK sampler K (DATALOADER.NUM_INSTANCE).")
    return p.parse_args()

def parse_seeds(raw: List[str]) -> List[int]:
    flat: List[str] = []
    for s in raw:
        flat += [x for x in s.split(",") if x.strip()]
    return [int(x) for x in flat]

def main():
    args = build_cli()
    seeds = parse_seeds(args.seeds)
    log_root = pick_log_root(args.output_root)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    print(f"[TX] Using log_root={log_root} | timestamp={ts} | PK={args.pk}")

    seed_best: Dict[int, Dict[str, Any]] = {}
    for seed in seeds:
        rec = ensure_full_run(seed, args.epochs, args.ckp, args.eval, args.pk,
                              log_root, train_only=args.train_only)
        if rec:
            seed_best[seed] = rec

    if seed_best:
        mAPs = [r["mAP"] for r in seed_best.values()]
        R1s  = [r["Rank-1"] for r in seed_best.values()]
        summary = {
            "config": CONFIG,
            "epochs": args.epochs,
            "ckp_period": args.ckp,
            "eval_period": args.eval,
            "pk": args.pk,
            "log_root": str(log_root),
            "mean": {"mAP": safe_mean(mAPs), "Rank-1": safe_mean(R1s)},
            "std":  {"mAP": safe_stdev(mAPs), "Rank-1": safe_stdev(R1s)},
        }
        (log_root / f"veri776_ce_tripletx_k{args.pk}_{ts}_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[TX] Wrote summary → veri776_ce_tripletx_k{args.pk}_{ts}_summary.json")
    else:
        print("[TX] No valid results collected.")
    print("[TX] Done.")

if __name__ == "__main__":
    main()
