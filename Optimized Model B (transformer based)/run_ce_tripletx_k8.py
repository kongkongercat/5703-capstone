#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===========================================================
# File: run_ce_tripletx.py
# Purpose: TripletX-only (CrossEntropy + TripletX) training/eval launcher
#          with unified naming, multi-seed support, auto-resume, and
#          robust re-test of missing epochs.
# Author: Hang Zhang (hzha0521)
# ===========================================================
# Change Log
# [2025-10-10 | Hang Zhang] Initial version (TripletX-only runner, 120e default, PK configurable).
# [2025-10-12 | Hang Zhang] Add --test-only / --from-run / --tag / --cfg flows; fixed-tag resume.
# [2025-10-18 | Hang Zhang] Unify style with run_ce_triplet/run_b0/run_b1:
#                           - Section layout (Defaults/Env&Paths/CLI/Helpers/Launcher/Main)
#                           - Consistent prints and tag naming with DS prefix
#                           - Explicit OUTPUT_ROOT/DATA_ROOT handling
# [2025-10-18 | Hang Zhang] Default PK updated to 8 (DATALOADER.NUM_INSTANCE=8).
# ===========================================================

import argparse
import json
import os
import re
import statistics as stats
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ================= Defaults =================
DEFAULT_CONFIG      = "configs/VeRi/deit_transreid_stride_b2_supcon_tripletx.yml"
DEFAULT_EPOCHS      = 120
DEFAULT_SAVE_EVERY  = 3         # also used as eval period by default
DEFAULT_SEEDS       = "0"
DEFAULT_DSPREFIX    = "veri776"
DEFAULT_PK          = 8         # <-- CHANGED: default K for PK sampler

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

def detect_data_root() -> str:
    env = os.getenv("DATASETS_ROOT") or os.getenv("DATASETS.ROOT_DIR")
    if env and (Path(env) / "VeRi").exists():
        return env
    candidates = [
        "/content/5703-capstone/Optimized Model B (transformer based)/datasets",
        "/content/drive/MyDrive/5703(hzha0521)/Optimized Model B (transformer based)/datasets",
        "/content/drive/MyDrive/datasets",
        "/content/datasets",
        "/workspace/datasets",
        str(Path.cwd() / "datasets"),
    ]
    for c in candidates:
        if (Path(c) / "VeRi").exists():
            return c
    return str(Path.cwd() / "datasets")

LOG_ROOT = pick_log_root()
LOG_ROOT.mkdir(parents=True, exist_ok=True)
DATA_ROOT = detect_data_root()

def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")

# ================= CLI =================
def build_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CE + TripletX with unified naming (DS prefix, PK, seeds).")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    p.add_argument("--seeds", type=str, default=DEFAULT_SEEDS, help="comma/space separated, e.g. '0,1,2'")
    p.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    p.add_argument("--output-root", type=str, default=None)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--ds-prefix", type=str, default=DEFAULT_DSPREFIX)
    p.add_argument("--pk", type=int, default=DEFAULT_PK, help="PK sampler K (DATALOADER.NUM_INSTANCE)")
    p.add_argument("--opts", nargs=argparse.REMAINDER, help="extra YACS options passed to run_modelB_deit.py")
    # evaluation-only utilities (style aligned with previous script family)
    p.add_argument("--test-only", action="store_true", help="Only run evaluation for an existing TAG.")
    p.add_argument("--from-run", type=str, default=None, help="Path to existing *_deit_run/_deit_test, or a TAG.")
    p.add_argument("--tag", type=str, default=None, help="Direct TAG (e.g., ce_tripletx_k8_20251010_1507_seed0).")
    p.add_argument("--cfg", type=str, default=None, help="Alias of --config for compatibility.")
    return p.parse_args()

# ================= Helpers =================
def _max_epoch_from_checkpoints(run_dir: Path) -> int:
    if not run_dir.exists():
        return 0
    max_ep = 0
    for p in run_dir.glob("transformer_*.pth"):
        m = re.search(r"transformer_(\d+)\.pth", p.name)
        if m:
            max_ep = max(max_ep, int(m.group(1)))
    return max_ep

def _infer_tag_from_input(s: str) -> str:
    """
    Accept:
      - full path to <ds>_<TAG>_deit_run or _deit_test
      - pure TAG ('ce_tripletx_k8_20251010_1507_seed0')
    Return the <TAG> part only.
    """
    p = Path(s)
    name = p.name
    m = re.match(r".+?_(.+)_(?:deit_run|deit_test)$", name)
    if m:
        return m.group(1)
    m = re.search(r".+?_(.+?)_(?:deit_run|deit_test)", str(p))
    if m:
        return m.group(1)
    return name

def _safe_mean(v: List[float]) -> float:
    vals = [x for x in v if x >= 0]
    return round(stats.mean(vals), 4) if vals else -1.0

def _safe_stdev(v: List[float]) -> float:
    vals = [x for x in v if x >= 0]
    return round(stats.stdev(vals), 4) if len(vals) >= 2 else 0.0

def _parse_metrics_from_epoch_dir(epoch_dir: Path) -> Tuple[float, float, float, float]:
    for name in ("summary.json", "test_summary.txt", "results.txt", "log.txt", "test_log.txt"):
        p = epoch_dir / name
        if not p.exists():
            continue
        s = p.read_text(encoding="utf-8", errors="ignore")
        if name == "summary.json":
            try:
                o = json.loads(s)
                return float(o.get("mAP", -1)), float(o.get("Rank-1", -1)), \
                       float(o.get("Rank-5", -1)), float(o.get("Rank-10", -1))
            except Exception:
                pass
        # fallback regex parse
        mAP = _pick_float(s, r"mAP[^0-9]*([0-9]+(?:\.[0-9]+)?)")
        r1  = _pick_float(s, r"Rank[-\s]?1[^0-9]*([0-9]+(?:\.[0-9]+)?)")
        r5  = _pick_float(s, r"Rank[-\s]?5[^0-9]*([0-9]+(?:\.[0-9]+)?)")
        r10 = _pick_float(s, r"Rank[-\s]?10[^0-9]*([0-9]+(?:\.[0-9]+)?)")
        if mAP >= 0:
            return mAP, r1, r5, r10
    return -1.0, -1.0, -1.0, -1.0

def _pick_float(s: str, pat: str) -> float:
    m = re.search(pat, s, re.I)
    return float(m.group(1)) if m else -1.0

def _int_from_name(p: Path) -> int:
    m = re.search(r"(\d+)", p.name)
    return int(m.group(1)) if m else 0

def _pick_eval_device() -> str:
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

def _eval_missing_epochs(tag: str, cfg_path: str, ds_prefix: str, log_root: Path):
    run_dir  = log_root / f"{ds_prefix}_{tag}_deit_run"
    test_dir = log_root / f"{ds_prefix}_{tag}_deit_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    ckpts = sorted(run_dir.glob("transformer_*.pth"), key=_int_from_name)
    device = _pick_eval_device()

    for ck in ckpts:
        ep = _int_from_name(ck)
        out_ep = test_dir / f"epoch_{ep}"
        if any((out_ep / name).exists() for name in ("summary.json", "test_summary.txt", "results.txt", "log.txt", "test_log.txt", "dist_mat.npy")):
            continue
        out_ep.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "test.py", "--config_file", str(cfg_path),
            "MODEL.DEVICE", device,
            "TEST.WEIGHT", str(ck),
            "OUTPUT_DIR", str(out_ep),
            "DATASETS.ROOT_DIR", DATA_ROOT,
        ]
        print("[TripletX][eval] ", " ".join(cmd))
        subprocess.call(cmd)

def _pick_best_epoch_metrics(test_dir: Path) -> Optional[Dict[str, Any]]:
    epochs = sorted([p for p in test_dir.glob("epoch_*") if p.is_dir()],
                    key=lambda p: int(p.name.split("_")[-1]))
    best, best_key = None, None
    for ep in epochs:
        idx = int(ep.name.split("_")[-1])
        mAP, r1, r5, r10 = _parse_metrics_from_epoch_dir(ep)
        if mAP < 0:
            continue
        key = (mAP, r1)
        if (best_key is None) or (key > best_key):
            best_key = key
            best = {"epoch": idx, "mAP": mAP, "Rank-1": r1, "Rank-5": r5, "Rank-10": r10}
    return best

# ================= Train launcher =================
def _launch_training(config_path: str,
                     seed: int,
                     target_epochs: int,
                     log_root: Path,
                     ds_prefix: str,
                     data_root: str,
                     ckpt_period: int,
                     eval_period: int,
                     pk: int,
                     extra_opts,
                     fixed_tag: Optional[str] = None) -> str:
    """
    Train (resume if needed). If fixed_tag is provided, reuse existing run/test dirs.
    """
    ts = _now_stamp()
    tag = fixed_tag if fixed_tag else f"ce_tripletx_k{pk}_{ts}_seed{seed}"
    run_dir  = log_root / f"{ds_prefix}_{tag}_deit_run"
    test_dir = log_root / f"{ds_prefix}_{tag}_deit_test"
    run_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    trained_max = _max_epoch_from_checkpoints(run_dir)

    # YACS opts
    opts = [
        "SOLVER.MAX_EPOCHS", str(target_epochs),
        "SOLVER.SEED", str(seed),
        "SOLVER.CHECKPOINT_PERIOD", str(ckpt_period),
        "SOLVER.EVAL_PERIOD", str(eval_period),  # set big number to skip during train if needed
        "DATALOADER.NUM_INSTANCE", str(pk),
        "LOSS.SUPCON.ENABLE", "False",
        "LOSS.TRIPLETX.ENABLE", "True",
        "MODEL.TRAINING_MODE", "supervised",
        "DATASETS.ROOT_DIR", data_root,
        "OUTPUT_DIR", str(log_root),
        "TAG", tag,
    ]
    if extra_opts:
        opts += extra_opts

    # resume
    if trained_max > 0:
        last_ckpt = run_dir / f"transformer_{trained_max}.pth"
        if last_ckpt.exists():
            print(f"[TripletX][resume] {last_ckpt} (seed={seed})")
            opts += [
                "MODEL.PRETRAIN_CHOICE", "resume",
                "MODEL.PRETRAIN_PATH", str(last_ckpt),
            ]

    cmd = [sys.executable, "run_modelB_deit.py", "--config", config_path, "--opts"] + opts
    print(f"[TripletX] Launch tag={tag} | PK={pk} | seed={seed}")
    print("[TripletX] CMD:", " ".join(cmd))
    subprocess.check_call(cmd)
    return tag

# ================= Main =================
def main():
    args = build_cli()

    # config alias compatibility
    if args.cfg and not args.config:
        args.config = args.cfg

    global LOG_ROOT, DATA_ROOT
    if args.output_root:
        LOG_ROOT = Path(args.output_root); LOG_ROOT.mkdir(parents=True, exist_ok=True)
    if args.data_root:
        DATA_ROOT = args.data_root

    seed_list = [int(s) for s in re.split(r"[,\s]+", args.seeds) if s.strip()]
    if not seed_list:
        raise SystemExit("[TripletX] No valid seeds given.")

    print(f"[TripletX-CLI] CONFIG={args.config}")
    print(f"[TripletX-CLI] EPOCHS={args.epochs}")
    print(f"[TripletX-CLI] SEEDS={seed_list}")
    print(f"[TripletX-CLI] OUTPUT_ROOT={LOG_ROOT}")
    print(f"[TripletX-CLI] DATA_ROOT={DATA_ROOT}")
    print(f"[TripletX-CLI] DS_PREFIX={args.ds_prefix}")
    print(f"[TripletX-CLI] PK={args.pk}")

    # fixed TAG from --from-run / --tag
    fixed_tag: Optional[str] = None
    if args.from_run:
        fixed_tag = _infer_tag_from_input(args.from_run)
        print(f"[TripletX-CLI] Fixed TAG from --from-run → {fixed_tag}")
    elif args.tag:
        fixed_tag = args.tag
        print(f"[TripletX-CLI] Fixed TAG from --tag → {fixed_tag}")

    # test-only flow
    if args.test_only:
        if not fixed_tag:
            raise ValueError("--test-only requires --from-run or --tag to locate checkpoints.")
        _eval_missing_epochs(fixed_tag, args.config, args.ds_prefix, LOG_ROOT)
        test_dir = LOG_ROOT / f"{args.ds_prefix}_{fixed_tag}_deit_test"
        best = _pick_best_epoch_metrics(test_dir)
        if best:
            print(f"[TripletX][test-only] Best epoch: {best}")
        else:
            print("[TripletX][test-only] No valid metrics found.")
        print("[TripletX] Done (test-only).")
        return

    # normal flow (train + re-test)
    extra_opts = (args.opts or [])
    best_records: Dict[int, Dict[str, Any]] = {}

    for seed in seed_list:
        tag = _launch_training(
            config_path=args.config,
            seed=seed,
            target_epochs=args.epochs,
            log_root=LOG_ROOT,
            ds_prefix=args.ds_prefix,
            data_root=DATA_ROOT,
            ckpt_period=args.save_every,
            eval_period=args.save_every,
            pk=args.pk,
            extra_opts=extra_opts,
            fixed_tag=fixed_tag
        )
        # robust re-test after training
        _eval_missing_epochs(tag if not fixed_tag else fixed_tag, args.config, args.ds_prefix, LOG_ROOT)
        test_dir = LOG_ROOT / f"{args.ds_prefix}_{tag if not fixed_tag else fixed_tag}_deit_test"
        best = _pick_best_epoch_metrics(test_dir)
        if best:
            best_records[seed] = best
            (LOG_ROOT / f"{(tag if not fixed_tag else fixed_tag)}_best.json").write_text(json.dumps(best, indent=2))
            print(f."[TripletX] Seed {seed} best: {best}")

    # summary across seeds
    if best_records:
        mAPs = [rec["mAP"] for rec in best_records.values()]
        R1s  = [rec["Rank-1"] for rec in best_records.values()]
        ts = _now_stamp()
        summary = {
            "config": args.config,
            "epochs": args.epochs,
            "save_every": args.save_every,
            "pk": args.pk,
            "log_root": str(LOG_ROOT),
            "mean": {"mAP": _safe_mean(mAPs), "Rank-1": _safe_mean(R1s)},
            "std":  {"mAP": _safe_stdev(mAPs), "Rank-1": _safe_stdev(R1s)},
            "seeds": seed_list,
        }
        (LOG_ROOT / f"{args.ds_prefix}_ce_tripletx_k{args.pk}_{ts}_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[TripletX] Wrote summary → {args.ds_prefix}_ce_tripletx_k{args.pk}_{ts}_summary.json")
    else:
        print("[TripletX] No valid results collected.")

    print("[TripletX] Done.")

if __name__ == "__main__":
    main()
