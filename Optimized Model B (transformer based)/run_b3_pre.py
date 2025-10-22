#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: run_b3_self_supervised.py
Purpose: B3 SupCon grid search (self-supervised pretraining) with training-only runs.
Author: Zeyu Yang
"""
from __future__ import annotations

import itertools
import json
import os
import re
import subprocess
import sys
import time
import glob
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import statistics as stats
from collections import defaultdict

# ====== Configurable user settings ======
CONFIG = "configs/VeRi/deit_transreid_stride_b3_ssl_pretrain.yml"
SEARCH_EPOCHS = 30        # run 30 epochs for each (T,W) during search
FULL_EPOCHS = 30          # if you want to do a full retrain later
SEARCH_SEED = 0           # single-seed screening
FULL_SEEDS = [0, 1, 2]    # optional full retrain seeds

FORCE_SSL_OVERRIDES = True  # force SSL-style overrides (SupCon-only training)
# ========================================

def _in_colab() -> bool:
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return False

def pick_log_root() -> Path:
    """Choose log root. Prefer environment, then Colab Drive path, else local logs/."""
    env = os.getenv("OUTPUT_ROOT")
    if env:
        return Path(env)
    if _in_colab():
        try:
            from google.colab import drive  # type: ignore
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

# ---------------- YAML grid loader ----------------
def _to_float_list(x) -> List[float]:
    if isinstance(x, (int, float)):
        return [float(x)]
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    return []

def load_grid_from_yaml(cfg_path: str) -> Tuple[List[float], List[float]]:
    """
    Load T/W grid from YAML. Try YACS then PyYAML. Fallback to sensible defaults.
    """
    # Try YACS
    try:
        from yacs.config import CfgNode as CN  # type: ignore
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

    # fallback
    return [0.07], [0.25, 0.30, 0.35]

# ---------------- regex + stats helpers ----------------
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

# ---------------- train/run helpers ----------------
def _resolve_test_dir(log_root: Path, tag: str) -> Path:
    """
    Resolve test directory for a given tag. Fallback to glob search.
    Standard pattern used by run_modelB_deit.py: veri776_{tag}_deit_test
    """
    expected = log_root / f"veri776_{tag}_deit_test"
    if expected.exists():
        return expected
    # fallback glob search
    candidates = glob.glob(str(log_root / f"**/*{tag}*deit_test"), recursive=True)
    if candidates:
        candidates = sorted(candidates, key=lambda p: os.path.getmtime(p), reverse=True)
        return Path(candidates[0])
    return expected

def _resolve_run_dir(log_root: Path, tag: str) -> Path:
    """
    Resolve run directory for a given tag. Expect veri776_{tag}_deit_run by default.
    """
    expected = log_root / f"veri776_{tag}_deit_run"
    if expected.exists():
        return expected
    candidates = glob.glob(str(log_root / f"**/*{tag}*deit_run"), recursive=True)
    if candidates:
        candidates = sorted(candidates, key=lambda p: os.path.getmtime(p), reverse=True)
        return Path(candidates[0])
    return expected

# ---------------- SSL overrides (training-only) ----------------
def _ssl_overrides_for_opts(t: float, w: float) -> List[str]:
    """
    Recommended overrides for SSL-style SupCon training:
      - Use softmax_triplet sampler but set ID/TRIPLET weights to 0
      - Disable internal testing during training (TEST.EVAL=False)
      - Start from ImageNet pretrain to avoid accidental resume
    """
    if not FORCE_SSL_OVERRIDES:
        return [
            "LOSS.SUPCON.ENABLE", "True",
            "LOSS.SUPCON.T", str(t),
            "LOSS.SUPCON.W", str(w),
        ]
    return [
        "LOSS.SUPCON.ENABLE", "True",
        "LOSS.SUPCON.T", str(t),
        "LOSS.SUPCON.W", str(w),
        "LOSS.SUPCON.POS_RULE", "class",

        "DATALOADER.SAMPLER", "softmax_triplet",
        "DATALOADER.NUM_INSTANCE", "4",

        # Pass weights as strings to avoid YACS dtype mismatch
        "MODEL.ID_LOSS_WEIGHT", "0.0",
        "MODEL.TRIPLET_LOSS_WEIGHT", "0.0",

        "MODEL.TRAINING_MODE", "self_supervised",
        "MODEL.PRETRAIN_CHOICE", "imagenet",

        # Turn off internal validation in training; we will evaluate on train set afterwards
        "TEST.EVAL", "False",
    ]

# ------------------ Evaluate training checkpoints on training set ------------------
def _evaluate_checkpoints_on_train(cfg_path: str, run_dir: Path, output_dir: Path):
    """
    After training is finished, evaluate all transformer_*.pth in run_dir on the training set.
    Build a deterministic 1-query-per-PID split (first image per pid as query).
    Save best checkpoint by training-set mAP as transformer_best_train_map.pth under run_dir.
    Also copy best to output_dir for convenience.
    """
    # lazy imports that depend on repo layout
    import torch
    from config import cfg as global_cfg  # repo's global cfg object
    from datasets import make_dataloader
    from datasets.bases import ImageDataset
    from model import make_model
    from utils.metrics import R1_mAP_eval

    print(f"[eval-train] Loading cfg from: {cfg_path}")
    global_cfg.defrost()
    global_cfg.merge_from_file(cfg_path)
    global_cfg.freeze()

    # build dataloaders to extract dataset/train list and transforms
    train_loader, train_loader_normal, val_loader, num_query_cfg, num_classes, camera_num, view_num = make_dataloader(global_cfg)

    # attempt to extract raw train items list
    raw_train_items = None
    try:
        raw_train_items = train_loader.dataset.dataset.data
    except Exception:
        try:
            raw_train_items = train_loader.dataset.data
        except Exception:
            raise RuntimeError("Cannot extract training item list from dataloader; eval-on-train requires access to dataset.train list.")

    # group by pid
    by_pid = defaultdict(list)
    for it in raw_train_items:
        img_path, pid, camid, viewid = it[:4]
        by_pid[int(pid)].append((img_path, int(pid), int(camid), int(viewid)))

    query, gallery = [], []
    for pid, items in by_pid.items():
        items = list(items)
        query.append(items[0])  # deterministic: pick first as query
        gallery.extend(items)

    eval_list = query + gallery
    num_query_train = len(query)
    print(f"[eval-train] Query={len(query)} Gallery={len(gallery)} Total={len(eval_list)}")

    # get non-aug transform
    val_tf = train_loader_normal.dataset.transform

    # build eval dataset + loader
    eval_set = ImageDataset(eval_list, val_tf)

    def val_collate_fn(batch):
        imgs, pids, camids, viewids, img_paths = zip(*batch)
        viewids = torch.tensor(viewids, dtype=torch.int64)
        camids_batch = torch.tensor(camids, dtype=torch.int64)
        return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths

    eval_loader = torch.utils.data.DataLoader(
        eval_set, batch_size=global_cfg.TEST.IMS_PER_BATCH, shuffle=False,
        num_workers=global_cfg.DATALOADER.NUM_WORKERS, collate_fn=val_collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(global_cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num).to(device)
    evaluator = R1_mAP_eval(num_query_train, max_rank=50, feat_norm=global_cfg.TEST.FEAT_NORM)

    # collect checkpoints
    ckpts = sorted([p for p in run_dir.glob("transformer_*.pth")], key=lambda p: int(re.search(r"transformer_(\d+)\.pth", p.name).group(1)) if re.search(r"transformer_(\d+)\.pth", p.name) else -1)
    if not ckpts:
        print(f"[eval-train] No checkpoints found in {run_dir}")
        return None

    best = {"mAP": -1.0, "r1": -1.0, "epoch": -1, "path": None}
    print(f"[eval-train] Evaluating {len(ckpts)} checkpoints (this may take time)...")

    for p in ckpts:
        try:
            sd = torch.load(p, map_location="cpu")
            model.load_state_dict(sd, strict=False)
            model.eval()
            evaluator.reset()
            with torch.no_grad():
                for img, pids, camids, camids_batch, viewids, _ in eval_loader:
                    img = img.to(device, non_blocking=True)
                    camids_batch = camids_batch.to(device, non_blocking=True)
                    viewids = viewids.to(device, non_blocking=True)
                    feat = model(img, cam_label=camids_batch, view_label=viewids)
                    evaluator.update((feat, pids, camids))
            cmc, mAP, *_ = evaluator.compute()
            r1 = float(cmc[0]); mAP = float(mAP)
            epoch = int(re.search(r"transformer_(\d+)\.pth", p.name).group(1)) if re.search(r"transformer_(\d+)\.pth", p.name) else -1
            print(f"[eval-train] epoch={epoch:3d}  mAP={mAP*100:6.2f}%  Rank-1={r1*100:6.2f}%  ({p.name})")
            if mAP > best["mAP"]:
                best.update({"mAP": mAP, "r1": r1, "epoch": epoch, "path": p})
                # save best into run_dir
                tmp = run_dir / "transformer_best_train_map.tmp.pth"
                dst = run_dir / "transformer_best_train_map.pth"
                torch.save(torch.load(p, map_location="cpu"), tmp)
                os.replace(tmp, dst)
                (run_dir / "best_train_map.txt").write_text(f"best_epoch={epoch}\ntrain_mAP={mAP}\ntrain_rank1={r1}\nsource={p.name}\n")
                # copy to global output_dir for convenience (best per run)
                try:
                    dst2 = output_dir / f"{run_dir.name}_transformer_best_train_map.pth"
                    tmp2 = output_dir / f"{run_dir.name}_transformer_best_train_map.tmp.pth"
                    torch.save(torch.load(p, map_location="cpu"), tmp2)
                    os.replace(tmp2, dst2)
                except Exception:
                    pass
        except Exception as e:
            print(f"[eval-train][warn] failed on {p.name}: {e}")

    print(f"[eval-train] Done. Best epoch={best['epoch']}, mAP={best['mAP']:.6f}, r1={best['r1']:.6f}")
    return best

# ------------------ Main run_one_search (train-only + eval-on-train) ------------------
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
        "SOLVER.CHECKPOINT_PERIOD", "1",
    ]
    print(f"[B3] Running training (no internal test): {' '.join(cmd)}")
    subprocess.check_call(cmd)

    # resolve run directory (where checkpoints written)
    run_dir = _resolve_run_dir(LOG_ROOT, tag)
    if not run_dir.exists():
        # fallback
        alt = LOG_ROOT / f"veri776_{tag}_deit_run"
        if alt.exists():
            run_dir = alt

    if not run_dir.exists():
        raise RuntimeError(f"[B3] Cannot find run directory for tag {tag} under {LOG_ROOT}")

    # evaluate all checkpoints on training set and pick best
    best = _evaluate_checkpoints_on_train(CONFIG, run_dir, LOG_ROOT)
    if best is None:
        raise RuntimeError(f"[B3] No valid evaluation results for run {run_dir}")

    rec = {
        "tag": tag,
        "T": t, "W": w,
        "mAP": best["mAP"], "R1": best["r1"],
        "best_epoch": best["epoch"],
        "best_weight": str(best["path"]),
        "run_dir": str(run_dir),
    }
    print(f"[B3] Completed search run: tag={tag}, best_epoch={best['epoch']}, mAP={best['mAP']}, R1={best['r1']}")
    return rec

# ------------------ Main entry ------------------
def main():
    T_list, W_list = load_grid_from_yaml(CONFIG)
    print(f"[B3] Search grid → T={T_list} ; W={W_list}")
    print(f"[B3] Epochs per run: {SEARCH_EPOCHS}")
    print(f"[B3] Single-seed: {SEARCH_SEED}, full seeds: {FULL_SEEDS}")
    print(f"[B3] LOG_ROOT: {LOG_ROOT}")

    runs = []
    for t, w in itertools.product(T_list, W_list):
        try:
            rec = run_one_search(t, w)
            runs.append(rec)
        except Exception as e:
            print(f"[B3][warn] Run (T={t},W={w}) failed: {e}")

    if not runs:
        raise SystemExit("[B3] No successful runs executed.")

    best = max(runs, key=lambda x: (x["mAP"], x["R1"]))
    BEST_JSON.write_text(json.dumps({
        "T": best["T"], "W": best["W"],
        "mAP": best["mAP"], "Rank-1": best["R1"],
        "best_epoch": best.get("best_epoch", -1),
        "best_weight": best.get("best_weight", ""),
        "source_tag": best["tag"],
    }, indent=2))
    print(f"[B3] Saved best summary → {BEST_JSON}")
    print("[B3] All runs finished.")

if __name__ == "__main__":
    main()
