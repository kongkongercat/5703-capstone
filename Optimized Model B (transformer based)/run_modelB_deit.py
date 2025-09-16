#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Model B Launcher (Based on TransReID, DeiT backbone, VeRi-776)

This launcher runs experiments for Optimized Model B, which is built upon the
TransReID framework with a DeiT backbone, trained and evaluated on the VeRi-776 dataset.

- Auto-detect device (cuda > mps > cpu)
- Auto-resume from latest checkpoint in OUTPUT_DIR
- Train with per-epoch checkpoints
- Evaluate each checkpoint epoch-by-epoch
- Support --only-epoch N to evaluate a single checkpoint
- Logs to files + console

[2025-08-30 | Hang Zhang] Updated CLI help; added --target-global.
[2025-08-30 | Hang Zhang] Fixed argparse mapping: use args.target_global.
[2025-08-30 | Hang Zhang] Ensure training/rename/eval blocks always run.
[2025-09-13 | Hang Zhang] Added --tag argument to isolate output dirs for different experiments.
[2025-09-13 | Hang Zhang] Changed default --config to b0 baseline and auto-derived tag from config when --tag is not provided.
[2025-09-13 | Hang Zhang] Append `_deit` suffix to output directories for clarity.
[2025-09-14 | Hang Zhang] Changed default epochs to 30 for quicker experiments.
[2025-09-14 | Hang Zhang] Added dataset/pretrained path fallback (relative → Google Drive).
[2025-09-14 | Hang Zhang] Finalize dataset fallback: default ./datasets; if missing, use Drive datasets/VeRi.
[2025-09-14 | Hang Zhang] Added PRETRAIN fallback (local → Google Drive) with clear logs.
[2025-09-15 | Hang Zhang] Add --opts passthrough (YACS overrides). Accept TAG/OUTPUT_DIR via --opts to override local tag/log root safely.
[2025-09-16 | Hang Zhang] Print environment info (Python/PyTorch/CUDA/MPS/GPU/CPU/OS) at start; logs to console and train.log.
[2025-09-16 | Hang Zhang] Fix resume sorting (numeric sort for transformer_*.pth) and auto-append seed to tag when SOLVER.SEED is provided.
[2025-09-16 | Hang Zhang] De-duplicate SOLVER.MAX_EPOCHS: parse from --opts, remove from forwarded_opts, and override args.epochs
"""

import argparse
import sys
import subprocess
import shutil
from pathlib import Path
import re
import time
import os

def detect_device() -> str:
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return "cpu"

def run_and_tee(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", buffering=1, encoding="utf-8") as lf:
        lf.write(f"\n$ {' '.join(cmd)}\n")
        lf.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            lf.write(line)
        return proc.wait()

def sanity_checks(cfg: Path, data_root: Path, pretrained: Path, require_viewpoint: bool) -> None:
    for script in ("train.py", "test.py"):
        if not Path(script).exists():
            raise SystemExit(f"ERROR: {script} not found in current directory.")
    if not cfg.exists():
        raise SystemExit(f"ERROR: Config not found: {cfg}")
    expected = [
        data_root / "VeRi" / "image_train",
        data_root / "VeRi" / "image_query",
        data_root / "VeRi" / "image_test",
    ]
    missing = [str(x) for x in expected if not x.exists()]
    if missing:
        raise SystemExit("ERROR: VeRi folders missing:\n  " + "\n  ".join(missing))
    if require_viewpoint:
        for f in ("datasets/keypoint_train.txt", "datasets/keypoint_test.txt"):
            if not Path(f).exists():
                raise SystemExit(f"ERROR: Missing viewpoint file: {f}")

_num_re = re.compile(r"transformer_(\d+)\.pth$")

def find_latest_ckpt(run_dir: Path) -> Path | None:
    ckpts = list(run_dir.glob("transformer_*.pth"))
    if not ckpts:
        return None
    ckpts.sort(key=lambda p: int(_num_re.search(p.name).group(1)) if _num_re.search(p.name) else -1)
    return ckpts[-1]

def is_github_repo(cwd: Path) -> bool:
    try:
        if (cwd / ".git").exists():
            remotes = subprocess.check_output(["git", "remote", "-v"], text=True)
            return "github.com" in remotes
        return False
    except Exception:
        return False

def drive_mounted() -> bool:
    return Path("/content/drive/MyDrive").exists()

DRIVE_PRETRAIN_PATH = Path(
    "/content/drive/MyDrive/5703(hzha0521)/Optimized Model B (transformer based)/pretrained/deit_base_distilled_patch16_224-df68dfff.pth"
)

def print_env_info() -> None:
    try:
        import platform, torch  # type: ignore
        print("========== Environment Info ==========")
        print(f"Python        : {platform.python_version()}")
        print(f"PyTorch       : {getattr(torch, '__version__','<unknown>')}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"MPS available : {getattr(torch.backends,'mps',None) and torch.backends.mps.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device   : {torch.cuda.get_device_name(0)}")
        print(f"CPU           : {platform.processor() or f'{os.cpu_count()} logical cores'}")
        print(f"OS            : {platform.system()} {platform.release()}")
        print("======================================")
    except Exception as e:
        print(f"[env] Warning: failed to collect env info: {e}")

def main():
    parser = argparse.ArgumentParser(description="Optimized Model B Launcher (DeiT backbone, VeRi-776).")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs for THIS run (default: 30).")
    parser.add_argument("--target-global", dest="target_global", type=int, default=None)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--config", type=str, default="configs/VeRi/deit_transreid_stride_b0_baseline.yml")
    parser.add_argument("--data-root", type=str, default="./datasets")
    parser.add_argument("--pretrained", type=str, default="pretrained/deit_base_distilled_patch16_224-df68dfff.pth")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "mps", "cpu"])
    parser.add_argument("--require-viewpoint", action="store_true")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--only-epoch", type=int, default=None)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    forwarded_opts: list[str] = []
    tag_from_opts: str | None = None
    output_root_from_opts: Path | None = None
    seed_from_opts: str | None = None
    epochs_from_opts: int | None = None   # <--- NEW

    if args.opts:
        if len(args.opts) % 2 != 0:
            raise SystemExit("ERROR: --opts must contain KEY VALUE pairs.")
        for k, v in zip(args.opts[0::2], args.opts[1::2]):
            if k == "OUTPUT_DIR":
                output_root_from_opts = Path(v).resolve()
                print(f"[opts] OUTPUT_DIR captured: {output_root_from_opts}")
                continue
            if k == "TAG":
                tag_from_opts = str(v)
                print(f"[opts] TAG captured: {tag_from_opts}")
                continue
            if k == "SOLVER.SEED":
                seed_from_opts = str(v)
                forwarded_opts += [k, v]
                continue
            if k == "SOLVER.MAX_EPOCHS":     # <--- NEW
                try:
                    epochs_from_opts = int(v)
                except:
                    raise SystemExit(f"ERROR: SOLVER.MAX_EPOCHS expects int, got: {v}")
                continue
            forwarded_opts += [k, v]

    if epochs_from_opts is not None:
        args.epochs = epochs_from_opts

    device = args.device or detect_device()
    workers = args.num_workers if args.num_workers is not None else (8 if device in ("cuda","mps") else 0)

    cfg = Path(args.config).resolve()
    pretrained = Path(args.pretrained).resolve()
    if not pretrained.exists() and drive_mounted() and DRIVE_PRETRAIN_PATH.exists():
        print(f"[info] Using Drive pretrained: {DRIVE_PRETRAIN_PATH}")
        pretrained = DRIVE_PRETRAIN_PATH

    data_root = Path(args.data_root).resolve()
    if not (data_root / "VeRi").exists():
        drive_data = Path("/content/drive/MyDrive/5703(hzha0521)/Optimized Model B (transformer based)/datasets/VeRi")
        if drive_data.exists():
            data_root = drive_data.parent
            print(f"[info] Using Drive dataset at: {data_root}")

    tag = args.tag.strip() if args.tag else (tag_from_opts or cfg.stem.replace("deit_transreid_stride_",""))
    if seed_from_opts and "seed" not in tag.lower():
        tag = f"{tag}_seed{seed_from_opts}"

    cwd = Path.cwd().resolve()
    drive_root = Path("/content/drive/MyDrive/5703(hzha0521)/Optimized Model B (transformer based)")
    if output_root_from_opts is not None:
        log_root = output_root_from_opts
    elif is_github_repo(cwd) and drive_mounted():
        log_root = drive_root / "logs"
    else:
        log_root = Path("logs").resolve()

    out_run = (log_root / f"veri776_{tag}_deit_run").resolve()
    out_test = (log_root / f"veri776_{tag}_deit_test").resolve()
    train_log = out_run / "train.log"
    test_log = out_test / "test_all.log"
    out_run.mkdir(parents=True, exist_ok=True)
    out_test.mkdir(parents=True, exist_ok=True)
    Path("pretrained").mkdir(parents=True, exist_ok=True)

    sanity_checks(cfg, data_root, pretrained, args.require_viewpoint)
    latest_ckpt = find_latest_ckpt(out_run)
    is_resume = latest_ckpt is not None
    prev_max = max([int(_num_re.search(p.name).group(1)) for p in out_run.glob("transformer_*.pth") if _num_re.search(p.name)], default=0)

    if args.target_global is not None and prev_max < args.target_global:
        args.epochs = args.target_global - prev_max

    start_ts = time.time()
    print_env_info()
    print(f"Epochs (this run): {args.epochs} | Batch: {args.batch}")

    py = shutil.which(args.python) or args.python

    if not args.skip_train:
        train_cmd = [
            py, "train.py", "--config_file", str(cfg),
            "MODEL.DEVICE", device,
            "DATASETS.ROOT_DIR", str(data_root),
            "DATALOADER.NUM_WORKERS", str(workers),
            "SOLVER.IMS_PER_BATCH", str(args.batch),
            "SOLVER.MAX_EPOCHS", str(args.epochs),
            "SOLVER.CHECKPOINT_PERIOD", "1",
            "OUTPUT_DIR", str(out_run),
        ]
        if is_resume:
            train_cmd += ["MODEL.PRETRAIN_CHOICE","resume","MODEL.PRETRAIN_PATH",str(latest_ckpt)]
        else:
            train_cmd += ["MODEL.PRETRAIN_CHOICE","imagenet","MODEL.PRETRAIN_PATH",str(pretrained)]
        if forwarded_opts:
            train_cmd += forwarded_opts
        code = run_and_tee(train_cmd, train_log)
        if code != 0:
            raise SystemExit(code)

    try:
        for e in range(1, args.epochs+1):
            src = out_run / f"transformer_{e}.pth"
            if src.exists() and os.path.getmtime(src) >= start_ts - 1:
                dst = out_run / f"transformer_{prev_max+e}.pth"
                if not dst.exists():
                    src.rename(dst)
    except Exception as err:
        print(f"[rename] Warning: {err}")

    test_log.parent.mkdir(parents=True, exist_ok=True)
    with open(test_log,"a",encoding="utf-8") as f: f.write("===== Begin per-epoch evaluation =====\n")
    if args.only_epoch is not None:
        epochs_to_test = [args.only_epoch]
    else:
        epochs_to_test = range(max(1, prev_max+1), prev_max+args.epochs+1)

    for e in epochs_to_test:
        ckpt = out_run / f"transformer_{e}.pth"
        if ckpt.exists():
            subtest = out_test / f"epoch_{e}"
            subtest.mkdir(parents=True, exist_ok=True)
            test_cmd = [
                py, "test.py", "--config_file", str(cfg),
                "MODEL.DEVICE", device,
                "DATASETS.ROOT_DIR", str(data_root),
                "TEST.WEIGHT", str(ckpt),
                "OUTPUT_DIR", str(subtest),
            ]
            if forwarded_opts:
                test_cmd += forwarded_opts
            code = run_and_tee(test_cmd, test_log)
            if code != 0: raise SystemExit(code)

if __name__ == "__main__":
    main()
