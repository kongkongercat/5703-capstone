#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TransReID - VeRi-776 launcher (Model B experiments, DeiT backbone)

This launcher is dedicated to Model B experiments, which are based on the DeiT backbone
within the TransReID framework and trained/evaluated on the VeRi-776 dataset.

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
    """Return best-available device: 'cuda' > 'mps' > 'cpu'."""
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return "cpu"


def run_and_tee(cmd: list[str], log_path: Path) -> int:
    """Run a subprocess and tee stdout to console + logfile."""
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
    """Basic existence checks for scripts, config and dataset."""
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


def find_latest_ckpt(run_dir: Path) -> Path | None:
    """Return newest transformer_*.pth in run_dir, or None if absent."""
    ckpts = sorted(run_dir.glob("transformer_*.pth"))
    return ckpts[-1] if ckpts else None


def is_github_repo(cwd: Path) -> bool:
    """Return True if current code is a Git repo cloned from GitHub."""
    try:
        if (cwd / ".git").exists():
            remotes = subprocess.check_output(["git", "remote", "-v"], text=True)
            return "github.com" in remotes
        return False
    except Exception:
        return False


def drive_mounted() -> bool:
    """Return True if Google Drive is mounted at /content/drive/MyDrive."""
    return Path("/content/drive/MyDrive").exists()


# Fixed Google Drive pretrained path (fallback target)
DRIVE_PRETRAIN_PATH = Path(
    "/content/drive/MyDrive/5703(hzha0521)/Optimized Model B (transformer based)/pretrained/deit_base_distilled_patch16_224-df68dfff.pth"
)


def main():
    parser = argparse.ArgumentParser(description="TransReID VeRi-776 launcher (Model B - DeiT backbone).")

    # Core training controls
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs for THIS run (default: 30).")
    parser.add_argument("--target-global", dest="target_global", type=int, default=None,
                        help="Global target epoch index; e.g., 20 means you want transformer_1..20 in total.")
    parser.add_argument("--batch", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Dataloader workers (default: 8 if cuda/mps else 0)")

    # Config & paths
    parser.add_argument("--config", type=str,
                        default="configs/VeRi/deit_transreid_stride_b0_baseline.yml",
                        help="DeiT config (default: b0 baseline).")
    parser.add_argument("--data-root", type=str, default="./datasets",
                        help="Dataset root containing VeRi/ (default: ./datasets)")
    parser.add_argument(
        "--pretrained",
        type=str,
        default="pretrained/deit_base_distilled_patch16_224-df68dfff.pth",
        help="DeiT ImageNet weight. Will fallback to fixed Google Drive path if local file is missing and Drive is mounted."
    )
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "mps", "cpu"],
                        help="Force device; if omitted, auto-detect")
    parser.add_argument("--require-viewpoint", action="store_true", help="Check datasets/keypoint_*.txt exist")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable for train/test")

    # Eval & run control
    parser.add_argument("--skip-train", action="store_true", help="Skip training and only evaluate")
    parser.add_argument("--start-epoch", type=int, default=1, help="Eval from this epoch (default: 1)")
    parser.add_argument("--only-epoch", type=int, default=None,
                        help="If set, only evaluate this epoch (e.g., 120). Overrides start-epoch/epochs.")
    parser.add_argument("--tag", type=str, default=None,
                        help="Short tag for log folders (e.g., b0 / b1 / b2 / b3_pre / b3_fine). "
                             "If omitted, a tag is derived from the config basename.")

    # NEW: passthrough YACS overrides; safe-handle TAG/OUTPUT_DIR coming from --opts
    parser.add_argument(
        "--opts",
        nargs=argparse.REMAINDER,
        help="Extra YACS overrides as KEY VALUE pairs, e.g. "
             "LOSS.SUPCON.ENABLE True LOSS.SUPCON.T 0.07 LOSS.SUPCON.W 0.30 "
             "SOLVER.MAX_EPOCHS 12 OUTPUT_DIR ./logs TAG myrun"
    )

    args = parser.parse_args()

    # ---- Parse --opts into pairs; capture OUTPUT_DIR/TAG overrides, forward the rest ----
    forwarded_opts: list[str] = []
    tag_from_opts: str | None = None
    output_root_from_opts: Path | None = None
    if args.opts:
        if len(args.opts) % 2 != 0:
            raise SystemExit("ERROR: --opts must contain KEY VALUE pairs (even number of tokens).")
        for k, v in zip(args.opts[0::2], args.opts[1::2]):
            if k == "OUTPUT_DIR":
                output_root_from_opts = Path(v).resolve()
                print(f"[opts] OUTPUT_DIR captured as log root override: {output_root_from_opts}")
                continue
            if k == "TAG":
                tag_from_opts = str(v)
                print(f"[opts] TAG captured as tag override: {tag_from_opts}")
                continue
            forwarded_opts += [k, v]

    # Device & worker setup
    device = args.device or detect_device()
    workers = args.num_workers if args.num_workers is not None else (8 if device in ("cuda", "mps") else 0)

    # Resolve paths
    cfg = Path(args.config).resolve()
    pretrained = Path(args.pretrained).resolve()

    # PRETRAIN fallback (local → Google Drive)
    if not pretrained.exists():
        if drive_mounted() and DRIVE_PRETRAIN_PATH.exists():
            print(f"[info] Local pretrained not found → using Google Drive pretrained: {DRIVE_PRETRAIN_PATH}")
            pretrained = DRIVE_PRETRAIN_PATH
        else:
            print(f"[warn] Pretrained not found at {pretrained}; no Drive fallback available. "
                  f"Training will fail unless you provide a valid file.")

    # Dataset root with fallback (local → Google Drive)
    data_root = Path(args.data_root).resolve()
    if not (data_root / "VeRi").exists():
        drive_data = Path("/content/drive/MyDrive/5703(hzha0521)/Optimized Model B (transformer based)/datasets/VeRi")
        if drive_data.exists():
            data_root = drive_data.parent
            print(f"[info] Local dataset not found → using Google Drive dataset at: {data_root}")
        else:
            print(f"[warn] VeRi dataset not found at {data_root} or Google Drive; will fail in sanity_checks.")

    # Tag derivation (CLI --tag > --opts TAG > derive from config)
    if args.tag is not None and args.tag.strip():
        tag = args.tag.strip()
    elif tag_from_opts is not None and tag_from_opts.strip():
        tag = tag_from_opts.strip()
    else:
        stem = cfg.stem
        tag = stem[len("deit_transreid_stride_"):] if stem.startswith("deit_transreid_stride_") else stem

    # Decide log root (override via --opts OUTPUT_DIR; else Drive vs local heuristic)
    cwd = Path.cwd().resolve()
    drive_root = Path("/content/drive/MyDrive/5703(hzha0521)/Optimized Model B (transformer based)")
    if output_root_from_opts is not None:
        log_root = output_root_from_opts
        print(f"[info] Using log root from --opts OUTPUT_DIR: {log_root}")
    elif is_github_repo(cwd) and drive_mounted():
        log_root = drive_root / "logs"
        print("[info] Detected GitHub repo + mounted Drive → saving logs/checkpoints to Google Drive.")
    else:
        log_root = Path("logs").resolve()
        print("[info] Saving logs/checkpoints locally under ./logs")

    out_run = (log_root / f"veri776_{tag}_deit_run").resolve()
    out_test = (log_root / f"veri776_{tag}_deit_test").resolve()
    train_log = out_run / "train.log"
    test_log = out_test / "test_all.log"
    out_run.mkdir(parents=True, exist_ok=True)
    out_test.mkdir(parents=True, exist_ok=True)
    Path("pretrained").mkdir(parents=True, exist_ok=True)

    # Pre-flight checks
    sanity_checks(cfg, data_root, pretrained, args.require_viewpoint)

    latest_ckpt = find_latest_ckpt(out_run)
    is_resume = latest_ckpt is not None

    prev_max = 0
    for p in out_run.glob("transformer_*.pth"):
        m = re.search(r"transformer_(\d+)\.pth$", p.name)
        if m:
            prev_max = max(prev_max, int(m.group(1)))

    # Target-global logic
    if args.target_global is not None:
        if prev_max >= args.target_global:
            print(f"[info] Already reached target-global {args.target_global}, nothing to do.")
            return
        args.epochs = args.target_global - prev_max

    start_ts = time.time()

    # ----- Start banner -----
    print("========== Run Settings (DeiT / TransReID / VeRi-776) ==========")
    print(f"Using device      : {device} (num_workers={workers})")
    print(f"Epochs (this run) : {args.epochs} | Batch: {args.batch}")
    print(f"Config            : {cfg}")
    print(f"Tag               : {tag}")
    print(f"Data root (final) : {data_root}  (expects VeRi/*)")
    print(f"Logs / checkpoints: {out_run}")
    print(f"Eval outputs      : {out_test}")
    if is_resume:
        print(f"Resume from       : {latest_ckpt}")
    else:
        if not pretrained.exists():
            raise SystemExit(f"ERROR: Pretrained weight not found: {pretrained}")
        loc = "Google Drive" if str(pretrained).startswith("/content/drive/") else "local"
        print(f"Pretrained weight : {pretrained}  ({loc})")
    if forwarded_opts:
        # Show only keys for readability
        keys = ", ".join(forwarded_opts[0::2])
        print(f"[opts] Forwarding YACS keys to train/test: {keys}")
    print("===============================================================")

    py = shutil.which(args.python) or args.python

    # ------------------- Train -------------------
    if not args.skip_train:
        print("===== Begin training (DeiT) =====")
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
            train_cmd += ["MODEL.PRETRAIN_CHOICE", "resume", "MODEL.PRETRAIN_PATH", str(latest_ckpt)]
        else:
            train_cmd += ["MODEL.PRETRAIN_CHOICE", "imagenet", "MODEL.PRETRAIN_PATH", str(pretrained)]

        # Append forwarded --opts (without TAG/OUTPUT_DIR)
        if forwarded_opts:
            train_cmd += forwarded_opts

        code = run_and_tee(train_cmd, train_log)
        if code != 0:
            raise SystemExit(code)
    else:
        print(">> Skipping training (--skip-train).")

    # ------------------- Post-rename -------------------
    try:
        new_count = 0
        for e in range(1, args.epochs + 1):
            src = out_run / f"transformer_{e}.pth"
            if src.exists() and os.path.getmtime(src) >= start_ts - 1:
                dst = out_run / f"transformer_{prev_max + e}.pth"
                if not dst.exists():
                    src.rename(dst)
                    new_count += 1
        if new_count:
            print(f"[rename] Previous max: {prev_max}. Renamed {new_count} new checkpoints to continuous numbering.")
    except Exception as err:
        print(f"[rename] Warning: post-rename failed: {err}")

    # ------------------- Evaluate -------------------
    print("===== Begin per-epoch evaluation (DeiT) =====")
    test_log.parent.mkdir(parents=True, exist_ok=True)
    with open(test_log, "a", encoding="utf-8") as f:
        f.write("===== Begin per-epoch evaluation =====\n")

    if args.only_epoch is not None:
        epochs_to_test = [args.only_epoch]
    else:
        if not args.skip_train:
            epochs_to_test = range(max(1, prev_max + 1), prev_max + args.epochs + 1)
        else:
            epochs_to_test = range(max(1, args.start_epoch), args.epochs + 1)

    for e in epochs_to_test:
        ckpt = out_run / f"transformer_{e}.pth"
        if ckpt.exists():
            subtest = out_test / f"epoch_{e}"
            subtest.mkdir(parents=True, exist_ok=True)
            print(f">>> Testing epoch {e} with weight: {ckpt}")
            test_cmd = [
                py, "test.py", "--config_file", str(cfg),
                "MODEL.DEVICE", device,
                "DATASETS.ROOT_DIR", str(data_root),
                "TEST.WEIGHT", str(ckpt),
                "OUTPUT_DIR", str(subtest),
            ]
            # Forward the same YACS overrides to test (safe subset)
            if forwarded_opts:
                test_cmd += forwarded_opts

            code = run_and_tee(test_cmd, test_log)
            if code != 0:
                raise SystemExit(code)
        else:
            msg = f">>> Skip epoch {e} (checkpoint not found: {ckpt})\n"
            print(msg, end="")
            with open(test_log, "a", encoding="utf-8") as f:
                f.write(msg)

    print(f"===== All done (DeiT). Logs:\n  {out_run}\n  {out_test} =====")


if __name__ == "__main__":
    main()
