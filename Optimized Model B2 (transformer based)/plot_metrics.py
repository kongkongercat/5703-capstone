#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot mAP and CMC (Rank-1/5/10) trends from TransReID test_all.log

[2025-09-01 | Hang Zhang]
- Added CLI (log path, output image, title).
- Auto-detect default log if script is placed inside the test folder.
- Parse metrics robustly from test_all.log.
- Plot curves and SAVE the figure.
- Annotate last-epoch values and best-mAP point.
"""

import argparse
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt


def parse_log(log_text: str):
    """
    Parse epochs and metrics from test_all.log text.
    Expected snippets like:
        ... epoch_12 ...
        mAP: 72.3%
        CMC curve, Rank-1: 88.1%
        CMC curve, Rank-5: 94.5%
        CMC curve, Rank-10: 96.7%
    Returns: lists (epochs, mAPs, r1s, r5s, r10s)
    """
    # Match an "epoch_N" block and capture numbers that follow in that block
    block_pat = re.compile(
        r"epoch_(\d+).*?mAP:\s*([\d.]+)%.*?Rank-1\s*:\s*([\d.]+)%.*?Rank-5\s*:\s*([\d.]+)%.*?Rank-10\s*:\s*([\d.]+)%",
        re.S
    )

    epochs, mAPs, r1s, r5s, r10s = [], [], [], [], []
    for m in block_pat.finditer(log_text):
        epochs.append(int(m.group(1)))
        mAPs.append(float(m.group(2)))
        r1s.append(float(m.group(3)))
        r5s.append(float(m.group(4)))
        r10s.append(float(m.group(5)))

    # Sort by epoch index just in case
    idx = sorted(range(len(epochs)), key=lambda i: epochs[i])
    epochs = [epochs[i] for i in idx]
    mAPs   = [mAPs[i]   for i in idx]
    r1s    = [r1s[i]    for i in idx]
    r5s    = [r5s[i]    for i in idx]
    r10s   = [r10s[i]   for i in idx]
    return epochs, mAPs, r1s, r5s, r10s


def annotate_last_and_best(ax, x, y, label: str, highlight_best: bool = False):
    """
    Annotate the last point and optionally the best (max) point on the curve.
    """
    if not x:
        return
    # Last epoch annotation
    ax.annotate(f"{label} @e{x[-1]}: {y[-1]:.1f}%",
                xy=(x[-1], y[-1]),
                xytext=(5, 8),
                textcoords="offset points",
                fontsize=9)

    # Best point (by max y)
    if highlight_best:
        best_i = max(range(len(y)), key=lambda i: y[i])
        ax.scatter([x[best_i]], [y[best_i]], s=50, marker="*", zorder=5)
        ax.annotate(f"BEST {label} @e{x[best_i]}: {y[best_i]:.1f}%",
                    xy=(x[best_i], y[best_i]),
                    xytext=(5, -12),
                    textcoords="offset points",
                    fontsize=9)


def main():
    parser = argparse.ArgumentParser(description="Plot TransReID test metrics trends.")
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Path to test_all.log. If omitted: "
             "1) use ./test_all.log if it exists; "
             "2) else try ./logs/veri776_deit_test/test_all.log."
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output image path (.png). Default: metrics_trend_YYYYMMDD_HHMMSS.png in the log folder."
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Performance Trend on VeRi-776",
        help="Figure title."
    )
    args = parser.parse_args()

    # Resolve log path
    if args.log is not None:
        log_path = args.log
    else:
        if os.path.exists("test_all.log"):
            log_path = "test_all.log"
        elif os.path.exists("logs/veri776_deit_test/test_all.log"):
            log_path = "logs/veri776_deit_test/test_all.log"
        elif os.path.exists("logs/veri776_vit_test/test_all.log"):
            log_path = "logs/veri776_vit_test/test_all.log"
        else:
            raise SystemExit("ERROR: test_all.log not found. Use --log to specify its path.")

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        log_text = f.read()

    epochs, mAPs, r1s, r5s, r10s = parse_log(log_text)
    if not epochs:
        raise SystemExit("ERROR: No metrics parsed from log. Check the log format or path.")

    # Prepare figure
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    ax.plot(epochs, mAPs,  marker="o", label="mAP")
    ax.plot(epochs, r1s,   marker="s", label="Rank-1")
    ax.plot(epochs, r5s,   marker="^", label="Rank-5")
    ax.plot(epochs, r10s,  marker="d", label="Rank-10")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(args.title)
    ax.grid(True)
    ax.legend()

    # Annotations: last points for all curves, best for mAP
    annotate_last_and_best(ax, epochs, mAPs,  "mAP",     highlight_best=True)
    annotate_last_and_best(ax, epochs, r1s,   "R1")
    annotate_last_and_best(ax, epochs, r5s,   "R5")
    annotate_last_and_best(ax, epochs, r10s,  "R10")

    plt.tight_layout()

    # Decide output filename
    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.dirname(os.path.abspath(log_path))
        out_path = os.path.join(out_dir, f"metrics_trend_{ts}.png")
    else:
        out_path = args.out

    plt.savefig(out_path, dpi=150)
    print(f"[Saved] {out_path}")

    # Optionally show (comment out if running on headless servers)
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
