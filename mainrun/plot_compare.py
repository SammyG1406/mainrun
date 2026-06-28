"""
    Compare N training runs side by side.

    Usage:
        python plot_compare.py <log1> [log2 ...] [--labels "L1" "L2" ...] [--out output.png]

    Examples:
        python plot_compare.py logs/baseline.log logs/mainrun_validate_2026-06-22T10-06-42wAdamW.log
        python plot_compare.py logs/baseline.log logs/a.log logs/b.log logs/c.log --labels Baseline AdamW RoPE Scaled
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def parse_log(path: str):
    train_steps, train_losses = [], []
    val_steps, val_losses = [], []
    max_steps = None
    with open(path) as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = e.get("event")
            if event == "training_step":
                train_steps.append(e["step"])
                train_losses.append(e["loss"])
                max_steps = e.get("max_steps", max_steps)
            elif event == "validation_step":
                val_steps.append(e["step"])
                val_losses.append(e["loss"])
                max_steps = e.get("max_steps", max_steps)
    epochs_x_train = [s / max_steps * 7 for s in train_steps] if max_steps else train_steps
    epochs_x_val   = [s / max_steps * 7 for s in val_steps]   if max_steps else val_steps
    return epochs_x_train, train_losses, epochs_x_val, val_losses


def plot(logs, labels, out):
    n = len(logs)
    colours = [cm.tab10(i / max(n, 10)) for i in range(n)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training run comparison", fontsize=13, fontweight="bold")

    for path, label, colour in zip(logs, labels, colours):
        tx, tl, vx, vl = parse_log(path)
        final = f"{vl[-1]:.4f}" if vl else "N/A"

        ax1.plot(tx, tl, alpha=0.35, linewidth=0.7, color=colour, label=label)
        ax2.plot(vx, vl, "o-", linewidth=1.8, markersize=4, color=colour,
                 label=f"{label}  ({final})")

    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Cross-entropy loss")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    for label, path in zip(labels, logs):
        _, _, _, vl = parse_log(path)
        print(f"  {label}: {vl[-1]:.4f}" if vl else f"  {label}: N/A")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare N training run logs.")
    parser.add_argument("logs", nargs="+", help="Paths to log files")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Display names (must match number of logs)")
    parser.add_argument("--out", default=None,
                        help="Output image path (default: comparison.png)")
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.logs):
        parser.error(f"--labels count ({len(args.labels)}) must match log count ({len(args.logs)})")

    labels = args.labels if args.labels else [Path(p).stem for p in args.logs]
    out    = args.out or "comparison.png"

    plot(args.logs, labels, out)
