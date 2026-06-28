"""
    Compare two training runs side by side.

    Usage:
        python plot_compare.py <log_a> <log_b> [--labels "Label A" "Label B"] [--out output.png]

    Examples:
        python plot_compare.py logs/baseline.log logs/mainrun_validate_2026-06-22T14-30-00.log
        python plot_compare.py logs/baseline.log logs/mainrun_validate_base.log --labels Baseline Optimised
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


# Parses a JSON log file and returns train/val steps and losses normalised to epoch fractions.
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
    ## Normalise to epoch so runs with different dataset sizes are comparable
    epochs_x_train = [s / max_steps * 7 for s in train_steps] if max_steps else train_steps
    epochs_x_val   = [s / max_steps * 7 for s in val_steps]   if max_steps else val_steps
    return epochs_x_train, train_losses, epochs_x_val, val_losses


# Plots training and validation loss curves for two runs side by side and saves to file.
def plot(log_a, log_b, label_a, label_b, out):
    ta, la, va_x, va_y = parse_log(log_a)
    tb, lb, vb_x, vb_y = parse_log(log_b)

    final_a = f"{va_y[-1]:.4f}" if va_y else "N/A"
    final_b = f"{vb_y[-1]:.4f}" if vb_y else "N/A"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{label_a}  vs  {label_b}", fontsize=13, fontweight="bold")

    ## Training loss
    ax1.plot(ta, la, alpha=0.35, linewidth=0.7, color="steelblue",  label=label_a)
    ax1.plot(tb, lb, alpha=0.35, linewidth=0.7, color="darkorange", label=label_b)
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ## Validation loss
    ax2.plot(va_x, va_y, "o-", linewidth=1.8, markersize=4, color="steelblue",
             label=f"{label_a}  (final {final_a})")
    ax2.plot(vb_x, vb_y, "o-", linewidth=1.8, markersize=4, color="darkorange",
             label=f"{label_b}  (final {final_b})")
    if va_y:
        ax2.axhline(va_y[-1], color="red", linestyle="--", linewidth=1,
                    label=f"{label_a} final ({va_y[-1]:.4f})")
    ax2.set_title("Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Cross-entropy loss")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    print(f"  {label_a}: final val loss = {final_a}")
    print(f"  {label_b}: final val loss = {final_b}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two training run logs.")
    parser.add_argument("log_a", help="Path to first log file")
    parser.add_argument("log_b", help="Path to second log file")
    parser.add_argument("--labels", nargs=2, default=None,
                        metavar=("LABEL_A", "LABEL_B"),
                        help="Display names for the two runs")
    parser.add_argument("--out", default=None,
                        help="Output image path (default: comparison_<stem_a>_vs_<stem_b>.png)")
    args = parser.parse_args()

    label_a, label_b = args.labels if args.labels else (
        Path(args.log_a).stem, Path(args.log_b).stem
    )
    out = args.out or f"comparison_{Path(args.log_a).stem}_vs_{Path(args.log_b).stem}.png"

    plot(args.log_a, args.log_b, label_a, label_b, out)
