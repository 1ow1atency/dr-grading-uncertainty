"""Selective prediction analysis for the DR severity classifier.

Ranks validation images by MC Dropout predictive entropy (most uncertain
first) and measures how accuracy on the remaining, non-deferred images
changes as increasing fractions of the most uncertain images are deferred
(e.g. to a human reviewer). Reuses the per-image predictions already
computed by calibration.py rather than rerunning inference.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

MC_COLOR = "#0072B2"
NAIVE_COLOR = "#8A8A8A"


def parse_args():
    parser = argparse.ArgumentParser(description="Selective prediction analysis via MC Dropout uncertainty.")
    parser.add_argument("--per_image_csv", type=str, default="results/metrics/per_image_predictions.csv")
    parser.add_argument("--output_csv", type=str, default="results/metrics/selective_prediction.csv")
    parser.add_argument(
        "--plot_path",
        type=str,
        default="results/plots/selective_prediction/selective_prediction_curve.png",
    )
    parser.add_argument("--max_defer_pct", type=float, default=0.50)
    parser.add_argument("--step_pct", type=float, default=0.05)
    return parser.parse_args()


def selective_prediction_table(df: pd.DataFrame, max_defer_pct: float = 0.50, step_pct: float = 0.05) -> pd.DataFrame:
    """Sorts by predictive entropy (most uncertain first) and, for each
    deferral threshold, computes accuracy on the remaining (least
    uncertain) images."""
    ranked = df.sort_values("mc_entropy", ascending=False).reset_index(drop=True)
    n_total = len(ranked)

    n_steps = int(round(max_defer_pct / step_pct)) + 1
    thresholds = [round(i * step_pct, 10) for i in range(n_steps)]

    rows = []
    for t in thresholds:
        n_deferred = int(round(t * n_total))
        kept = ranked.iloc[n_deferred:]
        accuracy = kept["mc_correct"].mean() if len(kept) > 0 else float("nan")
        rows.append(
            {
                "pct_deferred": t,
                "n_deferred": n_deferred,
                "n_kept": len(kept),
                "accuracy": accuracy,
            }
        )
    return pd.DataFrame(rows)


def plot_selective_prediction(table: pd.DataFrame, naive_accuracy: float, plot_path: str):
    """Selective prediction curve: % deferred vs. accuracy on the rest,
    against a horizontal reference line at naive (0%-deferral) accuracy."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

    ax.plot(
        table["pct_deferred"] * 100, table["accuracy"],
        marker="o", markersize=6, linewidth=2, color=MC_COLOR,
        label="MC Dropout selective accuracy",
    )
    ax.axhline(
        naive_accuracy, linestyle="--", linewidth=1.5, color=NAIVE_COLOR,
        label=f"Naive full-dataset accuracy ({naive_accuracy:.3f})",
    )

    ax.set_xlabel("% deferred (most uncertain)")
    ax.set_ylabel("Accuracy on remaining images")
    ax.set_title("Selective Prediction Curve")
    ax.set_xlim(0, table["pct_deferred"].max() * 100)
    ax.legend(loc="best", frameon=True, facecolor="white", edgecolor="none", framealpha=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)


def main():
    args = parse_args()
    df = pd.read_csv(args.per_image_csv)

    naive_accuracy = df["naive_correct"].mean()
    table = selective_prediction_table(df, args.max_defer_pct, args.step_pct)

    print(f"Naive full-dataset accuracy: {naive_accuracy:.4f}")
    print(table.to_string(index=False))

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    table.to_csv(args.output_csv, index=False)
    print(f"Saved selective prediction table to {args.output_csv}")

    plot_selective_prediction(table, naive_accuracy, args.plot_path)
    print(f"Saved selective prediction curve to {args.plot_path}")


if __name__ == "__main__":
    main()
