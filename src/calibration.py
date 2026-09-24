"""Calibration evaluation for the DR severity classifier.

Compares a single deterministic ("naive") forward pass against MC Dropout
on the validation set: Expected Calibration Error (ECE), Brier score, and
a reliability diagram plotting confidence vs. actual accuracy per bin.
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

try:
    from .data_loader import NUM_CLASSES, get_dataloaders
    from .model import build_model, predict_with_uncertainty
except ImportError:
    from data_loader import NUM_CLASSES, get_dataloaders
    from model import build_model, predict_with_uncertainty

# Okabe-Ito colorblind-safe pair.
NAIVE_COLOR = "#D55E00"
MC_COLOR = "#0072B2"
DIAGONAL_COLOR = "#8A8A8A"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate calibration of the DR severity classifier.")
    parser.add_argument("--csv_path", type=str, default="data/raw/train.csv")
    parser.add_argument("--img_dir", type=str, default="data/raw/train_images")
    parser.add_argument("--img_ext", type=str, default=".png")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_path", type=str, default="results/checkpoints/best_model.pt")
    parser.add_argument("--n_mc_samples", type=int, default=30)
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--metrics_csv", type=str, default="results/metrics/calibration_metrics.csv")
    parser.add_argument(
        "--plot_path",
        type=str,
        default="results/plots/reliability_diagrams/reliability_comparison.png",
    )
    return parser.parse_args()


@torch.no_grad()
def naive_predictions(model, loader, device):
    """Single deterministic forward pass per image (dropout off via model.eval()).

    Returns (confidences, correct, probs, labels): max softmax probability as
    confidence, whether the argmax prediction matches the label, the full
    per-class probability vectors (for the Brier score), and the labels.
    """
    model.eval()
    confidences, correct, probs_all, labels_all = [], [], [], []
    for images, labels in loader:
        images = images.to(device)
        probs = F.softmax(model(images), dim=1)
        conf, pred = probs.max(dim=1)
        confidences.append(conf.cpu())
        correct.append((pred.cpu() == labels).numpy())
        probs_all.append(probs.cpu())
        labels_all.append(labels)
    return (
        torch.cat(confidences).numpy(),
        np.concatenate(correct),
        torch.cat(probs_all).numpy(),
        torch.cat(labels_all).numpy(),
    )


def mc_dropout_predictions(model, loader, device, n_samples):
    """MC Dropout predictions, one image at a time via predict_with_uncertainty.

    Returns (confidences, correct, entropies, probs, labels): max of the mean
    predicted probability as confidence, whether that prediction is correct,
    the predictive entropy as the uncertainty score, the mean probability
    vectors, and the labels.
    """
    confidences, correct, entropies, probs_all, labels_all = [], [], [], [], []
    for images, labels in loader:
        for image, label in zip(images, labels):
            mean_probs, entropy = predict_with_uncertainty(model, image, n_samples=n_samples)
            conf, pred = mean_probs.max(dim=0)
            confidences.append(conf.item())
            correct.append(pred.item() == label.item())
            entropies.append(entropy)
            probs_all.append(mean_probs.numpy())
            labels_all.append(label.item())
    model.eval()  # predict_with_uncertainty leaves Dropout in train mode
    return (
        np.array(confidences),
        np.array(correct),
        np.array(entropies),
        np.stack(probs_all),
        np.array(labels_all),
    )


def compute_ece(confidences, correct, n_bins=10):
    """Expected Calibration Error plus per-bin stats for the reliability diagram.

    Bins predictions into n_bins equal-width confidence bins over [0, 1]; ECE
    is the bin-count-weighted average gap between each bin's mean confidence
    and its actual accuracy.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(confidences, bin_edges[1:-1], right=True), 0, n_bins - 1)

    bin_confidence = np.full(n_bins, np.nan)
    bin_accuracy = np.full(n_bins, np.nan)
    bin_count = np.zeros(n_bins, dtype=int)

    n = len(confidences)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        bin_count[b] = count
        if count == 0:
            continue
        bin_confidence[b] = confidences[mask].mean()
        bin_accuracy[b] = correct[mask].mean()
        ece += (count / n) * abs(bin_accuracy[b] - bin_confidence[b])

    return ece, bin_edges, bin_confidence, bin_accuracy, bin_count


def brier_score(probs, labels, num_classes=NUM_CLASSES):
    """Multi-class Brier score: mean squared error between predicted
    probability vectors and one-hot labels, averaged over samples."""
    one_hot = np.eye(num_classes)[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def plot_reliability_diagram(naive_result, mc_result, plot_path):
    """Two-panel reliability diagram: accuracy vs. confidence per bin (top,
    naive vs. MC Dropout, against the perfect-calibration diagonal) and the
    per-bin sample counts (bottom), since sparsely populated bins produce
    noisy accuracy estimates that are easy to over-read.
    """
    naive_ece, bin_edges, naive_conf, naive_acc, naive_count = naive_result
    mc_ece, _, mc_conf, mc_acc, mc_count = mc_result

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    fig, (ax_rel, ax_hist) = plt.subplots(
        2, 1, figsize=(6, 7.5), dpi=150, height_ratios=[3, 1], sharex=True
    )

    ax_rel.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color=DIAGONAL_COLOR, label="Perfect calibration")

    naive_mask = ~np.isnan(naive_conf)
    mc_mask = ~np.isnan(mc_conf)
    ax_rel.plot(
        bin_centers[naive_mask], naive_acc[naive_mask],
        marker="o", markersize=6, linewidth=2, color=NAIVE_COLOR,
        label=f"Naive (ECE={naive_ece:.3f})",
    )
    ax_rel.plot(
        bin_centers[mc_mask], mc_acc[mc_mask],
        marker="o", markersize=6, linewidth=2, color=MC_COLOR,
        label=f"MC Dropout (ECE={mc_ece:.3f})",
    )

    ax_rel.set_xlim(0, 1)
    ax_rel.set_ylim(0, 1)
    ax_rel.set_ylabel("Accuracy")
    ax_rel.set_title("Reliability Diagram: Naive vs. MC Dropout")
    ax_rel.legend(loc="upper left", frameon=False)
    ax_rel.spines["top"].set_visible(False)
    ax_rel.spines["right"].set_visible(False)

    offset = bin_width * 0.2
    ax_hist.bar(bin_centers - offset, naive_count, width=bin_width * 0.4, color=NAIVE_COLOR, label="Naive")
    ax_hist.bar(bin_centers + offset, mc_count, width=bin_width * 0.4, color=MC_COLOR, label="MC Dropout")
    ax_hist.set_xlabel("Confidence")
    ax_hist.set_ylabel("Count")
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)

    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _train_loader, val_loader, _test_loader, _class_weights = get_dataloaders(
        csv_path=args.csv_path,
        img_dir=args.img_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_size=args.val_size,
        test_size=args.test_size,
        num_workers=args.num_workers,
        seed=args.seed,
        img_ext=args.img_ext,
    )

    model = build_model(pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Running naive (single forward pass) predictions...")
    naive_conf, naive_correct, naive_probs, naive_labels = naive_predictions(model, val_loader, device)

    print(f"Running MC Dropout predictions (n_samples={args.n_mc_samples})...")
    mc_conf, mc_correct, _mc_entropy, mc_probs, mc_labels = mc_dropout_predictions(
        model, val_loader, device, args.n_mc_samples
    )

    naive_result = compute_ece(naive_conf, naive_correct, args.n_bins)
    mc_result = compute_ece(mc_conf, mc_correct, args.n_bins)
    naive_ece = naive_result[0]
    mc_ece = mc_result[0]

    naive_brier = brier_score(naive_probs, naive_labels)
    mc_brier = brier_score(mc_probs, mc_labels)

    print(f"Naive:      ECE={naive_ece:.4f}  Brier={naive_brier:.4f}  n={len(naive_labels)}")
    print(f"MC Dropout: ECE={mc_ece:.4f}  Brier={mc_brier:.4f}  n={len(mc_labels)}")

    os.makedirs(os.path.dirname(args.metrics_csv), exist_ok=True)
    with open(args.metrics_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "ece", "brier_score", "n_bins", "n_samples"])
        writer.writeheader()
        writer.writerow(
            {"method": "naive", "ece": naive_ece, "brier_score": naive_brier, "n_bins": args.n_bins, "n_samples": len(naive_labels)}
        )
        writer.writerow(
            {"method": "mc_dropout", "ece": mc_ece, "brier_score": mc_brier, "n_bins": args.n_bins, "n_samples": len(mc_labels)}
        )
    print(f"Saved metrics to {args.metrics_csv}")

    plot_reliability_diagram(naive_result, mc_result, args.plot_path)
    print(f"Saved reliability diagram to {args.plot_path}")


if __name__ == "__main__":
    main()
