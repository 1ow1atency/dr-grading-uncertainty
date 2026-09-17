"""Trains the EfficientNet-B0 DR severity classifier and logs per-epoch metrics."""

import argparse
import csv
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

try:
    from .data_loader import NUM_CLASSES, get_dataloaders
    from .model import build_model
except ImportError:
    from data_loader import NUM_CLASSES, get_dataloaders
    from model import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train the DR severity classifier.")
    parser.add_argument("--csv_path", type=str, default="data/raw/train.csv")
    parser.add_argument("--img_dir", type=str, default="data/raw/train_images")
    parser.add_argument("--img_ext", type=str, default=".png")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scheduler", type=str, choices=["plateau", "cosine"], default="plateau")
    parser.add_argument("--checkpoint_dir", type=str, default="results/checkpoints")
    parser.add_argument("--log_csv", type=str, default="results/train_log.csv")
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Returns (avg_loss, accuracy, macro_f1, per_class) so a majority-class model
    that scores well on accuracy alone still shows up as weak on macro-F1 / the
    minority classes' precision-recall."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        all_preds.append(outputs.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    avg_loss = total_loss / len(loader.dataset)

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    per_class = {
        c: {"precision": precision[c], "recall": recall[c], "f1": f1[c]} for c in range(NUM_CLASSES)
    }
    return avg_loss, accuracy, macro_f1, per_class


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, _test_loader, class_weights = get_dataloaders(
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

    model = build_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_csv), exist_ok=True)
    best_ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pt")

    fieldnames = ["epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1"]
    for c in range(NUM_CLASSES):
        fieldnames += [f"class_{c}_precision", f"class_{c}_recall", f"class_{c}_f1"]

    best_macro_f1 = -1.0
    with open(args.log_csv, "w", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            start = time.time()
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_macro_f1, per_class = evaluate(model, val_loader, criterion, device)

            if args.scheduler == "cosine":
                scheduler.step()
            else:
                scheduler.step(val_macro_f1)

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_macro_f1,
            }
            for c in range(NUM_CLASSES):
                row[f"class_{c}_precision"] = per_class[c]["precision"]
                row[f"class_{c}_recall"] = per_class[c]["recall"]
                row[f"class_{c}_f1"] = per_class[c]["f1"]
            writer.writerow(row)
            log_file.flush()

            elapsed = time.time() - start
            print(
                f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_macro_f1:.4f} "
                f"({elapsed:.1f}s)"
            )

            if val_macro_f1 > best_macro_f1:
                best_macro_f1 = val_macro_f1
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_macro_f1": val_macro_f1,
                        "val_accuracy": val_acc,
                    },
                    best_ckpt_path,
                )
                print(f"  -> new best model saved to {best_ckpt_path} (macro-F1={val_macro_f1:.4f})")

    print(f"Training complete. Best val macro-F1: {best_macro_f1:.4f}")


if __name__ == "__main__":
    main()
