"""Fine-tune the CNN/LSTM action model from labelled normal/violent videos.

Run this on Google Colab (recommended):
    python train_violence_model.py --data data --epochs 15
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from action_model import ViolenceCNNLSTM

VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}


def read_clip(path, sequence_length, image_size):
    capture = cv2.VideoCapture(str(path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(total - 1, 0), sequence_length).astype(int)
    frames, last_frame = [], None
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if ok:
            last_frame = frame
        if last_frame is None:
            capture.release()
            raise ValueError(f"Cannot read video: {path}")
        frame = cv2.resize(last_frame, (image_size, image_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frame = (frame - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        frames.append(np.transpose(frame, (2, 0, 1)))
    capture.release()
    return torch.from_numpy(np.stack(frames)).float()


class ViolenceVideoDataset(Dataset):
    def __init__(self, items, sequence_length, image_size):
        self.items, self.sequence_length, self.image_size = items, sequence_length, image_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        return read_clip(path, self.sequence_length, self.image_size), torch.tensor(label, dtype=torch.float32)


def collect_items(data_dir):
    items = []
    for label, folder in ((0, "normal"), (1, "violent")):
        directory = Path(data_dir) / folder
        items.extend((path, label) for path in directory.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS)
    if not items:
        raise ValueError("Add .mp4/.avi clips under data/normal and data/violent first.")
    if not any(label == 0 for _, label in items) or not any(label == 1 for _, label in items):
        raise ValueError("Both normal and violent folders need at least one video.")
    return items


def collect_rwf_items(data_dir):
    """Collect Fight/NonFight clips from one RWF-2000 split directory."""
    root = Path(data_dir)
    items = []
    for label, folder in ((0, "NonFight"), (1, "Fight")):
        directory = root / folder
        items.extend(
            (path, label) for path in directory.rglob("*")
            if path.suffix.lower() in VIDEO_EXTENSIONS
        )
    if not items or not any(label == 0 for _, label in items) or not any(label == 1 for _, label in items):
        raise ValueError(f"Both Fight and NonFight folders need videos under {root}.")
    return items


def load_test_items(data_dir):
    """Accept either RWF-2000 (Fight/NonFight) or custom (normal/violent) labels."""
    root = Path(data_dir)
    if (root / "Fight").is_dir() or (root / "NonFight").is_dir():
        return collect_rwf_items(root)
    return collect_items(root)


def load_splits(data_dir):
    """Use RWF-2000's official train/val split when it is available."""
    root = Path(data_dir)
    official_train, official_validation = [], []
    for split_name, destination in (("train", official_train), ("val", official_validation)):
        for label, folder in ((0, "NonFight"), (1, "Fight")):
            directory = root / split_name / folder
            destination.extend(
                (path, label) for path in directory.rglob("*")
                if path.suffix.lower() in VIDEO_EXTENSIONS
            )
    if official_train and official_validation:
        if not any(label == 0 for _, label in official_train) or not any(label == 1 for _, label in official_train):
            raise ValueError("The RWF-2000 training split must contain Fight and NonFight clips.")
        return official_train, official_validation

    items = collect_items(data_dir)
    random.Random(42).shuffle(items)
    split = max(1, int(len(items) * 0.8))
    train_items, validation_items = items[:split], items[split:]
    return train_items, validation_items or train_items[-1:]


def empty_metrics():
    """Create counters for one binary-classification epoch."""
    return {"loss_total": 0.0, "samples": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0}


def update_metrics(metrics, logits, labels, loss):
    """Accumulate loss and threshold-0.5 binary classification metrics."""
    labels = labels.detach().bool().cpu()
    predictions = (logits.detach().cpu() >= 0.0)  # sigmoid(logit) >= 0.5
    metrics["samples"] += labels.numel()
    metrics["loss_total"] += loss.item() * labels.numel()
    metrics["tp"] += (predictions & labels).sum().item()
    metrics["tn"] += ((~predictions) & (~labels)).sum().item()
    metrics["fp"] += (predictions & (~labels)).sum().item()
    metrics["fn"] += ((~predictions) & labels).sum().item()


def finalise_metrics(metrics):
    """Return loss, accuracy, precision, recall, and F1 from epoch counters."""
    total = max(metrics["samples"], 1)
    precision_denominator = metrics["tp"] + metrics["fp"]
    recall_denominator = metrics["tp"] + metrics["fn"]
    precision = metrics["tp"] / precision_denominator if precision_denominator else 0.0
    recall = metrics["tp"] / recall_denominator if recall_denominator else 0.0
    f1_denominator = precision + recall
    return {
        "loss": metrics["loss_total"] / total,
        "accuracy": (metrics["tp"] + metrics["tn"]) / total,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / f1_denominator if f1_denominator else 0.0,
    }


def metrics_text(name, metrics):
    return (
        f"{name}: loss {metrics['loss']:.4f}, accuracy {metrics['accuracy']:.1%}, "
        f"precision {metrics['precision']:.1%}, recall {metrics['recall']:.1%}, "
        f"F1 {metrics['f1']:.1%}"
    )


def evaluate(model, loader, loss_fn, device):
    """Evaluate once without gradient updates and return binary metrics."""
    metrics = empty_metrics()
    model.eval()
    with torch.inference_mode():
        for clips, labels in loader:
            clips, labels = clips.to(device), labels.to(device)
            logits = model(clips)
            loss = loss_fn(logits, labels)
            update_metrics(metrics, logits, labels, loss)
    return finalise_metrics(metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--hidden-size", type=int, default=256, help="LSTM hidden units; use 64 or 128 for a small dataset.")
    parser.add_argument(
        "--freeze-backbone", action="store_true",
        help="Train only the LSTM classifier head; recommended for a very small dataset.",
    )
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--output", default="models/violence_action_model.pt")
    parser.add_argument(
        "--test-data", default=None,
        help="Separate untouched test folder: Fight/NonFight or normal/violent subfolders.",
    )
    parser.add_argument(
        "--patience", type=int, default=5,
        help="Stop after this many epochs without validation-F1 improvement; 0 disables it.",
    )
    args = parser.parse_args()

    train_items, validation_items = load_splits(args.data)
    test_items = load_test_items(args.test_data) if args.test_data else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_message = f"Training on {device}: {len(train_items)} train / {len(validation_items)} validation clips"
    if test_items:
        split_message += f" / {len(test_items)} held-out test clips"
    print(split_message)
    train_loader = DataLoader(ViolenceVideoDataset(train_items, args.sequence_length, args.image_size), batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(ViolenceVideoDataset(validation_items, args.sequence_length, args.image_size), batch_size=args.batch_size)
    test_loader = (
        DataLoader(ViolenceVideoDataset(test_items, args.sequence_length, args.image_size), batch_size=args.batch_size)
        if test_items else None
    )
    model = ViolenceCNNLSTM(hidden_size=args.hidden_size, pretrained=True).to(device)
    if args.freeze_backbone:
        for parameter in model.cnn.parameters():
            parameter.requires_grad = False
        print("MobileNetV3 feature backbone frozen; training LSTM classifier head only.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    best_f1 = -1.0
    epochs_without_improvement = 0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "sequence_length": args.sequence_length,
        "image_size": args.image_size,
        "threshold": args.threshold,
        "hidden_size": args.hidden_size,
        "lstm_layers": 1,
    }
    for epoch in range(args.epochs):
        model.train()
        train_metrics = empty_metrics()
        for clips, labels in train_loader:
            clips, labels = clips.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(clips)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            update_metrics(train_metrics, logits, labels, loss)

        train_results = finalise_metrics(train_metrics)
        validation_results = evaluate(model, validation_loader, loss_fn, device)
        print(f"Epoch {epoch + 1}/{args.epochs} | {metrics_text('train', train_results)}")
        print(f"                 | {metrics_text('validation', validation_results)}")

        if validation_results["f1"] > best_f1:
            best_f1 = validation_results["f1"]
            epochs_without_improvement = 0
            torch.save({
                "model_state": model.state_dict(),
                "config": config,
                "epoch": epoch + 1,
                "validation_metrics": validation_results,
            }, output)
            print(f"Saved new best checkpoint (validation F1 {best_f1:.1%})")
        else:
            epochs_without_improvement += 1
            if args.patience and epochs_without_improvement >= args.patience:
                print(f"Early stopping after {args.patience} epochs without validation-F1 improvement.")
                break
    print(f"Saved best model to {output} (validation F1 {best_f1:.1%})")
    if test_loader:
        checkpoint = torch.load(output, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        test_results = evaluate(model, test_loader, loss_fn, device)
        print(metrics_text("held-out test", test_results))
    else:
        print("Test metrics not run: reserve separate clips and pass them with --test-data.")


if __name__ == "__main__":
    main()
