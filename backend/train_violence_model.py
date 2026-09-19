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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--output", default="models/violence_action_model.pt")
    args = parser.parse_args()

    train_items, validation_items = load_splits(args.data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}: {len(train_items)} train / {len(validation_items)} validation clips")
    train_loader = DataLoader(ViolenceVideoDataset(train_items, args.sequence_length, args.image_size), batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(ViolenceVideoDataset(validation_items, args.sequence_length, args.image_size), batch_size=args.batch_size)
    model = ViolenceCNNLSTM(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    best_accuracy = -1.0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    config = {"sequence_length": args.sequence_length, "image_size": args.image_size, "threshold": args.threshold, "hidden_size": 256, "lstm_layers": 1}
    for epoch in range(args.epochs):
        model.train()
        for clips, labels in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(clips.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        correct = total = 0
        with torch.inference_mode():
            for clips, labels in validation_loader:
                predictions = torch.sigmoid(model(clips.to(device))) >= 0.5
                correct += (predictions.cpu() == labels.bool()).sum().item()
                total += len(labels)
        accuracy = correct / total
        print(f"Epoch {epoch + 1}/{args.epochs}: validation accuracy {accuracy:.1%}")
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            torch.save({"model_state": model.state_dict(), "config": config}, output)
    print(f"Saved best model to {output} (validation accuracy {best_accuracy:.1%})")


if __name__ == "__main__":
    main()
