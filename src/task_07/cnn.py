"""Task 7-1 - a reproducible PyTorch CNN for segmented characters.

Dataset layout::

    data/processed/characters/A/*.png
    data/processed/characters/B/*.png

The module imports PyTorch lazily. This keeps the image-processing tasks
usable on machines where only the classical dependencies are installed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from common.datasets import discover_classification_dataset, stratified_split
from common.image_io import require_numpy
from common.features import resize_nearest


def _torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Task 7 CNN requires torch. Install requirements.txt or the CPU PyTorch build.") from exc
    return torch, nn, DataLoader, TensorDataset


def build_model(num_classes: int, image_size: int = 32):
    torch, nn, _, _ = _torch()

    class CharacterCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(64 * 4 * 4, 64), nn.ReLU(), nn.Linear(64, num_classes))

        def forward(self, inputs):
            return self.classifier(self.features(inputs))

    return CharacterCNN()


def make_tensor_dataset(images, labels, image_size: int = 32):
    torch, _, _, TensorDataset = _torch()
    np = require_numpy()
    array = np.stack([resize_nearest(image, (image_size, image_size)) for image in images]).astype("float32") / 255.0
    tensor = torch.from_numpy(array[:, None, :, :])
    return TensorDataset(tensor, torch.as_tensor(labels, dtype=torch.long))


def train(root, output, epochs: int = 20, batch_size: int = 32, learning_rate: float = 1e-3, test_fraction: float = 0.2, seed: int = 42, image_size: int = 32):
    torch, _, DataLoader, _ = _torch()
    torch.manual_seed(seed)
    images, labels, class_names = discover_classification_dataset(root)
    train_indices, test_indices = stratified_split(labels, test_fraction, seed)
    train_loader = DataLoader(make_tensor_dataset([images[i] for i in train_indices], labels[train_indices], image_size), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(make_tensor_dataset([images[i] for i in test_indices], labels[test_indices], image_size), batch_size=batch_size)
    model = build_model(len(class_names), image_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    history = []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, count = 0.0, 0, 0
        for batch_images, batch_labels in train_loader:
            optimizer.zero_grad()
            logits = model(batch_images)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(batch_labels)
            correct += int((logits.argmax(1) == batch_labels).sum().item())
            count += len(batch_labels)
        test_accuracy = evaluate_model(model, test_loader)
        history.append({"epoch": epoch, "train_loss": running_loss / max(1, count), "train_accuracy": correct / max(1, count), "test_accuracy": test_accuracy})
    duration = time.perf_counter() - start
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": class_names, "image_size": image_size, "history": history, "seed": seed}, output)
    return {"classes": class_names, "train_samples": len(train_indices), "test_samples": len(test_indices), "duration_seconds": duration, "history": history}


def evaluate_model(model, loader):
    torch, _, _, _ = _torch()
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            correct += int((model(images).argmax(1) == labels).sum().item())
            total += len(labels)
    return correct / max(1, total)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="class-directory dataset root")
    parser.add_argument("--model", required=True, help="output .pt checkpoint")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--metrics", help="optional JSON metrics output")
    args = parser.parse_args()
    metrics = train(args.data, args.model, args.epochs, args.batch_size, args.learning_rate, args.test_fraction, args.seed, args.image_size)
    if args.metrics:
        Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics).write_text(json.dumps(metrics, indent=2) + "\n")
    else:
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
