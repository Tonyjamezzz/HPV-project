import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import SimpleCNN


MODEL_REGISTRY = {"simplecnn": SimpleCNN}


def get_dataloaders(data_dir: str, batch_size: int) -> tuple[DataLoader, DataLoader]:
    """Prepare training and validation dataloaders."""
    data_path = Path(data_dir)
    train_tfms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]
    )
    val_tfms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]
    )
    train_ds = datasets.ImageFolder(data_path / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(data_path / "val", transform=val_tfms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for a single epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / len(loader.dataset)
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float]:
    """Evaluate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / len(loader.dataset)
    return epoch_loss, epoch_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a classifier on processed data")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--model", default="simplecnn", choices=MODEL_REGISTRY.keys())
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_dataloaders(args.data_dir, args.batch_size)

    num_classes = len(train_loader.dataset.classes)
    model = MODEL_REGISTRY[args.model](num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")


if __name__ == "__main__":
    main()
