"""
Train a CNN classifier on MNIST.

Usage:
    python -m src.train_cnn --epochs 10 --batch-size 64 --lr 0.001
"""
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from src.data import get_cnn_dataloaders
from src.models import MNISTCNN
from src.utils import get_logger, save_checkpoint, set_seed

logger = get_logger("train_cnn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN classifier on MNIST")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--checkpoint-dir", type=str, default="./outputs/cnn_models")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def train_one_epoch(model, device, train_loader, optimizer, criterion, epoch, log_interval):
    model.train()
    train_loss, correct = 0.0, 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % log_interval == 0:
            logger.info(
                "Train Epoch %d [%d/%d (%.0f%%)] Loss %.6f",
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),
            )

    train_loss /= len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)
    logger.info("Train set: Average loss %.4f, Accuracy %.2f%%", train_loss, accuracy)
    return train_loss, accuracy


def evaluate(model, device, test_loader, criterion):
    model.eval()
    test_loss, correct = 0.0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    logger.info("Test set: Average loss %.4f, Accuracy %.2f%%", test_loss, accuracy)
    return test_loss, accuracy


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)

    train_loader, test_loader = get_cnn_dataloaders(args.data_root, args.batch_size)

    model = MNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion, epoch, args.log_interval)
        test_loss, test_acc = evaluate(model, device, test_loader, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

    save_checkpoint(model.state_dict(), args.checkpoint_dir, "mnist_cnn_final.pth")
    logger.info("Training complete. Final test accuracy: %.2f%%", history["test_acc"][-1])


if __name__ == "__main__":
    train(parse_args())
