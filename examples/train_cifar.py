import argparse
import contextlib
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18

torch.cuda.set_per_process_memory_fraction(0.5)

parser = argparse.ArgumentParser(description="Train ResNet-18 on CIFAR-10.")
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument(
    "--bug",
    choices=["eval-no-grad", "train-total-loss"],
    help="eval-no-grad: evaluate without torch.no_grad() and add up the loss tensor. "
    "train-total-loss: add up the loss tensor in the training loop after backward() "
    "(harmless, oomless must not warn)",
)
args = parser.parse_args()

torch.manual_seed(0)
root = Path.home() / ".cache" / "torchvision"
normalize = transforms.Normalize((0.49, 0.48, 0.45), (0.25, 0.24, 0.26))
augment = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
train_set = datasets.CIFAR10(
    root,
    train=True,
    download=True,
    transform=transforms.Compose([*augment, transforms.ToTensor(), normalize]),
)
test_set = datasets.CIFAR10(
    root,
    train=False,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(), normalize]),
)
train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_set, args.batch_size, num_workers=4, pin_memory=True)

model = resnet18(num_classes=10)
# The photos are only 32x32 pixels, so don't shrink them at the start.
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
model.maxpool = nn.Identity()
model = model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)


def evaluate() -> tuple[float, float]:
    model.eval()
    val_loss = 0.0
    correct = 0
    with contextlib.nullcontext() if args.bug == "eval-no-grad" else torch.no_grad():
        for images, labels in test_loader:
            x, y = images.cuda(), labels.cuda()
            out = model(x)
            loss = nn.functional.cross_entropy(out, y)
            val_loss += loss if args.bug == "eval-no-grad" else loss.item()
            correct += (out.argmax(1) == y).sum().item()
    return float(val_loss) / len(test_loader), correct / len(test_set)


for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        x, y = images.cuda(), labels.cuda()
        loss = nn.functional.cross_entropy(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss if args.bug == "train-total-loss" else loss.item()
    val_loss, accuracy = evaluate()
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(
        f"epoch {epoch + 1}: train loss {float(running_loss) / len(train_loader):.3f}, "
        f"val loss {val_loss:.3f}, accuracy {accuracy:.1%}, peak {peak:.1f} GiB"
    )
