"""
Part 1: Custom CNN for CIFAR-10
Design: Lightweight Attention-Residual Network (LARNet)
- Depthwise separable convolutions for parameter efficiency
- Squeeze-and-Excitation (SE) channel attention
- Residual connections for training stability
- Global Average Pooling (no large FC layers)

Target: >70% accuracy, <5M parameters
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import os


# ==================== Model Definition ====================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.squeeze(x).view(b, c)
        w = self.excitation(w).view(b, c, 1, 1)
        return x * w


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: depthwise + pointwise."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                   stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class LARBlock(nn.Module):
    """Lightweight Attention-Residual Block."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = DepthwiseSeparableConv(in_channels, out_channels, stride=stride)
        self.se = SEBlock(out_channels)

        # Residual connection (with projection if dimensions change)
        self.use_residual = True
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv(x)
        out = self.se(out)
        out = out + residual
        out = self.relu(out)
        return out


class LARNet(nn.Module):
    """
    Lightweight Attention-Residual Network (LARNet)
    Custom CNN for CIFAR-10 classification.

    Architecture highlights:
    1. Depthwise separable convolutions reduce params by ~8-9x vs standard conv
    2. SE attention allows the network to focus on informative channels
    3. Residual connections enable stable training of deeper networks
    4. Global average pooling eliminates large FC layers
    """
    def __init__(self, num_classes=10):
        super().__init__()

        # Stem: standard conv to expand channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Stage 1: 32x32 spatial
        self.stage1 = nn.Sequential(
            LARBlock(32, 64, stride=1),
            LARBlock(64, 64, stride=1),
        )

        # Stage 2: 16x16 spatial (stride=2 in first block)
        self.stage2 = nn.Sequential(
            LARBlock(64, 128, stride=2),
            LARBlock(128, 128, stride=1),
            LARBlock(128, 128, stride=1),
        )

        # Stage 3: 8x8 spatial
        self.stage3 = nn.Sequential(
            LARBlock(128, 256, stride=2),
            LARBlock(256, 256, stride=1),
            LARBlock(256, 256, stride=1),
        )

        # Stage 4: 4x4 spatial
        self.stage4 = nn.Sequential(
            LARBlock(256, 512, stride=2),
            LARBlock(512, 512, stride=1),
        )

        # Head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# ==================== Training ====================

def get_data_loaders(batch_size=128):
    """CIFAR-10 data with standard augmentation."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Use the local CIFAR data from VGG_BatchNorm
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'VGG_BatchNorm', 'data')

    trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    return trainloader, testloader


def evaluate(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total


def train_model(epochs=40, batch_size=128, lr=0.01):
    device = torch.device('cpu')
    print(f"Device: {device}")

    trainloader, testloader = get_data_loaders(batch_size)
    print(f"Training samples: {len(trainloader.dataset)}")
    print(f"Test samples: {len(testloader.dataset)}")

    model = LARNet(num_classes=10).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,}")
    assert total_params < 5_000_000, f"Too many parameters: {total_params}"

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    train_losses = []
    test_accs = []
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        t0 = time.time()

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / len(trainloader)
        train_losses.append(avg_loss)

        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            acc = evaluate(model, testloader, device)
            test_accs.append((epoch + 1, acc))
            elapsed = time.time() - t0
            print(f"Epoch [{epoch+1}/{epochs}] loss={avg_loss:.4f} acc={acc:.2f}% "
                  f"lr={scheduler.get_last_lr()[0]:.6f} time={elapsed:.1f}s")
            if acc > best_acc:
                best_acc = acc
        else:
            elapsed = time.time() - t0
            print(f"Epoch [{epoch+1}/{epochs}] loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.6f} time={elapsed:.1f}s")

    print(f"\nBest test accuracy: {best_acc:.2f}%")

    # Save results
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_part1')
    os.makedirs(results_dir, exist_ok=True)

    # Plot training curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(train_losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('LARNet Training Loss')
    ax1.grid(True, alpha=0.3)

    epochs_eval, accs = zip(*test_accs)
    ax2.plot(epochs_eval, accs, 'o-')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('LARNet Test Accuracy')
    ax2.axhline(y=70, color='r', linestyle='--', label='70% target')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_curves.png'), dpi=150)
    print(f"Saved training curves to {results_dir}/training_curves.png")

    # Save model
    torch.save(model.state_dict(), os.path.join(results_dir, 'larnet_best.pth'))
    print(f"Saved model to {results_dir}/larnet_best.pth")

    return model, best_acc


if __name__ == '__main__':
    model, acc = train_model(epochs=20, batch_size=128, lr=0.05)