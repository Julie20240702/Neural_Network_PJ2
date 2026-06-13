import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm
import time

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm
from data.loaders import get_cifar_loader

# ## Constants
batch_size = 128
num_workers = 2

# Paths
home_path = os.path.dirname(os.path.abspath(__file__))
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')
os.makedirs(figures_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)

# Device
device = torch.device("cpu")
print(f"Using device: {device}")


def set_random_seeds(seed_value=0):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)


def get_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total


def train_for_landscape(model, optimizer, criterion, train_loader, epochs_n=10):
    """Train and record per-step losses (no accuracy eval for speed)."""
    model.to(device)
    model.train()
    all_step_losses = []

    for epoch in range(epochs_n):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            all_step_losses.append(loss.item())
            loss.backward()
            optimizer.step()

    return all_step_losses


def compute_min_max_curves(all_losses_list):
    """Compute step-wise min and max loss across multiple runs."""
    min_len = min(len(l) for l in all_losses_list)
    aligned = np.array([l[:min_len] for l in all_losses_list])
    min_curve = aligned.min(axis=0)
    max_curve = aligned.max(axis=0)
    return min_curve, max_curve


def plot_loss_landscape(min_vgg, max_vgg, min_bn, max_bn):
    """Plot loss landscape comparison."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    steps_vgg = np.arange(len(min_vgg))
    steps_bn = np.arange(len(min_bn))

    ax.fill_between(steps_vgg, min_vgg, max_vgg,
                    alpha=0.3, color='red', label='VGG-A (without BN)')
    ax.fill_between(steps_bn, min_bn, max_bn,
                    alpha=0.3, color='blue', label='VGG-A (with BN)')

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Landscape: VGG-A vs VGG-A + BatchNorm')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'loss_landscape.png'), dpi=150)
    print(f"Saved: {os.path.join(figures_path, 'loss_landscape.png')}")
    plt.close()


# ===================== MAIN =====================
if __name__ == '__main__':
    # Use subset for CPU speed (full dataset for final report can be run later)
    n_train_items = 2000
    epochs = 10
    learning_rates = [1e-3, 2e-3, 1e-4, 5e-3, 5e-4]

    print(f"Training config: {n_train_items} samples, {epochs} epochs, {len(learning_rates)} LRs")
    print(f"Estimated steps per run: {n_train_items // batch_size * epochs}")

    train_loader = get_cifar_loader(train=True, batch_size=batch_size,
                                     num_workers=0, n_items=n_train_items)
    val_loader = get_cifar_loader(train=False, batch_size=batch_size,
                                   num_workers=0)

    # Verify data
    for X, y in train_loader:
        print(f"Batch shape: {X.shape}, labels: {y[:5].tolist()}")
        break

    criterion = nn.CrossEntropyLoss()

    # --- Train VGG_A ---
    print("\n" + "=" * 50)
    print("Training VGG_A (without BatchNorm)")
    print("=" * 50)
    all_losses_vgg = []
    for lr in learning_rates:
        set_random_seeds(2020)
        model = VGG_A()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        t0 = time.time()
        losses = train_for_landscape(model, optimizer, criterion, train_loader, epochs_n=epochs)
        elapsed = time.time() - t0
        print(f"  lr={lr:.1e} | {len(losses)} steps | {elapsed:.1f}s | final_loss={losses[-1]:.4f}")
        all_losses_vgg.append(losses)

    # --- Train VGG_A_BatchNorm ---
    print("\n" + "=" * 50)
    print("Training VGG_A_BatchNorm")
    print("=" * 50)
    all_losses_bn = []
    for lr in learning_rates:
        set_random_seeds(2020)
        model = VGG_A_BatchNorm()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        t0 = time.time()
        losses = train_for_landscape(model, optimizer, criterion, train_loader, epochs_n=epochs)
        elapsed = time.time() - t0
        print(f"  lr={lr:.1e} | {len(losses)} steps | {elapsed:.1f}s | final_loss={losses[-1]:.4f}")
        all_losses_bn.append(losses)

    # --- Compute and plot ---
    min_vgg, max_vgg = compute_min_max_curves(all_losses_vgg)
    min_bn, max_bn = compute_min_max_curves(all_losses_bn)

    plot_loss_landscape(min_vgg, max_vgg, min_bn, max_bn)

    # Also save raw data for later analysis
    np.savez(os.path.join(figures_path, 'landscape_data.npz'),
             min_vgg=min_vgg, max_vgg=max_vgg,
             min_bn=min_bn, max_bn=max_bn,
             learning_rates=learning_rates)
    print("\nDone! Results saved to reports/figures/")