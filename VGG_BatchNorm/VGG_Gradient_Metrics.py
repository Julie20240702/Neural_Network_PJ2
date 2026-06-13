import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import os
import sys
import time

from models.vgg import VGG_A, VGG_A_BatchNorm
from data.loaders import get_cifar_loader

# Config
n_items = 2000
batch_size = 128
num_epochs = 10
learning_rates = [1e-3, 2e-3, 1e-4, 5e-3]

home_path = os.path.dirname(os.path.abspath(__file__))
figures_path = os.path.join(home_path, 'reports', 'figures')
os.makedirs(figures_path, exist_ok=True)

device = torch.device("cpu")
print(f"Using device: {device}", flush=True)


def train_with_gradient_tracking(model, train_loader, lr, num_epochs, device):
    """Train model and track gradient norms and gradient changes at each step."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    grad_norms = []       # ||∇L(θ_t)||
    grad_changes = []     # ||∇L(θ_t) - ∇L(θ_{t-1})|| / ||∇L(θ_{t-1})||
    losses = []
    
    prev_grad = None
    
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Collect current gradient as flat vector
            current_grad = []
            for p in model.parameters():
                if p.grad is not None:
                    current_grad.append(p.grad.detach().clone().flatten())
            current_grad = torch.cat(current_grad)
            
            # Record gradient norm
            grad_norm = current_grad.norm().item()
            grad_norms.append(grad_norm)
            losses.append(loss.item())
            
            # Record gradient change (predictiveness)
            if prev_grad is not None:
                change = (current_grad - prev_grad).norm().item()
                prev_norm = prev_grad.norm().item()
                if prev_norm > 1e-8:
                    grad_changes.append(change / prev_norm)
                else:
                    grad_changes.append(0.0)
            else:
                grad_changes.append(0.0)
            
            prev_grad = current_grad.clone()
            
            optimizer.step()
    
    return losses, grad_norms, grad_changes


def compute_beta_smoothness(model, train_loader, device, num_steps=50):
    """
    Estimate β-smoothness: ||∇L(θ + η·d) - ∇L(θ)|| / (η·||d||)
    at several points during a short training run.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    beta_values = []
    eta = 0.01  # step size for probing
    
    step_count = 0
    for data, target in train_loader:
        if step_count >= num_steps:
            break
        data, target = data.to(device), target.to(device)
        
        # Compute gradient at current θ
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        grad_at_theta = []
        for p in model.parameters():
            if p.grad is not None:
                grad_at_theta.append(p.grad.detach().clone())
        
        # d = gradient direction (unit vector not needed, we normalize below)
        d_flat = torch.cat([g.flatten() for g in grad_at_theta])
        d_norm = d_flat.norm().item()
        if d_norm < 1e-8:
            step_count += 1
            optimizer.step()
            continue
        
        # Move parameters: θ' = θ + η * d
        with torch.no_grad():
            idx = 0
            for p in model.parameters():
                if p.grad is not None:
                    numel = p.numel()
                    p.data.add_(eta * grad_at_theta[idx])
                    idx += 1
        
        # Compute gradient at θ'
        optimizer.zero_grad()
        output2 = model(data)
        loss2 = criterion(output2, target)
        loss2.backward()
        
        grad_at_theta_prime = []
        for p in model.parameters():
            if p.grad is not None:
                grad_at_theta_prime.append(p.grad.detach().clone())
        
        # Compute ||∇L(θ') - ∇L(θ)|| / (η * ||d||)
        diff = torch.cat([(g2 - g1).flatten() for g1, g2 in zip(grad_at_theta, grad_at_theta_prime)])
        beta = diff.norm().item() / (eta * d_norm)
        beta_values.append(beta)
        
        # Move parameters back: θ = θ' - η * d, then do actual step
        with torch.no_grad():
            idx = 0
            for p in model.parameters():
                if p.grad is not None:
                    p.data.sub_(eta * grad_at_theta[idx])
                    idx += 1
        
        optimizer.step()
        step_count += 1
    
    return beta_values


def run_all_metrics():
    print("Loading data...", flush=True)
    train_loader = get_cifar_loader(root=os.path.join(home_path, 'data'), 
                                     n_items=n_items, batch_size=batch_size, num_workers=0)
    
    # === Part A: Gradient predictiveness (train with multiple LRs) ===
    print(f"\nTraining with gradient tracking: {len(learning_rates)} LRs x 2 models", flush=True)
    
    all_grad_changes_vgg = []
    all_grad_changes_bn = []
    all_grad_norms_vgg = []
    all_grad_norms_bn = []
    
    for lr in learning_rates:
        # VGG_A
        print(f"  VGG_A lr={lr:.1e}...", end='', flush=True)
        t0 = time.time()
        model_vgg = VGG_A().to(device)
        _, norms_v, changes_v = train_with_gradient_tracking(model_vgg, train_loader, lr, num_epochs, device)
        all_grad_changes_vgg.append(changes_v)
        all_grad_norms_vgg.append(norms_v)
        print(f" {time.time()-t0:.0f}s", flush=True)
        del model_vgg
        
        # VGG_A_BN
        print(f"  VGG_BN lr={lr:.1e}...", end='', flush=True)
        t0 = time.time()
        model_bn = VGG_A_BatchNorm().to(device)
        _, norms_b, changes_b = train_with_gradient_tracking(model_bn, train_loader, lr, num_epochs, device)
        all_grad_changes_bn.append(changes_b)
        all_grad_norms_bn.append(norms_b)
        print(f" {time.time()-t0:.0f}s", flush=True)
        del model_bn
    
    # Compute min/max curves for gradient changes
    min_len = min(min(len(x) for x in all_grad_changes_vgg), min(len(x) for x in all_grad_changes_bn))
    
    changes_vgg_arr = np.array([x[:min_len] for x in all_grad_changes_vgg])
    changes_bn_arr = np.array([x[:min_len] for x in all_grad_changes_bn])
    norms_vgg_arr = np.array([x[:min_len] for x in all_grad_norms_vgg])
    norms_bn_arr = np.array([x[:min_len] for x in all_grad_norms_bn])
    
    # === Plot 1: Gradient Predictiveness ===
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    steps = np.arange(min_len)
    
    min_changes_vgg = changes_vgg_arr.min(axis=0)
    max_changes_vgg = changes_vgg_arr.max(axis=0)
    min_changes_bn = changes_bn_arr.min(axis=0)
    max_changes_bn = changes_bn_arr.max(axis=0)
    
    ax.fill_between(steps, min_changes_vgg, max_changes_vgg, alpha=0.3, color='red', label='VGG_A (without BN)')
    ax.fill_between(steps, min_changes_bn, max_changes_bn, alpha=0.3, color='blue', label='VGG_A_BN')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Gradient Change ||∇L(θ_t) - ∇L(θ_{t-1})|| / ||∇L(θ_{t-1})||')
    ax.set_title('Gradient Predictiveness: VGG_A vs VGG_A with BatchNorm')
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'gradient_predictiveness.png'), dpi=150)
    plt.close()
    print(f"Saved: gradient_predictiveness.png", flush=True)
    
    # === Part B: β-smoothness ===
    print("\nComputing β-smoothness...", flush=True)
    
    # Use fresh models, run beta measurement over training batches
    model_vgg = VGG_A().to(device)
    model_bn = VGG_A_BatchNorm().to(device)
    
    print("  VGG_A...", end='', flush=True)
    t0 = time.time()
    beta_vgg = compute_beta_smoothness(model_vgg, train_loader, device, num_steps=min_len)
    print(f" {time.time()-t0:.0f}s", flush=True)
    
    print("  VGG_BN...", end='', flush=True)
    t0 = time.time()
    beta_bn = compute_beta_smoothness(model_bn, train_loader, device, num_steps=min_len)
    print(f" {time.time()-t0:.0f}s", flush=True)
    
    # === Plot 2: β-smoothness ===
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    steps_beta = np.arange(len(beta_vgg))
    ax.plot(steps_beta, beta_vgg, alpha=0.7, color='red', label='VGG_A (without BN)')
    steps_beta_bn = np.arange(len(beta_bn))
    ax.plot(steps_beta_bn, beta_bn, alpha=0.7, color='blue', label='VGG_A_BN')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Effective β-smoothness ||∇L(θ+ηd) - ∇L(θ)|| / (η||d||)')
    ax.set_title('β-Smoothness (Gradient Lipschitz Constant): VGG_A vs VGG_A_BN')
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'beta_smoothness.png'), dpi=150)
    plt.close()
    print(f"Saved: beta_smoothness.png", flush=True)
    
    # Save all data
    np.savez(os.path.join(figures_path, 'gradient_metrics.npz'),
             changes_vgg=changes_vgg_arr, changes_bn=changes_bn_arr,
             norms_vgg=norms_vgg_arr, norms_bn=norms_bn_arr,
             beta_vgg=np.array(beta_vgg), beta_bn=np.array(beta_bn),
             learning_rates=learning_rates)
    print("\nAll gradient metrics complete!", flush=True)


if __name__ == '__main__':
    run_all_metrics()