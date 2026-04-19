"""
==============================================================================
  Self-Pruning Neural Network  –  Case Study (AI Engineer)
==============================================================================
A feed-forward neural network for CIFAR-10 that **learns to prune itself**
during training via learnable gate parameters and an L1 sparsity penalty.

Deliverables implemented in this single script:
  1. PrunableLinear  – custom linear layer with per-weight sigmoid gates
  2. SelfPruningNet  – multi-layer classifier built from PrunableLinear
  3. Training loop   - cross-entropy + lambda * L1(gates) sparsity loss
  4. Evaluation      – test accuracy & sparsity level reporting
  5. Multi-lambda sweep - compares low / medium / high regularisation
  6. Visualisation   – matplotlib histogram of gate values for the best model
==============================================================================
"""

import os
import sys
import time
import math

# Force unbuffered output (critical on Windows for real-time progress)
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


# ──────────────────────────────────────────────────────────────────────
# Part 1 ▸ The "Prunable" Linear Layer
# ──────────────────────────────────────────────────────────────────────
class PrunableLinear(nn.Module):
    """
    A fully-connected layer whose every weight has an associated learnable
    *gate score*.  During the forward pass the raw gate scores are passed
    through a **sigmoid** to produce values in [0, 1], and each weight is
    element-wise multiplied by its gate.  A gate that converges to 0
    effectively prunes the corresponding weight.

    Parameters
    ----------
    in_features  : int – size of each input sample
    out_features : int – size of each output sample
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight and bias (same shapes as nn.Linear)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.empty(out_features))

        # Gate scores – same shape as weight; learnable via back-prop
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        # Initialise parameters
        self._reset_parameters()

    # -- Kaiming-style init for weights; gates start slightly positive so
    #    sigmoid(gate_score) ~ 0.73 -- keeps most connections alive early on.
    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        # Start gate scores at +1.0 → sigmoid(1) ≈ 0.73
        nn.init.constant_(self.gate_scores, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          1. gates = sigmoid(gate_scores)           – values in (0, 1)
          2. pruned_weights = weight * gates         – element-wise
          3. output = x @ pruned_weights^T + bias    – standard linear op
        """
        gates = torch.sigmoid(self.gate_scores)            # (out, in)
        pruned_weights = self.weight * gates                # (out, in)
        return F.linear(x, pruned_weights, self.bias)       # (batch, out)

    def get_gates(self) -> torch.Tensor:
        """Return the current gate activations (detached, on CPU)."""
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores).cpu()

    def extra_repr(self) -> str:  # noqa
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, bias=True, gated=True")


# ──────────────────────────────────────────────────────────────────────
# Helper ▸ Collect sparsity statistics across the whole model
# ──────────────────────────────────────────────────────────────────────
def collect_gate_stats(model: nn.Module, threshold: float = 1e-2):
    """
    Walk every PrunableLinear in *model* and return:
      - all_gates  : 1-D tensor of every gate value
      - sparsity   : fraction of gates below *threshold*
    """
    gate_list = []
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gate_list.append(m.get_gates().flatten())
    all_gates = torch.cat(gate_list)
    sparsity  = (all_gates < threshold).float().mean().item()
    return all_gates, sparsity


def sparsity_loss(model: nn.Module) -> torch.Tensor:
    """
    L1 norm of all sigmoid-gated values across every PrunableLinear layer.
    Since sigmoid outputs are always >= 0, L1 = simple sum.
    """
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            penalty = penalty + torch.sigmoid(m.gate_scores).sum()
    return penalty


# ──────────────────────────────────────────────────────────────────────
# Part 1 (cont.) ▸ The Network Architecture
# ──────────────────────────────────────────────────────────────────────
class SelfPruningNet(nn.Module):
    """
    Simple feed-forward classifier for CIFAR-10 (32x32x3 images, 10 classes).
    Uses PrunableLinear layers so the network can learn to prune itself.

    Architecture:
        Flatten -> 3072 -> 512 -> 256 -> 128 -> 10
        ReLU activations between hidden layers; no activation on output.
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = PrunableLinear(3 * 32 * 32, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 128)
        self.fc4 = PrunableLinear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)           # raw logits
        return x


# ──────────────────────────────────────────────────────────────────────
# Part 3 ▸ Data Loading (CIFAR-10)
# ──────────────────────────────────────────────────────────────────────
def get_cifar10_loaders(batch_size: int = 128, data_dir: str = "./data"):
    """Download (if needed) and return train / test DataLoaders for CIFAR-10."""

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True,  download=True, transform=transform_train)
    test_set  = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test)

    use_cuda = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=0,
                              pin_memory=use_cuda)
    test_loader  = DataLoader(test_set,  batch_size=batch_size,
                              shuffle=False, num_workers=0,
                              pin_memory=use_cuda)
    return train_loader, test_loader


# ──────────────────────────────────────────────────────────────────────
# Part 3 ▸ Training & Evaluation
# ──────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, lam, device):
    """
    Train for one epoch.
    Total Loss = CrossEntropy + λ × SparsityLoss (L1 of sigmoid gates).
    Returns average total loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)

        cls_loss = F.cross_entropy(logits, labels)
        sp_loss  = sparsity_loss(model)
        total    = cls_loss + lam * sp_loss

        total.backward()
        optimizer.step()

        running_loss += total.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    """Return top-1 accuracy on the given loader (as a percentage)."""
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


def run_experiment(lam: float,
                   epochs: int      = 20,
                   lr: float        = 1e-3,
                   batch_size: int  = 128,
                   device: str      = "auto",
                   verbose: bool    = True):
    """
    Full training run for a single λ value.

    Returns
    -------
    dict with keys: lambda, test_acc, sparsity, model, all_gates
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  lambda = {lam}   |   device = {device}   |   epochs = {epochs}")
    print(f"{'='*60}")

    # Data
    train_loader, test_loader = get_cifar10_loaders(batch_size)

    # Model & Optimiser
    model     = SelfPruningNet().to(device)
    
    # Give gate_scores a much higher learning rate so L1 penalty can effectively push them to 0
    # despite Adam normalizing the gradients.
    gate_params = [p for n, p in model.named_parameters() if 'gate_scores' in n]
    weight_params = [p for n, p in model.named_parameters() if 'gate_scores' not in n]
    
    optimizer = optim.Adam([
        {'params': gate_params, 'lr': 1e-2},
        {'params': weight_params, 'lr': lr}
    ])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    # -- Training loop --
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, lam, device)
        dt   = time.time() - t0

        if verbose and (epoch % 5 == 0 or epoch == 1):
            _, sp = collect_gate_stats(model)
            acc   = evaluate(model, test_loader, device)
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"loss={loss:.4f}  acc={acc:.1f}%  "
                  f"sparsity={sp*100:.1f}%  ({dt:.1f}s)")

        scheduler.step()

    # -- Final evaluation --
    test_acc = evaluate(model, test_loader, device)
    all_gates, sparsity = collect_gate_stats(model)
    print(f"\n  >> Final  |  Test Acc = {test_acc:.2f}%  "
          f"|  Sparsity = {sparsity*100:.2f}%")

    return {
        "lambda":    lam,
        "test_acc":  test_acc,
        "sparsity":  sparsity * 100,
        "model":     model,
        "all_gates": all_gates,
    }


# ──────────────────────────────────────────────────────────────────────
# Part 3 (cont.) ▸ Visualisation – Gate-value histogram
# ──────────────────────────────────────────────────────────────────────
def plot_gate_distribution(all_gates: torch.Tensor,
                           lam: float,
                           save_path: str = "gate_distribution.png"):
    """
    Save a matplotlib histogram of the gate values for the given model.
    A well-pruned network should show a large spike near 0 and a cluster
    of values away from 0.
    """
    vals = all_gates.numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(vals, bins=100, color="#2196F3", edgecolor="black", alpha=0.85)
    ax.set_xlabel("Gate Value (sigmoid output)", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title(f"Distribution of Gate Values  (lambda = {lam})", fontsize=15)
    ax.axvline(x=0.01, color="red", linestyle="--", label="Pruning threshold (0.01)")
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\n  [PLOT] Gate distribution plot saved -> {save_path}")


# ──────────────────────────────────────────────────────────────────────
# Main – Multi-λ sweep
# ──────────────────────────────────────────────────────────────────────
def main():
    # Three lambda values: low, medium, high
    lambda_values = [1e-4, 5e-4, 2e-3]
    results = []

    for lam in lambda_values:
        res = run_experiment(lam=lam, epochs=20)
        results.append(res)

    # ── Summary table ──
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Lambda':<12} {'Test Acc (%)':<16} {'Sparsity (%)':<14}")
    print(f"  {'-'*12} {'-'*16} {'-'*14}")
    for r in results:
        print(f"  {r['lambda']:<12.1e} {r['test_acc']:<16.2f} {r['sparsity']:<14.2f}")
    print("=" * 60)

    # ── Pick best model (highest sparsity among those with acc > baseline-5%)
    best = max(results, key=lambda r: r["sparsity"])
    print(f"\n  Best model: lambda={best['lambda']:.1e}  "
          f"(Acc={best['test_acc']:.2f}%, Sparsity={best['sparsity']:.2f}%)")

    # ── Save gate distribution plot for best model
    plot_gate_distribution(best["all_gates"], best["lambda"],
                           save_path="gate_distribution.png")

    # -- Also save a comparison plot across all lambda values
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    for ax, r in zip(axes, results):
        vals = r["all_gates"].numpy()
        ax.hist(vals, bins=80, color="#FF9800", edgecolor="black", alpha=0.8)
        ax.set_title(f"lambda = {r['lambda']:.1e}\n"
                     f"Acc={r['test_acc']:.1f}%  Sp={r['sparsity']:.1f}%",
                     fontsize=11)
        ax.set_xlabel("Gate Value")
        ax.set_ylabel("Count")
        ax.axvline(0.01, color="red", ls="--", lw=1)
    fig.suptitle("Gate Distribution Comparison Across Lambda Values", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig("gate_distribution_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [PLOT] Comparison plot saved -> gate_distribution_comparison.png")


if __name__ == "__main__":
    main()
