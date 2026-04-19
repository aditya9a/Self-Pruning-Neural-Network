# 🧠 Self-Pruning Neural Network

A custom PyTorch feed-forward neural network for CIFAR-10 that **learns to prune itself** natively during the training process. 

Normally, deploying large neural networks is constrained by memory. Typical techniques rely on a separate post-training pass to identify and remove weak connections. This repository demonstrates an implementation of dynamic compression: a network that identifies and drops up to **99.9%** of its weak connections dynamically *on the fly*, all driven directly inside the active regularization loop.

---

## ⚙️ Architecture and Logic

The core logic revolves around a custom mathematical layer (`PrunableLinear`) which is used exclusively to construct the network structure.

1. **The Gate Mechanism:** Every standard linear parameter connection is paired with a matching, unconstrained mathematical scalar called a `gate_score`.
2. **Sigmoid Squashing:** During forward propagation, `gate_scores` are bounded between `0` and `1` completely using the Sigmoid transformation. 
3. **Multiplication:** This newly mapped gate multiplies against the actual weight value element-wise.
4. **L1 Penalty:** An L1 Sparsity mathematically forces parameter gradients backwards towards `0`. Cross Entropy struggles to hold up necessary features but abandons weak connections. `lambda` controls the extreme pull of structural deletion.

```text
Input (3×32×32 = 3072)
    ↓
PrunableLinear(3072, 512)  + ReLU     ← 1,572,864 gated weights
    ↓
PrunableLinear(512, 256)   + ReLU     ←   131,072 gated weights
    ↓
PrunableLinear(256, 128)   + ReLU     ←    32,768 gated weights
    ↓
PrunableLinear(128, 10)               ←     1,280 gated weights
    ↓
Output (10 logits)
```

---

## 🚀 How to Run Locally

Since this repo is self-contained with no crazy dependencies inside a single script, running it and generating all experimental proofs is a single step.

### Requirements:
- Python 3.8+
- PyTorch
- Torchvision
- Matplotlib

### Execution:
Clone the repository and run the script. `Torchvision` will automatically download the CIFAR-10 data binaries safely to the `/data` folder without muddying the repo.
```bash
python self_pruning_network.py
```

---

## 📊 Results and the Pruning Tradeoff

Sparsity is the percentage of absolute mathematical connections successfully severed and deleted. By increasing the penalty (`lambda`), we enforce extreme compression constraints resulting in the network severing up to 99.9% of its structure.

| Lambda Penalty (λ) | Final Test Accuracy (%) | Sparsity Level (%) |
|:----------:|:-----------------:|:------------------:|
|   1.0e-04  |       50.14       |       **99.04**    |
|   5.0e-04  |       42.87       |       **99.89**    |
|   2.0e-03  |       34.18       |       **99.99**    |

> **Note:** The results above demonstrate the absolute trade-off. Even when 99.04% of its neural connections were deleted during calculation, the network achieved > 50% accuracy on classifying 10 complex images classes on CPU!

### Gate Severance Plot (λ = 2.0e-03)

If you load the generated visualization inside this repo (`gate_distribution.png`), you can easily visualize exactly how the mathematics forces bounds inward:
- You will see an absolute mathematical spike clustered mathematically straight onto **`0.0`** (the massive cluster of 99.9% of useless neural connections that were successfully severed).
- You will see the incredibly rare distribution surviving cleanly across the higher `1.0` graph.
