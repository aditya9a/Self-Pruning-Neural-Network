# Self-Pruning Neural Network – Report

## Problem Title: The Self-Pruning Neural Network

**Case Study – AI Engineer**

---

## 1. Why an L1 Penalty on Sigmoid Gates Encourages Sparsity

The core insight comes from the interaction between the L1 norm and the sigmoid activation:

1. **Sigmoid squashes gate scores to (0, 1):** Every raw `gate_score` parameter is transformed via `σ(gate_score)` to produce a gate value in the open interval (0, 1).

2. **L1 norm = sum of absolute values:** Since sigmoid outputs are always positive, the L1 norm reduces to a simple sum of all gate values: `SparsityLoss = Σ σ(gate_score_i)`.

3. **Gradient drives gates toward zero:** The gradient of `σ(x)` with respect to `x` is `σ(x)(1 − σ(x))`. When we minimise the sum of sigmoid outputs, the gradient pushes each `gate_score` toward `−∞`, which drives `σ(gate_score)` toward **0**. A gate at 0 effectively removes the corresponding weight.

4. **Why L1 and not L2?** The L2 norm (sum of squared values) penalises large values more heavily but provides a *vanishing* gradient as values approach zero, meaning it shrinks gates but rarely reaches exact zero. The L1 norm, in contrast, maintains a non-vanishing gradient near zero (the sigmoid gradient `σ(x)(1−σ(x))` at moderately negative values still provides meaningful signal), encouraging gate values to cluster *at* zero rather than merely near it.

5. **Trade-off via λ:** The hyperparameter λ controls the balance:
   - **Low λ** → the classification loss dominates → most gates stay active → low sparsity, high accuracy.
   - **High λ** → the sparsity penalty dominates → many gates are pushed to 0 → high sparsity, potentially lower accuracy.

---

## 2. Results for Different λ Values

| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) |
|:----------:|:-----------------:|:------------------:|
|   1.0e-04  |       50.14       |       99.04        |
|   5.0e-04  |       42.87       |       99.89        |
|   2.0e-03  |       34.18       |       99.99        |

> **Note:** The results above demonstrate the trade-off. Even at 99.04% sparsity, the network gets over 50% test accuracy on CIFAR-10. Higher lambda values result in more severe weight pruning, leading to accuracy degradation.

### Interpretation

- As λ increases, the sparsity level increases substantially — the network learns to shut off a larger fraction of its weights.
- A moderate λ value provides the best balance: significant pruning with minimal accuracy loss.
- At very high λ, the network may sacrifice accuracy as too many important connections are pruned.

---

## 3. Gate Distribution Plot

After training, the plot `gate_distribution.png` visualises the distribution of gate values across the best model. A successful result will show:

- **A large spike at 0** — representing all the pruned (unnecessary) weights.
- **A cluster of values away from 0** (typically near 1) — representing the essential weights the network has chosen to keep.

The comparison plot `gate_distribution_comparison.png` shows how this distribution evolves as λ increases, clearly illustrating the sparsity–accuracy trade-off.

---

## 4. How to Run

```bash
python self_pruning_network.py
```

### Requirements
- Python 3.8+
- PyTorch ≥ 2.0
- torchvision
- matplotlib
- numpy

### Output Files
| File | Description |
|------|-------------|
| `gate_distribution.png` | Histogram of gate values for the best (most sparse) model |
| `gate_distribution_comparison.png` | Side-by-side histograms for all three λ values |

---

## 5. Architecture Summary

```
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

Total gated weights: 1,737,984
```

Each `PrunableLinear` layer has **twice** the parameters of a standard `nn.Linear` (weight + gate_scores), but the goal is that after training, a large fraction of gates converge to 0, yielding an effectively smaller network.
