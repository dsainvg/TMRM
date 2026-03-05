# Encoders

## Overview

The encoder subsystem is the entry point of the network. It takes **N** single-channel `(1, n, n)` input matrices and transforms each one independently into a `(64, n, n)` feature representation. All N outputs are concatenated channel-wise to produce a `(64·N, n, n)` tensor that feeds the first decoder layer.

---

## Encoder Node

**File:** `utils/encoder.py`

A single `Encoder` node maps `(in_channels, n, n) → (out_channels, n, n)` using two 1×1 convolutions and a pairwise matrix product expansion.

### Internal Pipeline

```
(in_c, n, n)
     │
     ▼
 conv1 (1×1)                  in_c → 4 channels
     │
     ▼
 (4, n, n)  ─────────────────────────────────────────┐
     │                                               │
     ▼  Pairwise matmul                              │
 C(4,2) = 6 pairs  →  batched matmul  →  (6, n, n)  │
     │                                               │
     └──────── concat(4 original, 6 pairs) ──────────┘
                          │
                          ▼
                    (10, n, n)
                          │
                          ▼
                 conv2 (1×1)             10 → out_c channels
                          │
                          ▼
                   (out_c, n, n)
                          │
                          ▼
                    tanh  (element-wise, parameter-free)
                          │
                          ▼
                   (out_c, n, n)   ← stage output
```

The 4 channels from `conv1` are combined as all $\binom{4}{2} = 6$ ordered matrix products via batched `matmul`. These 6 interaction maps are concatenated with the original 4 channel maps, giving a 10-channel intermediate tensor. `conv2` projects this down to the desired `out_channels`, and `tanh` is applied element-wise at the end of each stage to bound the output before it enters the next stage's matmuls.

Inactive nodes short-circuit via `jax.lax.cond`, returning a pre-allocated zero tensor and propagating `is_active = False` downstream without executing any computation.

---

## EncoderLayer

**File:** `utils/encoder_layer.py`

Each input has a **1-to-8 branching tree** of **9 `Encoder` nodes total**:

| Stage | Nodes per input | Input per node | Output per node | `conv1` | `conv2` |
|-------|:---------------:|---------------|----------------|---------|--------|
| 1 — root   | 1 | `(1, n, n)` | `(8, n, n)` | 1 → 4 | 10 → 8 |
| 2 — leaves | 8 | `(1, n, n)` | `(8, n, n)` | 1 → 4 | 10 → 8 |

The stage-1 root outputs 8 channels; each channel is peeled off and fed to its own dedicated stage-2 leaf encoder. The 8 leaf outputs `(8, 8, n, n)` are flattened to `(64, n, n)` per input. All N stacks are double-vmapped in parallel:

```
Input:   (N, 1, n, n)  +  is_active: (N,)
              │
    ┌─────────┴─────────────────────────────────────────────┐
    │  Stage 1 (root)  Encoder(in=1, out=8)  → (8, n, n)   │  × N
    │       │  split into 8 × (1, n, n) channels            │
    │       ├── Leaf 0: Encoder(in=1, out=8) → (8, n, n)   │
    │       ├── Leaf 1: Encoder(in=1, out=8) → (8, n, n)   │
    │       │   ...  (vmapped over 8 leaves)                │
    │       └── Leaf 7: Encoder(in=1, out=8) → (8, n, n)   │
    │       concat → (64, n, n),  flags → (64,)             │
    └─────────┬─────────────────────────────────────────────┘
              │
         (N, 64, n, n)    reshape  →  (64·N, n, n)
         (N, 64)  flags   reshape  →  (64·N,)
```

---

## Parameter Count

Both convolutions are 1×1 with bias. For a Conv2d of shape `(out, in, 1, 1)`:

$$\text{params} = \text{out} \times \text{in} + \text{out}$$

### Single Encoder Node  `(in_c = 1, out_c = 8)`

Both stage-1 and each stage-2 leaf use identical configuration `(in_c=1, out_c=8)`:

| Layer | Weight | Bias | Subtotal |
|-------|--------|------|----------|
| `conv1`: 1 → 4  | $4 \times 1 = 4$ | $4$ | **8** |
| `conv2`: 10 → 8 | $10 \times 8 = 80$ | $8$ | **88** |
| **Node total** | | | **96** |

### Per Input Stack (1 root + 8 leaves)

| Component | Nodes | Params each | Subtotal |
|-----------|:-----:|:-----------:|:--------:|
| Stage-1 root | 1 | 96 | **96** |
| Stage-2 leaves | 8 | 96 | **768** |
| **Stack total** | **9** | | **864** |

$$1 \times 96 + 8 \times 96 = 9 \times 96 = \mathbf{864 \text{ parameters}}$$

### Total EncoderLayer Parameters for N Inputs

$$\boxed{864 \times N}$$

| N (inputs) | Encoder nodes | Total parameters |
|:----------:|:-------------:|:----------------:|
| 1  | 9   | 864    |
| 4  | 36  | 3,456  |
| 8  | 72  | 6,912  |
| 16 | 144 | 13,824 |
| 32 | 288 | 27,648 |
| 64 | 576 | 55,296 |

---

## Configuration Constants

Defined in `utils/config/encode.py` and `utils/config/encoder_layer.py`:

| Constant | Value | Role |
|----------|------:|------|
| `ENCODER_IN_CHANNELS` | 1 | Input channels to a stage-1 encoder |
| `ENCODER_EXPAND_CHANNELS` | 4 | `conv1` output channels (pairwise expansion base) |
| `ENCODER_INTERMEDIATE_CHANNELS` | 10 | `conv2` input channels (4 original + 6 pairs) |
| `ENCODER_OUT_CHANNELS` | 8 | Default `conv2` output channels |
| `ENCODER_INTER_STACK_CHANNELS` | 8 | Stage-1 output channels (= number of stage-2 leaves) |
| `ENCODER_STAGE2_COUNT` | 8 | Number of stage-2 leaf encoder nodes per input |
| `ENCODER_STACK_OUT_CHANNELS` | 64 | Total output channels per input (`ENCODER_STAGE2_COUNT × ENCODER_OUT_CHANNELS`) |
| `ENCODER_STACK_DEPTH` | 2 | Tree depth (root + leaf stages) |
