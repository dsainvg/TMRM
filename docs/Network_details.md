
# Comprehensive Architecture Specification: Dynamic DAG Neural Network

## High-Level Architectural Philosophy

This network is a highly dynamic, sparsely activated Directed Acyclic Graph (DAG). It functions similarly to a Mixture of Experts (MoE) or modular adapter network, but with a custom routing and feature-interaction mechanism.

The primary design principle is **Static Topology, Dynamic Execution**. Because the model will be compiled for accelerators (GPUs/TPUs) using JAX/XLA, the maximal graph topology must be fixed at compile time. The network achieves run-time dynamism via boolean masking and conditional execution primitives rather than altering graph shape.

## Component-Level Breakdown

The network is composed of two fundamental building blocks: **Encoders** and **Decoders**. Both modules expand features via pairwise 2D matrix multiplications rather than standard deep convolutional channel mixing.

### Encoder Module (The Adapter)

- **Purpose:** Adapters for specific input types. Responsible for initial feature extraction and dimensionality manipulation. Encoders do not mix information across input contexts.
- **Activation rule:** An Encoder node is *active* if its input is active; otherwise it short-circuits via `jax.lax.cond` and returns a zero tensor plus `is_active = False`.

#### Single Encoder node — mathematical flow (when active)

All shapes use **CHW** format `(channels, n, n)`, matching JAX/Equinox Conv2d convention.

- **Input:** `(in_c, n, n)` — one or more channel(s) of an $n \times n$ matrix.
- **conv1** (1×1): expand to 4 channels → `(4, n, n)`.
- **Pairwise interaction:** compute matrix products for every unique pair of the 4 channel-matrices.
  - Number of unique pairs: $\binom{4}{2} = 6$.
  - Pre-computed static index pairs; executed as a single batched `matmul` → `(6, n, n)`.
- **Concatenation:** `[4 original channels, 6 pair products]` → `(10, n, n)`.
- **conv2** (1×1): project 10 → `out_c` channels → `(out_c, n, n)`.
- **Output:** `(out_c, n, n)` with `is_active = True`.

#### EncoderLayer — tree structure (1 root + 8 leaves per input)

Two stages are stacked in a **1-to-8 branching tree** — 9 Encoder nodes per input:

| Stage | Nodes | `in_c` | `out_c` | Role |
|-------|:-----:|--------|---------|------|
| 1 — root   | 1 | 1 | 8 | Initial feature expansion |
| 2 — leaves | 8 | 1 | 8 | Each receives one channel from root output |

The root output `(8, n, n)` is split into 8 single-channel tensors; each is processed independently by its own leaf encoder. The 8 leaf outputs `(8, n, n)` each are concatenated → `(64, n, n)` per input.

For **N** inputs the full EncoderLayer maps:
```
(N, 1, n, n)  +  is_active: (N,)   →   (64·N, n, n)  +  flags: (64·N,)
```
All N trees are vmapped in parallel. Each input's root flag propagates to all 8 of its leaves.

### Decoder Module (The Cross-Context Builder)

- **Purpose:** Core hidden units that filter, mix, and route signals across contexts.
- **Activation rule (threshold gate):** Each Decoder accepts exactly 16 parent inputs and activates only if at least **12/16** parents have `is_active = True`. Otherwise it returns a zero tensor and `is_active = False`.

#### Mathematical flow (when active)

All shapes use **CHW** format `(channels, n, n)`.

- **Input:** `(16, n, n)` — 16 parent matrices, each `(n, n)`, plus `(16,)` activity flags.
- **Threshold gate:** `sum(is_active_flags) >= 12`; if not met → inactive path.
- **Determinant scoring:** compute `log|det|` per matrix via `slogdet` → `(16,)` scores.  
  Inactive parents are masked to `-inf` so they cannot enter the top-K.
- **Top-12 selection:** `top_k(scores, 12)` → `(12, n, n)`.
- **Split:**
  - `top_interact = top_x[:8]` → `(8, n, n)` — fed into pairwise mixing.
  - `top_preserve = top_x[8:]` → `(4, n, n)` — passed through raw.
- **Pairwise interaction:** all unique pairs of the 8 interact matrices.
  - Number of pairs: $\binom{8}{2} = 28$.
  - Static index pairs; single batched `matmul` → `(28, n, n)`.
- **Concatenation:** `[28 pair products, 4 preserved]` → `(32, n, n)`.
- **conv1** (1×1): `32 → 4` channels → `(4, n, n)`.
- **conv2** (1×1): `4 → 1` channel → `(1, n, n)`.
- **Swish activation** (element-wise, parameter-free): `out = out * sigmoid(out)`.
- **Squeeze:** `(1, n, n)` → `(n, n)`.
- **Output:** `(n, n)` with `is_active = True`.

### FCLayer Module (The Terminal Projection)

- **Purpose:** Final linear projection stage. Maps the flattened, concatenated `DecoderCluster` output to task-specific dimensionality (e.g. class logits, coordinate vectors, policy distributions).
- **Activation rule:** Always active — no `is_active` flag. The activation function (`relu` by default) is configured at construction time and stored as a static field.
- **Call signature:** `(in_features,) → (out_features,)` — input must be a 1-D vector.
- **Trainable parameters:** `in_features × out_features + out_features` (weight matrix + bias of `eqx.nn.Linear`).
- **Supported activations:** `relu` (default), `gelu`, `tanh`, `sigmoid`, `identity`. Unknown names raise `ValueError` at init.
- **Config constant:** `FC_DEFAULT_ACTIVATION = 'relu'` in `utils/config/fc.py`.

#### Connecting DecoderCluster output to FCLayer

```python
dec_out, _ = cluster(encoder_out, encoder_flags)
# dec_out: (total_output_nodes, n, n)
flat   = dec_out.reshape(-1)     # (total_output_nodes × n²,)
logits = fc_layer(flat)          # (out_features,)
```

## Graph Topology and Routing Mechanics

The network uses a randomized DAG topology (no cycles). Connections are sampled once at initialization and remain fixed during training and inference.

### Initialization and connectivity rules

- **Forward-only routing:** all edges point to deeper nodes; cycles are forbidden.
- **Gaussian fan-out (first half):** for nodes in the first half of the network, downstream fan-out is sampled from a clipped Gaussian (mean 18, clipped to [8, 24]).
- **Gaussian fan-out (second half):** deeper nodes use a narrower distribution (mean 12, clipped to [6, 20]).
- **Dynamic layer sizing:** the number of Decoder nodes instantiated at layer $L$ is determined by the sum of the outgoing connection counts from layer $L-1$.

### Terminal routing and final output

- **Gathering:** collect outputs from the final Decoder layer and any produced-but-unconsumed Decoder outputs from previous layers.
- **Concatenation:** concatenate along channel depth.
- **Final adaptation:** pass the concatenated block through 1–2 fully connected (dense) layers.
- **Input-specific outputs:** use different final dense heads depending on the input combination to produce the required output format.

## JAX / XLA Framework Guidelines & Optimizations

To run efficiently under JAX/XLA on accelerators, follow these constraints and patterns.

### 1. Static graph mandate

XLA requires shapes and the maximal execution graph to be known at compile time. Do not rely on Python lists or runtime graph mutation; instead, bake topology into static adjacency arrays and pad unused slots with zeros.

### 2. Handling conditional sparsity (`is_active`)

Avoid plain Python conditionals that force both branches to be traced. Use JAX conditional primitives (e.g., `jax.lax.cond`, `jax.lax.select`) and pass `is_active` masks so inactive nodes return pre-allocated zero tensors without performing heavy ops.

### 3. Vectorization of pairwise combinations

Pre-compute index pairs for all combinations and gather left/right matrices into batched tensors so the $\binom{4}{2} = 6$ (encoder) and $\binom{8}{2} = 28$ (decoder) products run as a single batched matrix-multiply operation.

### 4. Numerical stability for determinant-based ranking

Do not compute raw determinants for ranking. Instead compute the sign and log-absolute-determinant (e.g., via `jax.numpy.linalg.slogdet`) and rank by the log-absolute value. This yields the same Top-12 ordering while remaining numerically stable for large $n$.

### 5. Top-K sorting

Use JAX's `top_k` primitives to select the Top-12 scores efficiently. Assign inactive parents a score of `-inf` so they automatically fall to the bottom.

### 6. Gradient barrier between Decoder and Encoder

The `jax.lax.cond` threshold gate inside each Decoder node, combined with the non-differentiable `slogdet`-ranked `top_k` selection, creates a **gradient barrier**: backpropagation from a task loss flows through the `FCLayer` and into the `DecoderCluster` conv weights, but does **not** continue back through the Decoder gate into the `EncoderLayer` weights.

As a consequence, the two stages must be trained with separate losses:
- **Encoder loss** — applied directly to `EncoderLayer` outputs to train `conv1`/`conv2` encoder weights.
- **Task loss** — applied to `FCLayer` outputs to train `FCLayer` linear weights and all Decoder `conv` weights.

Do not attempt to train encoders end-to-end via a single task loss through the cluster; the gradient will be exactly zero at the encoder weights.

### 7. Swish activation in Decoder

After `conv2`, each active Decoder applies a **Swish** activation ($x \cdot \sigma(x)$) element-wise before returning. This is parameter-free and fused inside the `jax.lax.cond` active branch — inactive nodes still return plain zeros.

---

This document focuses on clear module responsibilities, static topology constraints required by XLA, and the efficient, numerically stable primitives necessary to implement the described dynamic DAG in JAX.
