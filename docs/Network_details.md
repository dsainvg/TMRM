
# Comprehensive Architecture Specification: Dynamic DAG Neural Network

## High-Level Architectural Philosophy

This network is a highly dynamic, sparsely activated Directed Acyclic Graph (DAG). It functions similarly to a Mixture of Experts (MoE) or modular adapter network, but with a custom routing and feature-interaction mechanism.

The primary design principle is **Static Topology, Dynamic Execution**. Because the model will be compiled for accelerators (GPUs/TPUs) using JAX/XLA, the maximal graph topology must be fixed at compile time. The network achieves run-time dynamism via boolean masking and conditional execution primitives rather than altering graph shape.

## Component-Level Breakdown

The network is composed of two fundamental building blocks: **Encoders** and **Decoders**. Both modules expand features via pairwise 2D matrix multiplications rather than standard deep convolutional channel mixing.

### Encoder Module (The Adapter)

- **Purpose:** Adapters for specific input types. Responsible for initial feature extraction and dimensionality manipulation. Encoders do not mix information across input contexts.
- **Activation rule:** An Encoder is *active* if its input type is present; otherwise it is *inactive* and returns a zero tensor plus `is_active = False`.

#### Mathematical flow (when active)

- **Input:** tensor of shape $(n, n, 8)$ (an $n \times n$ grid with 8 channels).
- **Dimensionality reduction:** a $1 \times 1$ convolution to 4 channels → $(n, n, 4)$.
- **Pairwise interaction:** split the 4 channels into four $n \times n$ matrices and compute matrix products for every unique pair.
	- Number of unique pairs: $\binom{4}{2} = 6$.
	- Produces 6 new $(n, n)$ matrices.
- **Concatenation:** concatenate the original 4-channel tensor with the 6 new matrices → output shape $(n, n, 10)$.
- **Output:** tensor $(n, n, 10)$ with `is_active = True`.

### Decoder Module (The Cross-Context Builder)

- **Purpose:** Core hidden units that filter, mix, and route signals across contexts.
- **Activation rule (threshold gate):** Each Decoder accepts up to 16 parents and activates only if at least 12/16 parents have `is_active = True`. Otherwise it returns a zero tensor and `is_active = False`.

#### Mathematical flow (when active)

- **Input:** up to 16 matrices, each $(n, n)$.
- **Determinant filter (Top-8 selection):** compute a stable score per matrix (see §4.4). Rank by absolute score and select the top 8 matrices → tensor $(n, n, 8)$.
- **Pairwise interaction:** compute all unique pairwise matrix products of these 8 matrices.
	- Number of pairs: $\binom{8}{2} = 28$.
	- Produces 28 new $(n, n)$ matrices.
- **Concatenation:** concatenate the 8 selected parents with the 28 new matrices → $(n, n, 36)$.
- **Dimensionality reduction:** $1 \times 1$ convolution down to 4 channels → output $(n, n, 4)$.
- **Output:** tensor $(n, n, 4)$ with `is_active = True`.

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

Pre-compute index pairs for all combinations and gather left/right matrices into batched tensors so the $\binom{4}{2}$ and $\binom{8}{2}$ products run as a single batched matrix-multiply operation.

### 4. Numerical stability for determinant-based ranking

Do not compute raw determinants for ranking. Instead compute the sign and log-absolute-determinant (e.g., via `jax.numpy.linalg.slogdet`) and rank by the log-absolute value. This yields the same Top-8 ordering while remaining numerically stable for large $n$.

### 5. Top-K sorting

Use JAX's `top_k` primitives (or similar) to select the Top-8 scores efficiently. Assign inactive nodes a score of `-inf` so they automatically fall to the bottom.

---

This document focuses on clear module responsibilities, static topology constraints required by XLA, and the efficient, numerically stable primitives necessary to implement the described dynamic DAG in JAX.
