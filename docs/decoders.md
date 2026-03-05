# Decoder Stack — Implementation Reference

This document covers all three decoding components:
`Decoder` (single node) → `DecoderLayer` (K parallel nodes) → `DecoderCluster` (full stacked graph).

---

## 1. `Decoder` — The Single Node (`utils/decoder.py`)

The atomic computation unit. Takes up to 16 input matrices and produces exactly one output matrix.

### What it holds

| Attribute | Type | Shape | Purpose |
|-----------|------|-------|---------|
| `conv1` | `eqx.nn.Conv2d` | weights `(4, 32, 1, 1)`, bias `(4,)` | Collapse 32 mixed channels → 4 |
| `conv2` | `eqx.nn.Conv2d` | weights `(1, 4, 1, 1)`, bias `(1,)` | Project 4 channels → 1 output |
| tanh | `jnp.tanh` | — (parameter-free) | Element-wise activation after `conv2` |
### Trainable parameters per node

| Layer | Weights | Bias | Subtotal |
|-------|---------|------|----------|
| `conv1` (32 → 4, kernel 1×1) | 32 × 4 = **128** | 4 | **132** |
| `conv2` (4 → 1, kernel 1×1) | 4 × 1 = **4** | 1 | **5** |
| **Node total** | | | **137** |

### Active path — step by step

Given `x: (16, n, n)` and `is_active_flags: (16,)`:

```
Step 1 — Threshold gate
  active_count = sum(is_active_flags)          # scalar int
  if active_count < 8  →  return zeros (n,n), False

Step 2 — Determinant scoring  [stable via slogdet]
  _, log_abs_det = jnp.linalg.slogdet(x)      # (16,) — log|det| per matrix
  scores = where(is_active_flags, log_abs_det, -inf)

Step 3 — Top-12 selection  [parameter-free]
  indices = top_k(scores, 12)                  # 12 best by |det|
  top_x   = x[indices]                         # (12, n, n)

Step 4 — Split
  top_interact_x  = top_x[:8]                  # (8, n, n)  → pairwise mix
  next_preserve_x = top_x[8:]                  # (4, n, n)  → kept raw

Step 5 — Batched pairwise matmul  [8C2 = 28 pairs]
  idx1, idx2 = triu_indices(8, k=1)            # 28 static index pairs
  pairs = matmul(top_interact_x[idx1],          # (28, n, n)
                 top_interact_x[idx2])

Step 6 — Concat
  concat = [pairs, next_preserve_x]            # (32, n, n)

Step 7 — conv1
  hidden = conv1(concat)                        # (4, n, n)

Step 8 — conv2
  out = conv2(hidden)                           # (1, n, n)

Step 9 — tanh  [parameter-free]
  out = tanh(out)                               # (1, n, n)  element-wise
  out = out.squeeze(0)                          # (n, n)

Return: out (n,n), True
```

### Inactive path

If `active_count < 12`:
```
return zeros(n, n), False   # zero tensor, no compute in conv1/conv2
```
Implemented with `jax.lax.cond` — XLA traces both branches but executes only one.

### Compute scaling with spatial size `n`

| Step | Cost |
|------|------|
| `slogdet` × 16 | O(16 n³) — dominant |
| 28 matmuls of (n,n)×(n,n) | O(28 n³) |
| `conv1` — 32×4 per pixel | O(128 n²) |
| `conv2` — 4×1 per pixel | O(4 n²) |
| tanh — element-wise | O(n²) — negligible |

For typical `n = 8`: slogdet ≈ 16 × 512 = 8 192 ops; matmuls ≈ 28 × 512 = 14 336 ops.
For `n = 32`: slogdet ≈ 524 K ops; matmuls ≈ 917 K ops — `n³` dominant.

---

## 2. `DecoderLayer` — K Nodes in Parallel (`utils/decoder_layer.py`)

Batches K independent `Decoder` nodes that each draw their 16 inputs from the same previous-layer output pool via a pre-baked adjacency table.

### What it holds

| Attribute | Type | Shape | Trainable |
|-----------|------|-------|-----------|
| `decoders` | vmapped `Decoder` | leading dim K, weights stacked | **Yes** — K × 137 params |
| `parent_indices` | `np.ndarray` (int32) | `(K, 16)` | **No** — compile-time constant |

`parent_indices` is a plain NumPy array (not a JAX traced leaf). XLA sees exact integer literals at compile time and constant-folds the gather — no dynamic indexing overhead.

### How K is decided

`DecoderLayer` does **not** decide K. It receives a fully computed `parent_indices: (K, 16)` array from `DecoderCluster` and simply reads `K = parent_indices.shape[0]`.

### Forward pass

```
prev_outputs:   (M, n, n)
prev_is_active: (M,) bool

gathered       = prev_outputs[parent_indices]        # (K, 16, n, n)
gathered_flags = prev_is_active[parent_indices]      # (K, 16) bool

out, out_active = vmap(Decoder)(gathered, gathered_flags)
                →  (K, n, n),  (K,) bool
```

Duplicates in `parent_indices` (same prev-layer index appearing in multiple slots) are handled transparently by the gather — JAX simply reads the same position multiple times.

### Trainable parameters

$$\text{params}(K) = K \times 137$$

| K nodes | Params |
|---------|--------|
| 4 | 548 |
| 16 | 2 192 |
| 64 | 8 768 |
| 256 | 35 072 |

---

## 3. `DecoderCluster` — The Full Graph (`utils/decoder_cluster.py`)

Stacks multiple `DecoderLayer`s according to the constrained Gaussian fan-out wiring algorithm, tracks which nodes are "output nodes" (unconnected downstream), and gathers them all for the FC head.

### Constructor signature

```python
DecoderCluster(
    n_layers:  int,       # L — target depth of hidden layers
    max_nodes: int,       # N — hard cap on total nodes across all layers
    n_inputs:  int,       # encoder output count (typically 64 × n_modalities)
    key:       jax.Array,
)
```

### Wiring algorithm — layer by layer

#### Layer 0 — straight wiring (encoder → first decoder layer)

$$K_0 = \left\lceil \frac{n\_inputs}{16} \right\rceil$$

The $n\_inputs$ encoder indices are tiled to fill $K_0 \times 16$ slots, shuffled, and reshaped to `(K0, 16)`. Every encoder output is guaranteed to appear at least once.

**Example:** 64 encoder outputs → $K_0 = 4$ decoder nodes, each receiving 16 inputs (some encoder outputs appear in 2 slots due to tiling).

#### Layers 1 … L+1 — Gaussian fan-out wiring

For each of the $M$ nodes in the previous layer, sample a fan-out $f_i$:

| Network half | μ | σ | Clip range |
|---|---|---|---|
| First half (`layer_idx < midpoint`) | 18 | 3.0 | [8, 24] |
| Second half (`layer_idx ≥ midpoint`) | 12 | 3.0 | [6, 20] |

where `midpoint = (n_layers + 2) // 2`.

Then:
```
pool = repeat(node_index_i, f_i)   # total size = Σf_i
shuffle(pool)
trim pool to floor(|pool| / 16) × 16 slots
K_this = len(trimmed_pool) / 16

parent_indices = trimmed_pool.reshape(K_this, 16)
```

Any previous-layer node whose index never appears in the **used** portion of the pool is **unconnected** — it has no downstream consumer and is marked as an output node of that layer.

**Expected K given M previous nodes:**

$$\mathbb{E}[K] = \left\lfloor \frac{M \cdot \mu}{16} \right\rfloor$$

| M prev nodes | First-half (μ=18) | Second-half (μ=12) |
|---|---|---|
| 4 | 4 | 3 |
| 16 | 18 | 12 |
| 64 | 72 | 48 |
| 256 | 288 | 192 |

#### Stopping conditions

**Case A — `max_nodes` hit (terminal unique wiring):**

Detected when the Gaussian fan-out pool for the next layer would produce more nodes than `nodes_remaining = max_nodes - total_so_far`. A separate `_terminal_wire` path runs instead of `_gaussian_wire`:

```
shuffled = shuffle(arange(n_prev))      # each previous index exactly once

k_terminal = min(nodes_remaining, n_prev // 16)

used     = shuffled[: k_terminal * 16]
leftover = shuffled[k_terminal * 16 :]  # not consumed → output nodes

parent_indices = used.reshape(k_terminal, 16)  # zero duplicates across all slots
```

Key properties vs normal wiring:
- Each previous-layer node feeds **at most one** input slot total — no index appears twice
- `k_terminal` is bounded by **both** the node budget **and** the unique-wiring capacity (`n_prev // 16`)
- Leftover previous-layer nodes → output nodes of the previous layer
- All `k_terminal` terminal nodes → output nodes

Edge cases:

| Condition | Outcome |
|---|---|
| `n_prev < 16` | `k_terminal = 0`; no decoder created; all `n_prev` previous nodes become output nodes |
| `n_prev // 16 < nodes_remaining` | unique-wiring capacity is binding; `k_terminal = n_prev // 16`; remainder → output |
| `n_prev >= nodes_remaining * 16` | budget is binding; `k_terminal = nodes_remaining`; remainder → output |

**Case B — L+1 layers built (depth target reached):**
After completing the (L+1)-th layer (L target + 1 extra), that final layer is terminal. All its nodes become output nodes. Normal Gaussian wiring is used.

**Case C — exact budget on layer boundary:**
If a completed layer leaves `nodes_remaining = 0`, the loop exits immediately and the safety net marks it as terminal.

#### Output node tracking

```
output_node_indices: List[np.ndarray]   # one entry per layer
```

- Intermediate layers: indices of nodes not consumed by the next layer (can be empty `[]`)
- Terminal layer: `arange(K_last)` — all nodes

All gathered at forward time into a single `(total_output_nodes, n, n)` tensor.

### Parameter count scaling

Every node in every layer, regardless of which layer it lives in, has exactly **137 trainable parameters** (tanh is parameter-free and does not change this count).

$$\text{Total params} = N_{total} \times 137$$

where $N_{total} = \sum_{l} K_l$ is the total number of decoder nodes across all layers.  
When `max_nodes` is hit, $N_{total} \le N$ exactly — the terminal unique wiring ensures the budget is never exceeded while also never wasting slots on duplicates.

| Total nodes | Total params |
|---|---|
| 100 | 13 700 |
| 500 | 68 500 |
| 1 000 | 137 000 |
| 5 000 | 685 000 |
| 10 000 | 1 370 000 (~1.37 M) |

Note: `parent_indices` arrays (wiring tables) occupy `N_{total} × 16 × 4` bytes of **non-trainable** memory. For 10 000 nodes this is 640 KB — negligible.

### Memory: activation tensors during forward pass

Each layer produces `(K_l, n, n)` float32. Peak activation memory is dominated by the widest layer:

$$\text{peak bytes} = K_{max} \times n^2 \times 4$$

| K_max | n=8 | n=16 | n=32 |
|---|---|---|---|
| 64 | 16 KB | 65 KB | 262 KB |
| 256 | 65 KB | 262 KB | 1 MB |
| 1 024 | 262 KB | 1 MB | 4 MB |

Intermediate per-node activations inside the active path (before conv1):
- `concat: (32, n, n)` — dominates node-level memory
- `pairs: (28, n, n)` — intermediate (32 - 4 = 28 pair channels)

### Forward pass output

```python
cluster(encoder_out, encoder_flags)
# encoder_out:   (n_inputs, n, n)
# encoder_flags: (n_inputs,) bool
# →
# out:   (total_output_nodes, n, n)
# flags: (total_output_nodes,) bool
```

All output nodes from all layers (unconnected intermediates + final layer) are concatenated along axis 0. This tensor feeds directly into the Port Adapter (PALayer).

### Forward pass — call path

```
cluster.__call__
  └─ for each DecoderLayer l:
       gather:  prev_outputs[parent_indices_l]  →  (K_l, 16, n, n)
       vmap:    Decoder × K_l                   →  (K_l, n, n), (K_l,) bool
  └─ gather output nodes per layer
  └─ jnp.concatenate across layers              →  (total_output_nodes, n, n)
```

The Python `for` loop unrolls at XLA trace time because `self.layers` is a fixed-length list. All shapes are static at compile time — no Python-level branching on traced values.

---

## 4. Configuration Reference (`utils/config/decode.py`)

| Constant | Value | Role |
|---|---|---|
| `DECODER_MAX_PARENTS` | 16 | Inputs per decoder node |
| `DECODER_ACTIVATION_THRESHOLD` | 8 | Min active parents to fire (8/16 = 50%) |
| `DECODER_TOP_K_EXTRACT` | 12 | How many inputs survive slogdet filter |
| `DECODER_INTERACT_RANKS` | 8 | Top-8 → pairwise combinations (8C2 = 28) |
| `DECODER_PRESERVE_RANKS` | 4 | Ranks 9-12 passed raw (12 - 8 = 4) |
| `DECODER_INTERMEDIATE_CHANNELS` | 32 | 28 pairs + 4 preserved = 32 into conv1 |
| `DECODER_HIDDEN_CHANNELS` | 4 | conv1 output channels |
| `DECODER_OUT_CHANNELS` | 1 | conv2 output channels (squeezed to n,n) |
| `FANOUT_FIRST_MU` | 18 | Mean fan-out, first-half layers |
| `FANOUT_FIRST_SIGMA` | 3.0 | Std dev, first-half layers |
| `FANOUT_FIRST_LO / HI` | 8 / 24 | Hard clip bounds, first-half |
| `FANOUT_SECOND_MU` | 12 | Mean fan-out, second-half layers |
| `FANOUT_SECOND_SIGMA` | 3.0 | Std dev, second-half layers |
| `FANOUT_SECOND_LO / HI` | 6 / 20 | Hard clip bounds, second-half |

---

## 5. `PALayer` — Terminal Projection (`utils/pa_layer.py`)

The `PALayer` (Port Adapter) is the final stage of the full TMRM pipeline. It receives the `(total_output_nodes, n, n)` output from the `DecoderCluster` and projects it to task-specific logits or values using a lightweight 1×1 convolution followed by a reshape to a 1-D vector.

All nodes in the PA layer are always active — there is **no** `is_active` flag in this module.

### What it holds

| Attribute | Type | Role |
|-----------|------|------|
| `conv` | `eqx.nn.Conv2d` | 1×1 conv: `(n_decoder_nodes → pa_out_channels, n, n)` |
| `activation` | `callable` (static, non-trainable) | Element-wise activation applied after `conv` |

### Supported activations

Specified by the `activation` string argument at construction time:

| Name | Function | Notes |
|------|----------|-------|
| `'sigmoid'` | `jax.nn.sigmoid` | **Default** (`PA_DEFAULT_ACTIVATION`) |
| `'relu'` | `jax.nn.relu` | |
| `'tanh'` | `jnp.tanh` | |
| `'gelu'` | `jax.nn.gelu` | |
| `'identity'` | pass-through | No nonlinearity |

Unknown activation names raise `ValueError` at construction time.

### Call signature

```python
PALayer(n_decoder_nodes, pa_out_channels, n, key, activation='sigmoid')

# Forward:
x: (n_decoder_nodes, n, n)   →   activation(conv(x)).reshape(-1)   : (pa_out_channels × n²,)
```

The `DecoderCluster` tensor is passed directly (no pre-flattening needed):

```python
dec_out, _ = cluster(encoder_out, encoder_flags)
# dec_out: (total_output_nodes, n, n)
logits = pa_layer(dec_out)       # (pa_out_channels × n²,)
```

### Parameter count

$$\text{params} = n\_decoder\_nodes \times pa\_out\_channels + pa\_out\_channels$$

For the current defaults (`n_decoder_nodes=15`, `pa_out_channels=4`): $15 \times 4 + 4 = \mathbf{64}$ params per head.

The activation callable is stored as a `static` pytree field and contributes **zero** trainable parameters.

### Configuration constant

| Constant | Value | File |
|----------|------:|------|
| `PA_DEFAULT_ACTIVATION` | `'sigmoid'` | `utils/config/trainparams.py` |

### Gradient flow note

`PALayer` is fully in the active gradient path from the task loss. Gradients flow back through the `conv` weights into the `DecoderCluster` output tensor. The cluster's Decoder `conv1`/`conv2` weights therefore receive gradients from the task loss via `PALayer`. Note that this gradient chain does **not** continue back through the Decoder gate into the Encoder weights — see the gradient barrier description in `architecture.md §4.3`.

---

## 6. End-to-End Shape Flow

```
EncoderLayer output
  (N_encoders × 64, n, n)        ← n_inputs to cluster

DecoderCluster Layer 0
  K0 = ceil(n_inputs / 16)        ← e.g. 4 nodes for 64 inputs
  output: (K0, n, n)

DecoderCluster Layers 1…
  K_l ≈ K_{l-1} × μ / 16         ← expands in first half, contracts in second
  output: (K_l, n, n)

Output gathering
  (total_output_nodes, n, n)      ← all unconnected + final layer nodes

Port Adapter (PALayer)
  1×1 conv → (pa_out_channels, n, n)  ← channel reduction
  reshape → 1D vector             ← pa_out_channels × n²
  task-specific logits / values
```
