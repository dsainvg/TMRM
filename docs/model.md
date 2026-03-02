# Model Class — Multi-Problem TMRM Network

This document describes the top-level `Model` class that ties together every
component of the Dynamic DAG Neural Network into a single, trainable
`eqx.Module`. It covers configuration, construction, forward pass semantics,
the multi-problem paradigm, encoder masking, parameter counts, training
strategy, performance characteristics, and serialisation.

---

## 1. Design Overview

The `Model` class implements a **shared backbone + per-problem head** pattern:

```
                   ┌─────────────┐
  (N,1,n,n) ──►   │ EncoderLayer │   ◄── shared across all problems
                   └──────┬──────┘
                          │  (64·N, n, n)  +  flags
                   ┌──────▼──────┐
                   │DecoderCluster│   ◄── shared across all problems
                   └──────┬──────┘
                          │  (output_nodes, n, n)
                      flatten → 1-D
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
       ┌─────────┐  ┌─────────┐  ┌─────────┐
       │ FCLayer₀ │  │ FCLayer₁ │  │ FCLayer₂ │  ◄── one per problem
       └────┬────┘  └────┬────┘  └────┬────┘
            ▼             ▼             ▼
        (out₀,)       (out₁,)       (out₂,)
```

**Key principle:** the internal DAG (encoders + decoders) is *identical* for
every problem. Only the terminal FC head and the *encoder mask* differ.

---

## 2. Configuration Dataclasses

Configuration is captured by frozen dataclasses in `utils/config/model.py` and
`utils/config/training.py`. They hold **no trainable arrays** — only structural
and optimisation metadata.

### 2.1 `ProblemConfig`

| Field              | Type  | Default   | Description                                                |
|--------------------|-------|-----------|------------------------------------------------------------|
| `n_encoders_used`  | `int` | —         | How many encoder slots this problem activates (≥ 1).       |
| `fc_out_features`  | `int` | —         | Dimensionality of the FC head output.                      |
| `fc_activation`    | `str` | `'relu'`  | Activation after the FC linear projection.                 |

Supported activations: `relu`, `gelu`, `tanh`, `sigmoid`, `identity`.

### 2.2 `ModelConfig`

| Field               | Type                       | Description                                              |
|----------------------|----------------------------|----------------------------------------------------------|
| `n`                  | `int`                      | Spatial matrix size — all matrices are $(n, n)$.         |
| `n_encoders`         | `int`                      | Total encoder slots (shared pool for all problems).      |
| `n_decoder_layers`   | `int`                      | Target depth $L$ for the `DecoderCluster`.               |
| `max_decoder_nodes`  | `int`                      | Hard cap on total decoder nodes across all layers.       |
| `problems`           | `tuple[ProblemConfig, ...]` | One entry per problem head. At least one required.       |

`ModelConfig.__post_init__` validates every field — positivity, bounds
(`n_encoders_used ≤ n_encoders`), and known activation names — raising
`ValueError` (or `TypeError`) immediately on construction of an invalid config.

#### On the role of `n`

`n` is a **structural config constant**, not a per-sample variable. It
determines the spatial dimension of every matrix throughout the entire network:
encoder convolutions, pairwise matmuls, decoder gates — all operate on
$(n, n)$ matrices. Changing `n` requires re-building the model from scratch
(new key, new weights). Think of it as the "resolution" knob: small $n$
(4–8) for fast prototyping, larger $n$ (16–64) for richer feature
representations at the cost of $O(n^3)$ matmul time per node.

### 2.3 `TrainingConfig`

| Field              | Type    | Default       | Description                                             |
|--------------------|---------|---------------|---------------------------------------------------------|
| `batch_size`       | `int`   | `1`           | Samples per gradient step.                              |
| `n_epochs`         | `int`   | `50`          | Total training epochs.                                  |
| `learning_rate`    | `float` | `3e-4`        | Peak / base LR.                                        |
| `optimiser`        | `str`   | `'adam'`       | `'adam'`, `'adamw'`, `'sgd'`, or `'rmsprop'`.          |
| `weight_decay`     | `float` | `0.0`         | L2 regularisation (used by `adamw`).                   |
| `lr_schedule`      | `str`   | `'constant'`  | `'constant'`, `'cosine'`, `'linear'`, `'warmup_cosine'`.|
| `warmup_steps`     | `int`   | `0`           | Linear warm-up steps before schedule starts.           |
| `grad_clip_norm`   | `float` | `0.0`         | Max global grad norm (0 = disabled).                   |
| `log_every`        | `int`   | `10`          | Log interval in steps.                                 |
| `seed`             | `int`   | `0`           | Base PRNG seed for data shuffling, etc.                |

All fields are validated on construction (`__post_init__`).

### 2.4 Convenience property

```python
cfg.n_problems   # → len(cfg.problems)
```

---

## 3. Model Construction

```python
from utils.config.model import ModelConfig, ProblemConfig
from model import Model
import jax

cfg = ModelConfig(
    n=8,
    n_encoders=4,
    n_decoder_layers=2,
    max_decoder_nodes=50,
    problems=(
        ProblemConfig(n_encoders_used=2, fc_out_features=10),
        ProblemConfig(n_encoders_used=3, fc_out_features=64, fc_activation='identity'),
    ),
)

model = Model(cfg, key=jax.random.key(0))
print(model.count_params())
```

Internally `__init__` splits the JAX PRNG key into four sub-keys:

| Sub-key   | Purpose                                            |
|-----------|----------------------------------------------------|
| `k_enc`   | Initialise the shared `EncoderLayer` weights.      |
| `k_dec`   | Initialise the shared `DecoderCluster` weights.    |
| `k_fc`    | Split further into one key per FC head.            |
| `k_mask`  | Seed a NumPy `default_rng` for encoder mask generation. |

### 3.1 Shared backbone

* **`EncoderLayer`** — constructed with `n_inputs = n_encoders`.  Each input is
  a 1-to-8 branching tree of 9 Encoder nodes producing 64 channels.
* **`DecoderCluster`** — receives `n_inputs = n_encoders × 64` and builds the
  randomised DAG with `n_decoder_layers` / `max_decoder_nodes`.

### 3.2 Per-problem FC heads

One `FCLayer` per problem, each with:

$$\text{in\_features} = \text{decoder\_cluster.n\_output\_nodes} \times n^{2}$$

and `out_features` / `activation` taken from the corresponding `ProblemConfig`.

### 3.3 Per-problem encoder masks

For each problem, a random subset of `n_encoders_used` encoder slots is chosen
(without replacement) from the `n_encoders` pool. This produces a boolean
NumPy array of shape `(n_encoders,)`.

```python
model.encoder_masks[0]              # e.g. array([ True, False, False,  True])
model.active_encoder_indices(0)     # e.g. array([0, 3])
```

Masks are **plain NumPy arrays** — compile-time constants for XLA,
following the same pattern as `parent_indices` throughout the codebase.

---

## 4. Parameter Counts

### 4.1 Per-node formulas

**Encoder node** — two 1×1 convolutions:

| Layer      | Kernel | Weights           | Bias | Total                       |
|------------|--------|-------------------|------|-----------------------------|
| `conv1`    | 1×1    | `in_c × 4`       | `4`  | `in_c × 4 + 4`             |
| `conv2`    | 1×1    | `10 × out_c`     | `out_c` | `10 × out_c + out_c`     |

For default root encoder (1 → 8): `conv1` = 1×4+4 = **8**, `conv2` = 10×8+8 = **88** → **96 params per node**.

For default leaf encoder (1 → 8): same → **96 params per node**.

**Decoder node** — two 1×1 convolutions:

| Layer      | Kernel | Weights          | Bias | Total     |
|------------|--------|------------------|------|-----------|
| `conv1`    | 1×1    | `32 × 4`        | `4`  | **132**   |
| `conv2`    | 1×1    | `4 × 1`         | `1`  | **5**     |

→ **137 params per Decoder node**.

**FCLayer** — one dense linear projection:

$$\text{params} = \text{fc\_in} \times \text{fc\_out} + \text{fc\_out}$$

### 4.2 Per-component rollup

| Component           | Formula                              | Example (N=4, D=20, K=16, n=8, fc_out=10) |
|---------------------|--------------------------------------|--------------------------------------------|
| **EncoderLayer**    | $N \times 9 \times 96$               | 4 × 9 × 96 = **3 456**                    |
| **DecoderCluster**  | $D \times 137$                       | 20 × 137 = **2 740**                      |
| **FCLayer (per head)** | $K \times n^2 \times \text{fc\_out} + \text{fc\_out}$ | 16 × 64 × 10 + 10 = **10 250**  |
| **Total (1 problem)** | sum                                 | **16 446**                                |

Where:
- $N$ = `n_encoders`, $D$ = total decoder nodes, $K$ = `n_output_nodes`
- Each encoder input spawns 9 nodes (1 root + 8 leaves), each with 96 params
- FC input size = $K \times n^2$

### 4.3 `model.count_params()` method

```python
>>> model.count_params()
{'encoder_layer': 3456, 'decoder_cluster': 2740, 'fc_heads': [10250], 'total': 16446}
```

Returns a dict with per-component breakdowns. Use this to verify sizing and
monitor scaling as you increase `n`, `n_encoders`, or `max_decoder_nodes`.

### 4.4 Scaling behaviour

| Knob increased       | Encoder params | Decoder params | FC params   | Dominant cost at scale |
|----------------------|----------------|----------------|-------------|------------------------|
| `n_encoders` ↑       | Linear in N    | ↑ (more inputs → more L0 nodes) | ↑ (K grows) | FC layer               |
| `max_decoder_nodes` ↑ | unchanged      | Linear in D    | ↑ (K grows) | FC layer               |
| `n` ↑                | unchanged      | unchanged      | Quadratic in $n$ ($n^2$ per output node) | FC layer |
| `fc_out_features` ↑  | unchanged      | unchanged      | Linear      | FC layer               |

**Key insight:** the FC layer dominates total parameter count for larger
models because its input dimension is $K \times n^2$ and can be very large.
The encoder and decoder are relatively lightweight (96–137 params per node).

---

## 5. Forward Pass

```python
out = model(problem_idx, xs)
```

| Argument       | Shape / Type                | Notes                                          |
|----------------|-----------------------------|-------------------------------------------------|
| `problem_idx`  | Python `int`                | Selects mask + FC head. **Not** JAX-traced.     |
| `xs`           | `(n_encoders, 1, n, n)`     | Full input tensor — all encoder slots.          |
| **returns**    | `(fc_out_features,)`        | Task-specific output from the selected FC head. |

### 5.1 Step-by-step data flow

1. **Encoder mask lookup** — `flags = jnp.array(encoder_masks[problem_idx])`.
   Slots marked `False` are deactivated: the Encoder's `lax.cond` gate
   short-circuits to zero tensors.

2. **Shared EncoderLayer** — maps `(N, 1, n, n)` → `(64·N, n, n)` plus
   `(64·N,)` activity flags.

3. **Shared DecoderCluster** — routes the encoder output through the
   randomised DAG.  Decoder nodes with < 12/16 active parents shut down.

4. **Flatten** — all decoder output nodes are reshaped into a single 1-D vector
   of length `n_output_nodes × n²`.

5. **Problem-specific FCLayer** — `fc_heads[problem_idx]` projects the flat
   vector to `(fc_out_features,)`.

### 5.2 Why `problem_idx` must be a Python int

Each problem may have a **different** `fc_out_features`, meaning the output
shape can vary. Because XLA programs are shape-specialised, each distinct
`problem_idx` value triggers a separate XLA compilation. This is intentional
and unavoidable.

### 5.3 Computational complexity per forward pass

| Stage           | Dominant operation                          | Complexity                          |
|-----------------|---------------------------------------------|-------------------------------------|
| Encoder (per node)  | 6 × batched matmul `(n,n)@(n,n)`       | $O(6 \cdot n^3)$                   |
| Encoder total   | N × 9 nodes                                 | $O(54 \cdot N \cdot n^3)$          |
| Decoder (per node)  | `slogdet` $(16,n,n)$ + 28 × matmul     | $O(16 \cdot n^3 + 28 \cdot n^3)$   |
| Decoder total   | D active nodes                               | $O(44 \cdot D_{\text{active}} \cdot n^3)$ |
| FC head         | Dense matmul `(K·n², fc_out)`                | $O(K \cdot n^2 \cdot \text{fc\_out})$ |

The pairwise matmuls in encoder/decoder are the bottleneck. They scale as
$O(n^3)$ per pair and there are many pairs (6 per encoder node, 28 per
decoder node).

---

## 6. Training Strategy

### 6.1 Two-stage loss paradigm

Due to the **gradient barrier** at the Decoder gate (`lax.cond` + `slogdet` +
`top_k`), a single end-to-end loss *cannot* train both encoders and
decoders/FC simultaneously:

| Loss              | Trains                          | Applied to              |
|-------------------|---------------------------------|-------------------------|
| **Encoder loss**  | `EncoderLayer` conv weights     | `EncoderLayer` outputs  |
| **Task loss**     | `DecoderCluster` + `FCLayer`    | `model(problem_idx, xs)`|

### 6.2 Training a single problem (task loss)

```python
import optax, equinox as eqx, jax.numpy as jnp
from utils.config.training import TrainingConfig

tcfg = TrainingConfig(
    batch_size=1,
    n_epochs=50,
    learning_rate=3e-4,
    optimiser='adam',
)

opt   = optax.adam(tcfg.learning_rate)
state = opt.init(eqx.filter(model, eqx.is_inexact_array))

@eqx.filter_jit
def step(model, opt_state, xs, target):
    def loss_fn(m):
        pred = m(0, xs)
        return jnp.mean((pred - target) ** 2)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, new_state = opt.update(
        eqx.filter(grads, eqx.is_inexact_array),
        opt_state,
        eqx.filter(model, eqx.is_inexact_array),
    )
    return eqx.apply_updates(model, updates), new_state, loss
```

### 6.3 Problem isolation

Training problem 0 produces **zero gradients** for problem 1's FC head.
The shared backbone receives gradients from whichever problem is trained;
the other FC heads are untouched.

### 6.4 Multi-problem round-robin training

```python
for epoch in range(tcfg.n_epochs):
    for p_idx in range(model.n_problems):
        xs, target = get_batch(p_idx)
        model, state, loss = step(model, state, xs, target)
```

Each `p_idx` value re-uses its cached XLA program after the first compilation.

---

## 7. Practical Training Guidance

### 7.1 Batch size recommendations

| Scenario                | Recommended `batch_size` | Rationale                                           |
|-------------------------|--------------------------|-----------------------------------------------------|
| Tiny model (n≤4, D≤20) | 1–4                      | Minimal memory footprint, quick compile.            |
| Medium model (n=8, D=50)| 4–16                     | Amortise XLA overhead over multiple samples.        |
| Large model (n≥16, D≥100)| 1–4                     | $n^3$ matmul cost dominates; memory is the bottleneck.|

**Why batch_size=1 is the default:** The model performs many $O(n^3)$ matmuls
per sample. Unlike standard NNs, the per-sample compute is already heavy.
Start with 1, measure memory, and increase only after confirming GPU memory
headroom.

For batch_size > 1, use an external loop over samples (each calling the
JIT-compiled step function) rather than vmapping the entire model — nested
vmaps over the internal double-vmapped EncoderLayer can trigger XLA shape
conflicts.

### 7.2 Learning rate

* **Adam / AdamW**: `1e-4` to `3e-3` works well. The default `3e-4` is a
  safe starting point.
* **SGD**: much higher LR needed (`0.01`–`0.1`); not recommended for this
  architecture due to the sparse gradient structure.

### 7.3 Gradient clipping

Recommended when using high learning rates or larger `n` — the `slogdet`
scores can produce large gradient magnitudes:

```python
tcfg = TrainingConfig(grad_clip_norm=1.0, learning_rate=1e-3)
```

### 7.4 Expected training behaviour

| Observation                                    | Explanation                                            |
|------------------------------------------------|--------------------------------------------------------|
| Loss drops quickly in first 5–10 steps         | FC head bias learns the mean of the target quickly.    |
| Loss plateau for 10–50 steps                   | Decoder internal routing stabilising — many nodes inactive initially. |
| Gradual descent after plateau                  | Active subgraph weights refine their cross-context mixing. |
| Encoder weights do not change (task loss only) | Expected — gradient barrier. Use encoder-specific loss. |

### 7.5 When does the model *not* work well?

* **Very small n (n=2–3):** Pairwise matmuls on 2×2 or 3×3 matrices carry
  almost no useful information. `slogdet` ranking is noisy on tiny matrices.
* **Very few active encoders (1 out of many):** Only 64 of the 64·N decoder
  inputs are active → nearly all decoder nodes fail the 12/16 threshold → the
  flattened vector is mostly zeros → FC head receives a near-constant zero
  input.
* **Too few decoder nodes (`max_decoder_nodes < ~16`):** The cluster has too
  few nodes for meaningful cross-context mixing.

### 7.6 Strengths

* **Modularity:** Add a new problem by appending a `ProblemConfig` — no
  re-architecture needed.
* **Sparse routing:** Only active decoders compute, so even large DAGs are
  efficient when many nodes are gated off.
* **Pairwise feature interaction:** The $\binom{8}{2} = 28$ matrix products in
  each decoder give every node a rich second-order feature representation
  without explicit attention mechanisms.
* **Static topology, dynamic execution:** Compiles cleanly to XLA — no Python
  overhead during the forward pass of a compiled step.
* **Implicit regularisation:** Sparse activation (inactive nodes → zero
  gradients) acts as dropout-like regularisation.

---

## 8. About `n` — The Spatial Matrix Size

### 8.1 What it controls

Every matrix in the network is $(n, n)$. This includes:
- Encoder inputs and outputs
- All pairwise matmul products
- `slogdet` computations in decoders
- Decoder outputs
- The flattened vector entering the FC layer (size $K \times n^2$)

### 8.2 `n` is a config constant, not a runtime parameter

`n` is baked into `ModelConfig` at construction time and determines:
- The shape of all weight tensors (conv filters, FC weight matrix)
- The XLA-compiled program's static shapes

You **cannot** change `n` without rebuilding the model. It is analogous to
"image resolution" in a CNN — fixed at architecture construction time.

### 8.3 Choosing `n`

| Use case                  | Suggested `n` | Notes                                                  |
|---------------------------|---------------|--------------------------------------------------------|
| Unit tests / debugging    | 4             | Fast compilation, tiny memory.                         |
| Prototyping               | 6–8           | Enough signal for meaningful `slogdet` ranking.        |
| Production (small tasks)  | 8–16          | Good balance of expressivity and speed.                |
| Production (rich features)| 16–32         | FC layer becomes very large ($K \times n^2$).          |
| Research / ablation       | 32–64         | Expensive; use only if feature matrices are naturally this large. |

### 8.4 Memory and compute scaling with `n`

| Quantity                    | Scaling    | Example: n=8 → n=16                     |
|-----------------------------|-----------|------------------------------------------|
| Per-matmul FLOPS            | $O(n^3)$  | 8× increase (512 → 4096)                |
| Per-node output size        | $O(n^2)$  | 4× increase                              |
| FC input dimension          | $O(K n^2)$| 4× increase per output node              |
| FC weight matrix            | $O(K n^2 \times \text{fc\_out})$ | 4× increase |
| Total memory (weights)      | Dominated by FC | ~4× for doubling $n$              |
| Total memory (activations)  | $O(D \cdot n^2)$ | 4× per forward pass             |

---

## 9. JIT Compilation Notes

| Item                     | Detail                                                  |
|--------------------------|---------------------------------------------------------|
| First call per problem   | Triggers XLA compilation (may take several seconds).    |
| Subsequent calls         | Reuses cached program — expected to be fast.            |
| Static arguments         | `problem_idx` (Python int), `config`, `encoder_masks`.  |
| Traced arguments         | `xs` (input tensor), model weights.                     |

### 9.1 Measured performance (tiny model: n=4, N=4, D=20)

| Metric                  | Typical value |
|-------------------------|---------------|
| JIT compilation         | 0.8 – 1.5 s  |
| Cached training step    | 8 – 50 ms    |
| Cached forward call     | 2 – 20 ms    |

These numbers are from CPU-only JAX. GPU execution is expected to be
significantly faster for the matmul-heavy stages.

---

## 10. Serialisation

Equinox provides `tree_serialise_leaves` / `tree_deserialise_leaves` for
checkpoint save/load. Because `encoder_masks` are plain NumPy arrays (not JAX
arrays), they **are not** serialised as trainable leaves. To restore a model:

```python
import io, equinox as eqx

# Save
buf = io.BytesIO()
eqx.tree_serialise_leaves(buf, model)

# Load — rebuild the skeleton first (masks regenerated), then overwrite weights
buf.seek(0)
skeleton = Model(cfg, jax.random.key(999))
restored = eqx.tree_deserialise_leaves(buf, skeleton)
```

> **Note:** The skeleton must use the same `ModelConfig`. If the same top-level
> PRNG key is used, masks regenerate identically. If a *different* key is used,
> masks will differ — but trainable weights are overwritten correctly regardless.

---

## 11. Helper Properties

| Property / Method                   | Returns                                            |
|-------------------------------------|----------------------------------------------------|
| `model.n_problems`                  | `int` — number of problem heads.                   |
| `model.n_decoder_output_nodes`      | `int` — total output nodes from the DecoderCluster.|
| `model.fc_in_features`              | `int` — flattened vector size entering each FC head.|
| `model.active_encoder_indices(idx)` | `np.ndarray[int]` — sorted active encoder indices. |
| `model.config`                      | `ModelConfig` — the frozen config (static field).  |
| `model.count_params()`              | `dict` — parameter breakdown (encoder, decoder, FC heads, total). |

---

## 12. Quick Reference — Shapes at Each Stage

For a `ModelConfig(n=8, n_encoders=4, n_decoder_layers=2, max_decoder_nodes=50)`:

| Stage                  | Shape                              | Notes                       |
|------------------------|------------------------------------|-----------------------------|
| Input `xs`             | `(4, 1, 8, 8)`                    | 4 encoder inputs            |
| After EncoderLayer     | `(256, 8, 8)` + `(256,)` flags    | 4 × 64 = 256 channels      |
| After DecoderCluster   | `(K, 8, 8)` + `(K,)` flags        | K = `n_output_nodes`        |
| Flatten                | `(K × 64,)`                       | K × 8²                     |
| FC head output         | `(fc_out_features,)`              | Problem-specific            |

---

## 13. File Map

| File                         | Contents                                                  |
|------------------------------|-----------------------------------------------------------|
| `model.py`                   | `Model` class (`eqx.Module`).                             |
| `utils/config/model.py`      | `ProblemConfig`, `ModelConfig` frozen dataclasses.         |
| `utils/config/training.py`   | `TrainingConfig` frozen dataclass.                        |
| `utils/config/__init__.py`   | Re-exports all config symbols.                            |
| `tests/test_model.py`        | Unit tests — config, construction, forward, training, JIT, serialisation. |
| `tests/test_model_training.py`| Stress tests — overfitting, batching, timing, stability, checkpoints. |

---

## 14. Limitations & Future Work

* **No sample-level batching in `__call__`.** The current forward processes a
  single `(N, 1, n, n)` input. Sample-level batching can be achieved
  externally via loop or selective vmap — nested vmaps over the internal
  double-vmapped EncoderLayer can trigger XLA shape conflicts.

* **`problem_idx` recompilation.** Each distinct `problem_idx` traces a
  separate XLA program. For many problems this increases compilation memory.

* **Encoder mask persistence.** Masks are not serialised as trainable leaves;
  use the same key (or persist masks separately) for exact round-trip fidelity.

* **Gradient barrier.** Encoders cannot be trained through the task loss —
  a separate encoder loss must be designed per application.

* **FC layer size dominance.** For large $n$ and many output nodes, the FC
  weight matrix can become very large. Future work could explore factored /
  low-rank projections or multiple smaller FC layers.
