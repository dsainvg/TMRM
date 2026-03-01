# AI Agent Instructions for TMRM (Dynamic DAG Neural Network)

## Architecture & Core Philosophy
- **Static Topology, Dynamic Execution**: The network is a sparsely activated Directed Acyclic Graph (DAG) designed for accelerator compilation. The maximal architecture is fixed at initialization, employing boolean masking (`is_active` flags) for sparse inference routing.
- **Key Modules (`utils/`):**
  - **Encoders (`utils/encoder.py`)**: Dimension adapters mapping inputs down to 4 channels. They compute $\binom{4}{2}=6$ pairwise matrix products, returning an $(n, n, 10)$ output plus an `is_active` boolean flag.
  - **Decoders (`utils/decoder.py`)**: Cross-context hidden units with a strict activation threshold ($\ge 12/16$ parents active). They select the Top-8 inputs using a Log-Absolute Determinant filter and compute $\binom{8}{2}=28$ pair mixes, eventually returning an $(n, n, 4)$ shaped output.

## Coding Conventions & Hardware Limitations
- **No Python-layer Graph Mutation**: Graph structure relies entirely on static adjacency arrays constructed during initialization. Avoid Python `if/else` conditionals, dynamic loops over array values, or Python list appends during inference or training loops.
- **Vectorized Conditionals for Masking**: Leverage targeted functional conditionals (e.g., `jax.lax.cond` or `jax.lax.select`) to circumvent execution overhead for inactive branches. Inactive branches must return pre-allocated zero-tensors coupled with `is_active=False`. *(Note check dependencies, even if `torch` is active in `requirements.txt`, static execution graphs must be maintained in equivalent primitives like `torch.where` or `torch.cond`)*.
- **Numerical Stability for Matrix Scoring**: Never use raw determinants for decoder matrix ranking. Evaluate via `log-absolute-determinant` (e.g., `slogdet`) to prevent gradient instability securely.
- **Sorting via Top-K Primitives**: Use hardware-efficient `.top_k` logic to enforce thresholding. When padding inactive or zero matrices that should evade selection, hardcode their scores to `-inf`.
- **Vectorized Matrix Combinations**: Never iterate pairwise computations elementally. Pre-compute left/right pairs of target indices statically and execute combinatorial cross-mixes concurrently as grouped, batched matrix multiplication operations.
- **Thinking in JAX**: Adhere to JAX functional programming paradigms by ensuring pure functions, proper JAX PRNG key management, and avoiding in-place updates. If needed, refer to the [Thinking in JAX documentation](https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html) to write idiomatic and compatible code.

