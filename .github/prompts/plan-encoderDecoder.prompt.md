## Plan: JAX/Equinox Encoders & Decoders with Updated Concatenation

This plan focuses exclusively on building out the `Encoder` and `Decoder` computational modules using strict JAX primitives and Equinox configurations, updating the decoder concatenation logic to use the 9th-12th ranked features rather than the top 8.

**Steps**

1. **Update Dependencies** in `requirements.txt`
   - Remove PyTorch (`torch`, `torchvision`).
   - Add JAX/XLA dependencies: `jax`, `jaxlib`, `equinox`, `optax`.

2. **Implement the Base Encoder Node** in `utils/encoder.py`
   - Define `Encoder` as an `eqx.Module`.
   - Take a `(1, n, n)` input and a boolean `is_active` flag.
   - **Active Path:**
     - Expand `1` channel to `4` via `eqx.nn.Conv2d`.
     - Perform batched $\binom{4}{2} = 6$ pairwise matrix products across the 4 channels.
     - Concatenate the 4 conv channels + 6 paired channels yielding `(10, n, n)`.
     - Output projection via `eqx.nn.Conv2d` compressing to `(8, n, n)`.
   - **Masking:** Wrap execution in `jax.lax.cond`. If `is_active=False`, cleanly bypass compute and return a zero-allocated `(8, n, n)` tensor alongside `is_active=False`.

3. **Implement the Base Decoder Node** in `utils/decoder.py`
   - Define `Decoder` as an `eqx.Module`.
   - Take a `(16, n, n)` input alongside a `(16,)` length boolean masking array for parent nodes.
   - **Gate Check:** Execute path conditionally only if `jnp.sum(is_active) >= 12` using `jax.lax.cond`. If false, bypass compute and return a pre-allocated zero tensor of shape `(4, n, n)` alongside `False`.
   - **Active Path:**
     - Run `jnp.linalg.slogdet` strictly on the 16 input matrices to calculate proxy scores. Overwrite inactive inputs with `-jnp.inf` using `jnp.where`.
     - Use `jax.lax.top_k(k=12)` to identify the top 12 valid matrices.
     - Split the selection:
       - The *first 8* matrices (Rank 1-8) feed a batched $\binom{8}{2} = 28$ paired matrix multiplication.
       - The *next 4* matrices (Rank 9-12) are reserved for concatenation.
     - Concatenate the 28 paired products and the 4 reserved feature maps yielding `(32, n, n)`.
     - Apply final dimensionality reduction via `eqx.nn.Conv2d` reducing 32 channels down to `(4, n, n)`.
