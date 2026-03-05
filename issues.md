## Known Issues

- **Vanishing gradients in decoder / port-adapter pathway**
  - Symptom: Gradients for decoder parameters (and downstream port-adapter
    layers) collapse toward zero during training, slowing or stopping
    convergence.
  - Likely causes:
    - Large/sparse pairwise matmul pipelines followed by strong nonlinearities
      (tanh/swish) producing small gradients.
    - Recent config changes increased `MAX_DECODER_NODES` and lowered
      `DECODER_ACTIVATION_THRESHOLD`, producing many tiny activations that
      dilute gradient signal.
  - Suggested mitigations:
    - Revert `MAX_DECODER_NODES` to previous sensible value (e.g. 45) or
      tune down; restore stricter `DECODER_ACTIVATION_THRESHOLD` (e.g. 12/16).
    - Replace hard saturating activations in hot paths with numerically
      stable alternatives or add explicit scaling (e.g. use `swish`/`relu`
      carefully, or scale pre-activations before `tanh`).
    - Add/verify gradient clipping (global L2) and monitor gradient norms.
    - Provide a `train_step_debug` path that computes/prints gradient norms
      (already present) and enable it selectively to tune.

- **Decoder layers all become fully active or fully inactive**
  - Symptom: Per-layer gating collapses to all-ones or all-zeros, removing the
    intended sparse dynamic routing and causing instability or wasted compute.
  - Likely causes:
    - Lowering `DECODER_ACTIVATION_THRESHOLD` from 12→9 makes many nodes
      satisfy the gate, so entire layers can flip to fully active.
    - Deterministic random seeding / slot assignment changes may align
      inputs so many nodes co-activate.
  - Suggested mitigations:
    - Make gating softer: replace hard threshold with a differentiable gate
      (e.g. sigmoid gating with a learned temperature) so the network can
      settle to an intermediate activation level.
    - Introduce a small stochastic perturbation in gating during training
      (concrete: additive Gaussian noise to gate logits before thresholding)
      and disable at eval time; keep RNG seed controlled for reproducibility.
    - Constrain max active fraction per layer via a Top-K soft selection
      (e.g. use top-k with continuous relaxation) rather than absolute
      thresholding.
    - Restore conservative `DECODER_ACTIVATION_THRESHOLD` while evaluating
      the softened-gate approach.

## Next steps

- If you want, I can implement one of the above mitigations now (pick one):
  - revert config (fast, immediate perf regain)
  - implement sigmoid/soft gating in `decoder_layer.py` (requires careful
    JIT-compatible changes)
  - add stochastic gating noise during training and controlled RNG handling
  - instrument more gradient logging and a small reproducible benchmark

Please tell me which mitigation to implement or if you prefer a config-only
rollback first.
