"""
Batched-training test suite for TMRM.

All tests in this file exercise the true SIMD-batch parallelism that becomes
possible after replacing lax.cond → jnp.where in Encoder.__call__.

Covers:
  - Encoder.__call__ under jax.vmap (active, inactive, mixed)
  - Encoder inactive path: zero output and zero gradient under vmap
  - EncoderLayer.__call__ under jax.vmap (full stack, batched inputs)
  - Model.__call__ under jax.vmap (batched forward pass)
  - train_step(B=32) returns scalar loss, updates weights, stays finite
  - Batched vmap loss == mean of individual per-sample losses (math check)
  - Loss decreases over several batched train steps (learning check)
  - No NaN/Inf produced by batched training for 10 steps
"""

import sys
import pathlib

# Ensure the workspace root is on sys.path when tests are run directly
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from utils.encoder import Encoder
from utils.encoder_layer import EncoderLayer
from model import Model
from utils.config.model import ModelConfig, ProblemConfig
from utils.config.training import TrainingConfig


# ─── Fixtures / helpers ───────────────────────────────────────────────────────

@pytest.fixture
def key():
    return jax.random.key(0)


def _sudoku_model(key, n=4, n_encoders=5):
    """Minimal TMRM model matching the 4×4 Sudoku train.py configuration."""
    cfg = ModelConfig(
        n=n,
        n_encoders=n_encoders,
        n_decoder_layers=2,
        max_decoder_nodes=20,
        problems=(
            ProblemConfig(
                n_encoders_used=n_encoders,
                fc_out_features=n * n * 4,   # 64 for 4×4 with 4 digit channels
                fc_activation="sigmoid",
            ),
        ),
    )
    return Model(cfg, key=key)


def _random_batch(key, B, n_encoders=5, n=4):
    """Return (xs_batch, y_batch) matching the Sudoku encoding shape."""
    k1, k2 = jax.random.split(key)
    xs = jax.random.normal(k1, (B, n_encoders, 1, n, n))
    y  = jax.random.uniform(k2, (B, n * n * 4))  # (B, 64)
    return xs, y


# ══════════════════════════════════════════════════════════════════════════════
# 1.  ENCODER VMAP COMPATIBILITY
# ══════════════════════════════════════════════════════════════════════════════

class TestEncoderVmapCompatible:
    """Encoder.__call__ must work under jax.vmap now that lax.cond is gone."""

    def test_vmap_all_active_shapes(self, key):
        """jax.vmap over batch of active inputs: shapes correct."""
        enc = Encoder(key)
        B, n = 8, 6
        xs    = jax.random.normal(key, (B, 1, n, n))
        flags = jnp.ones(B, dtype=bool)

        outs, acts = jax.vmap(enc)(xs, flags)

        assert outs.shape == (B, 8, n, n), f"Expected ({B},8,{n},{n}), got {outs.shape}"
        assert acts.shape == (B,)
        assert bool(jnp.all(acts))

    def test_vmap_all_inactive_are_zeros(self, key):
        """jax.vmap with all inactive flags: every output channel is zero."""
        enc = Encoder(key)
        B, n = 6, 4
        xs    = jax.random.normal(key, (B, 1, n, n))
        flags = jnp.zeros(B, dtype=bool)

        outs, acts = jax.vmap(enc)(xs, flags)

        assert outs.shape == (B, 8, n, n)
        assert bool(jnp.all(~acts))
        assert bool(jnp.all(outs == 0.0)), "All inactive outputs must be exactly zero"

    def test_vmap_mixed_flags(self, key):
        """Mixed active/inactive: active indices non-zero, inactive indices zero."""
        enc = Encoder(key)
        B, n = 4, 4
        xs    = jax.random.normal(key, (B, 1, n, n))
        flags = jnp.array([True, False, True, False])

        outs, acts = jax.vmap(enc)(xs, flags)

        assert outs.shape == (B, 8, n, n)
        assert bool(jnp.all(acts == flags))
        assert bool(jnp.any(outs[0] != 0.0)), "Active sample 0 must be non-zero"
        assert bool(jnp.any(outs[2] != 0.0)), "Active sample 2 must be non-zero"
        assert bool(jnp.all(outs[1] == 0.0)), "Inactive sample 1 must be zero"
        assert bool(jnp.all(outs[3] == 0.0)), "Inactive sample 3 must be zero"

    def test_vmap_jit_compiles(self, key):
        """eqx.filter_jit(jax.vmap(Encoder)) compiles and runs without error."""
        enc = Encoder(key)
        B, n = 4, 6
        xs    = jax.random.normal(key, (B, 1, n, n))
        flags = jnp.ones(B, dtype=bool)

        outs, acts = eqx.filter_jit(jax.vmap(enc))(xs, flags)

        assert outs.shape == (B, 8, n, n)
        assert bool(jnp.all(acts))

    def test_vmap_batch_outputs_differ(self, key):
        """Different inputs in a batch produce different outputs."""
        enc = Encoder(key)
        n = 4
        ks  = jax.random.split(key, 4)
        xs  = jnp.stack([jax.random.normal(k, (1, n, n)) for k in ks])
        flags = jnp.ones(4, dtype=bool)

        outs, _ = jax.vmap(enc)(xs, flags)

        # Any two distinct rows in outs should differ
        assert not bool(jnp.allclose(outs[0], outs[1])), "Different inputs must differ"

    def test_inactive_gradient_zero_under_vmap(self, key):
        """Inactive encoder weight gradient must be zero — jnp.where masks it."""
        enc = Encoder(key)
        n = 4
        x = jax.random.normal(key, (1, n, n))

        def loss(model):
            out, _ = model(x, jnp.array(False))
            return jnp.sum(out)

        grads = eqx.filter_grad(loss)(enc)
        for g in jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array)):
            assert bool(jnp.all(g == 0.0)), f"Inactive encoder grad non-zero: {g}"

    def test_inactive_input_gradient_zero(self, key):
        """Gradient w.r.t. input x must be zero when inactive."""
        enc = Encoder(key)
        n = 4
        x = jax.random.normal(key, (1, n, n))

        grad_x = jax.grad(lambda xv: jnp.sum(enc(xv, jnp.array(False))[0]))(x)
        assert bool(jnp.all(grad_x == 0.0)), "Input gradient must be zero when inactive"


# ══════════════════════════════════════════════════════════════════════════════
# 2.  ENCODERLAYER VMAP COMPATIBILITY
# ══════════════════════════════════════════════════════════════════════════════

class TestEncoderLayerVmapCompatible:
    """EncoderLayer.__call__ must be compatible with an outer jax.vmap batch."""

    def test_vmap_batch_forward_shapes(self, key):
        """jax.vmap over B inputs → (B, 64*N, n, n) output."""
        N, n = 5, 4
        layer = EncoderLayer(n_inputs=N, key=key)
        B     = 6
        xs_b  = jax.random.normal(key, (B, N, 1, n, n))
        flags = jnp.ones(N, dtype=bool)

        outs, acts = jax.vmap(lambda xs: layer(xs, flags))(xs_b)

        assert outs.shape == (B, N * 64, n, n), f"Unexpected shape {outs.shape}"
        assert acts.shape == (B, N * 64)

    def test_vmap_encoder_layer_jit(self, key):
        """EncoderLayer vmap forward compiles under eqx.filter_jit."""
        N, n = 3, 4
        layer = EncoderLayer(n_inputs=N, key=key)
        B     = 4
        xs_b  = jax.random.normal(key, (B, N, 1, n, n))
        flags = jnp.ones(N, dtype=bool)

        batched = eqx.filter_jit(jax.vmap(lambda xs: layer(xs, flags)))
        outs, acts = batched(xs_b)

        assert outs.shape == (B, N * 64, n, n)
        assert acts.shape == (B, N * 64)
        assert bool(jnp.all(jnp.isfinite(outs)))

    def test_vmap_different_inputs_differ(self, key):
        """Different batch items through EncoderLayer produce different outputs."""
        N, n = 3, 4
        layer = EncoderLayer(n_inputs=N, key=key)
        k1, k2 = jax.random.split(key)
        xs_b = jnp.stack([
            jax.random.normal(k1, (N, 1, n, n)),
            jax.random.normal(k2, (N, 1, n, n)),
        ])
        flags = jnp.ones(N, dtype=bool)

        outs, _ = jax.vmap(lambda xs: layer(xs, flags))(xs_b)

        assert not bool(jnp.allclose(outs[0], outs[1])), \
            "Different inputs must produce different EncoderLayer outputs"

    def test_vmap_inactive_stack_zero(self, key):
        """Inactive encoder flag zeroes the corresponding 64 channel block."""
        N, n = 3, 4
        layer = EncoderLayer(n_inputs=N, key=key)
        B = 4
        xs_b  = jax.random.normal(key, (B, N, 1, n, n))
        # Deactivate stack 1 for all batch items
        flags = jnp.array([True, False, True])

        outs, acts = jax.vmap(lambda xs: layer(xs, flags))(xs_b)

        # Channels 64-127 (stack 1) must be zero for every batch item
        assert bool(jnp.all(outs[:, 64:128] == 0.0)), \
            "Inactive stack 1 must produce zero channels across whole batch"
        assert bool(jnp.all(~acts[:, 64:128])), \
            "Inactive stack 1 flags must be False across whole batch"


# ══════════════════════════════════════════════════════════════════════════════
# 3.  FULL MODEL VMAP BATCHED FORWARD
# ══════════════════════════════════════════════════════════════════════════════

class TestModelVmapBatchedForward:

    def test_vmap_output_shape(self, key):
        """jax.vmap over Model.forward → shape (B, fc_out_features)."""
        model = _sudoku_model(key)
        B = 6
        xs_b = jax.random.normal(key, (B, 5, 1, 4, 4))

        outs = jax.vmap(lambda xs: model(0, xs))(xs_b)

        assert outs.shape == (B, 64), f"Expected ({B},64), got {outs.shape}"

    def test_vmap_outputs_finite(self, key):
        """Every output value in the batch must be finite."""
        model = _sudoku_model(key)
        B = 8
        xs_b = jax.random.normal(key, (B, 5, 1, 4, 4))

        outs = jax.vmap(lambda xs: model(0, xs))(xs_b)

        assert bool(jnp.all(jnp.isfinite(outs))), \
            f"Non-finite outputs: {outs[~jnp.isfinite(outs)]}"

    def test_vmap_different_inputs_differ(self, key):
        """Two distinct batch items must produce different model outputs."""
        model = _sudoku_model(key)
        k1, k2 = jax.random.split(key)
        xs_b = jnp.stack([
            jax.random.normal(k1, (5, 1, 4, 4)),
            jax.random.normal(k2, (5, 1, 4, 4)),
        ])

        outs = jax.vmap(lambda xs: model(0, xs))(xs_b)

        assert not bool(jnp.allclose(outs[0], outs[1])), \
            "Different inputs must produce different model outputs"

    def test_vmap_model_jit_compiles(self, key):
        """eqx.filter_jit(jax.vmap(model forward)) compiles and runs cleanly."""
        model = _sudoku_model(key)
        B = 8
        xs_b = jax.random.normal(key, (B, 5, 1, 4, 4))

        batched_fwd = eqx.filter_jit(jax.vmap(lambda xs: model(0, xs)))
        outs = batched_fwd(xs_b)

        assert outs.shape == (B, 64)
        assert bool(jnp.all(jnp.isfinite(outs)))

    def test_eqx_filter_vmap_model_shapes(self, key):
        """eqx.filter_vmap(model, in_axes=(None,0)) gives correct output shape."""
        model = _sudoku_model(key)
        B = 4
        xs_b = jax.random.normal(key, (B, 5, 1, 4, 4))

        batched = eqx.filter_vmap(lambda m, x: m(0, x), in_axes=(None, 0))
        outs = batched(model, xs_b)

        assert outs.shape == (B, 64)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  BATCHED TRAIN STEP (train.py)
# ══════════════════════════════════════════════════════════════════════════════

class TestBatchedTrainStep:
    """Tests for train_step in train.py using jax.vmap over B."""

    def _build(self, key):
        from train import build_model, build_optimiser
        model = build_model(key)
        cfg   = TrainingConfig(batch_size=8, n_epochs=1, learning_rate=1e-3,
                               grad_clip_norm=1.0)
        tx, opt_state = build_optimiser(cfg, model)
        return model, tx, opt_state

    def test_train_step_returns_scalar_loss(self, key):
        """train_step must return a finite scalar loss."""
        from train import train_step
        model, tx, opt_state = self._build(key)
        xs_b, y_b = _random_batch(key, B=8)

        _, _, loss = train_step(model, opt_state, tx, xs_b, y_b)

        assert loss.shape == (), f"Loss must be scalar, got {loss.shape}"
        assert bool(jnp.isfinite(loss)), f"Loss not finite: {loss}"

    def test_train_step_updates_encoder_weights(self, key):
        """Encoder stage-1 conv weights must change after one batched step."""
        from train import train_step
        model, tx, opt_state = self._build(key)
        xs_b, y_b = _random_batch(key, B=8)

        w_before = model.encoder_layer.stage1_encs.conv1.weight.copy()
        model, _, _ = train_step(model, opt_state, tx, xs_b, y_b)
        w_after = model.encoder_layer.stage1_encs.conv1.weight

        assert not bool(jnp.array_equal(w_before, w_after)), \
            "Encoder weights must change after a batched train step"

    def test_train_step_updates_fc_weights(self, key):
        """FC head weights must change after one batched step."""
        from train import train_step
        model, tx, opt_state = self._build(key)
        xs_b, y_b = _random_batch(key, B=8)

        w_before = model.fc_heads[0].linear.weight.copy()
        model, _, _ = train_step(model, opt_state, tx, xs_b, y_b)
        w_after = model.fc_heads[0].linear.weight

        assert not bool(jnp.array_equal(w_before, w_after)), \
            "FC head weights must change after a batched train step"

    def test_train_step_second_call_no_recompile(self, key):
        """Second JIT call with same shapes must not retrace (no crash)."""
        from train import train_step
        model, tx, opt_state = self._build(key)
        xs_b, y_b = _random_batch(key, B=8)

        model, opt_state, l1 = train_step(model, opt_state, tx, xs_b, y_b)
        xs_b2, y_b2 = _random_batch(jax.random.key(999), B=8)
        model, opt_state, l2 = train_step(model, opt_state, tx, xs_b2, y_b2)

        assert bool(jnp.isfinite(l1)) and bool(jnp.isfinite(l2))

    def test_vmap_loss_equals_mean_individual_losses(self, key):
        """jax.vmap batch loss == mean of per-sample losses (mathematical check)."""
        from train import _single_loss, build_model
        model = build_model(key)
        B = 4
        xs_b, y_b = _random_batch(key, B=B)

        # Batched loss via vmap
        @eqx.filter_jit
        def vmap_loss(m):
            per = jax.vmap(lambda xs, y: _single_loss(m, xs, y))(xs_b, y_b)
            return jnp.mean(per)

        # Per-sample losses via Python loop, then average
        @eqx.filter_jit
        def single_loss_i(m, i):
            return _single_loss(m, xs_b[i], y_b[i])

        individual = jnp.array([float(single_loss_i(model, i)) for i in range(B)])
        mean_indiv = float(jnp.mean(individual))
        batch_l    = float(vmap_loss(model))

        assert abs(batch_l - mean_indiv) < 1e-5, \
            f"Batched loss {batch_l:.6f} != mean individual {mean_indiv:.6f}"

    def test_loss_decreases_over_batched_steps(self, key):
        """5 batched train steps on a fixed batch should decrease loss."""
        from train import train_step, build_model, build_optimiser
        model = build_model(key)
        cfg   = TrainingConfig(batch_size=16, n_epochs=1, learning_rate=3e-3,
                               grad_clip_norm=1.0)
        tx, opt_state = build_optimiser(cfg, model)

        # Fixed batch for overfitting
        xs_b, y_b = _random_batch(jax.random.key(7), B=16)

        losses = []
        for _ in range(5):
            model, opt_state, loss = train_step(model, opt_state, tx, xs_b, y_b)
            losses.append(float(loss))

        assert losses[-1] < losses[0], \
            f"Loss did not decrease: {losses[0]:.6f} → {losses[-1]:.6f}"

    def test_no_nan_inf_over_10_steps(self, key):
        """10 batched train steps must produce no NaN or Inf."""
        from train import train_step, build_model, build_optimiser
        model = build_model(key)
        cfg   = TrainingConfig(batch_size=8, n_epochs=1, learning_rate=3e-4)
        tx, opt_state = build_optimiser(cfg, model)

        rng = np.random.default_rng(0)
        for i in range(10):
            xs_b, y_b = _random_batch(jax.random.key(i), B=8)
            model, opt_state, loss = train_step(model, opt_state, tx, xs_b, y_b)
            assert bool(jnp.isfinite(loss)), f"NaN/Inf loss at step {i}: {loss}"

    def test_larger_batch_no_crash(self, key):
        """Batch size 32 must compile and produce finite loss."""
        from train import train_step, build_model, build_optimiser
        model = build_model(key)
        cfg   = TrainingConfig(batch_size=32, n_epochs=1, learning_rate=3e-4,
                               grad_clip_norm=1.0)
        tx, opt_state = build_optimiser(cfg, model)
        xs_b, y_b = _random_batch(key, B=32)

        _, _, loss = train_step(model, opt_state, tx, xs_b, y_b)

        assert bool(jnp.isfinite(loss)), f"Batch-32 loss not finite: {loss}"


# ══════════════════════════════════════════════════════════════════════════════
# 5.  GRADIENT CORRECTNESS UNDER VMAP
# ══════════════════════════════════════════════════════════════════════════════

class TestBatchedGradientCorrectness:

    def test_encoder_weight_grad_nonzero_for_active_batch(self, key):
        """Active batch items must produce non-zero encoder weight gradients."""
        from train import _single_loss, build_model
        model = build_model(key)
        B = 4
        xs_b, y_b = _random_batch(key, B=B)

        @eqx.filter_jit
        def compute_grad(m):
            def loss(m_):
                per = jax.vmap(lambda xs, y: _single_loss(m_, xs, y))(xs_b, y_b)
                return jnp.mean(per)
            return eqx.filter_grad(loss)(m)

        grads = compute_grad(model)
        w_grad = grads.encoder_layer.stage1_encs.conv1.weight
        assert bool(jnp.any(w_grad != 0.0)), \
            "Encoder weight gradient must be non-zero for active-batch training"

    def test_fc_weight_grad_nonzero(self, key):
        """FC head weight gradient must be non-zero in the batched path."""
        from train import _single_loss, build_model
        model = build_model(key)
        B = 4
        xs_b, y_b = _random_batch(key, B=B)

        @eqx.filter_jit
        def compute_grad(m):
            def loss(m_):
                per = jax.vmap(lambda xs, y: _single_loss(m_, xs, y))(xs_b, y_b)
                return jnp.mean(per)
            return eqx.filter_grad(loss)(m)

        grads = compute_grad(model)
        fc_grad = grads.fc_heads[0].linear.weight
        assert bool(jnp.any(fc_grad != 0.0)), \
            "FC weight gradient must be non-zero in batched training"

    def test_grad_finite_for_all_leaves(self, key):
        """All gradient leaves must be finite after a batched loss computation."""
        from train import _single_loss, build_model
        model = build_model(key)
        B = 8
        xs_b, y_b = _random_batch(key, B=B)

        @eqx.filter_jit
        def compute_grad(m):
            def loss(m_):
                per = jax.vmap(lambda xs, y: _single_loss(m_, xs, y))(xs_b, y_b)
                return jnp.mean(per)
            return eqx.filter_grad(loss)(m)

        grads = compute_grad(model)
        for leaf in jax.tree_util.tree_leaves(
            eqx.filter(grads, eqx.is_inexact_array)
        ):
            assert bool(jnp.all(jnp.isfinite(leaf))), \
                f"Non-finite gradient leaf: {leaf}"
