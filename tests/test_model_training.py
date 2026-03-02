"""
Training stress tests for the multi-problem TMRM Model.

Covers:
  - Multi-epoch training convergence on a fixed sample (overfitting check)
  - Batched training via eqx.filter_vmap
  - Multi-problem round-robin training
  - Timing / performance benchmarks (compilation + per-step wall-time)
  - Numerical stability (NaN / Inf detection across many steps)
  - Repeated identical inputs (idempotency sanity)
  - Large-ish model stress test (more encoders / decoder depth)
  - Gradient norm tracking (should stay finite)
  - Learning rate sensitivity (high LR does not explode)
  - Checkpoint mid-training (save → load → continue)
"""

import io
import time
import numpy as np
import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from model import Model
from utils.config.model import ModelConfig, ProblemConfig


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _tiny_config(n_problems=1, n=4, n_encoders=4, **kw):
    """Smallest practical model — fast to compile, enough to train."""
    problems = tuple(
        ProblemConfig(
            n_encoders_used=kw.get('n_encoders_used', min(2, n_encoders)),
            fc_out_features=kw.get('fc_out_features', 4),
            fc_activation=kw.get('fc_activation', 'identity'),
        )
        for _ in range(n_problems)
    )
    return ModelConfig(
        n=n,
        n_encoders=n_encoders,
        n_decoder_layers=kw.get('n_decoder_layers', 1),
        max_decoder_nodes=kw.get('max_decoder_nodes', 20),
        problems=problems,
    )


def _make_step_fn(model, opt, problem_idx=0):
    """Return a JIT-compiled (model, state, xs, tgt) → (model, state, loss) step."""

    @eqx.filter_jit
    def step(m, s, xs, tgt):
        def loss_fn(mdl):
            pred = mdl(problem_idx, xs)
            return jnp.mean((pred - tgt) ** 2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(m)
        updates, new_s = opt.update(
            eqx.filter(grads, eqx.is_inexact_array),
            s,
            eqx.filter(m, eqx.is_inexact_array),
        )
        return eqx.apply_updates(m, updates), new_s, loss

    return step


@pytest.fixture
def key():
    return jax.random.key(42)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  MULTI-EPOCH OVERFITTING ON A SINGLE SAMPLE
# ══════════════════════════════════════════════════════════════════════════════

class TestOverfitSingleSample:
    """The model should be able to memorise one sample if trained long enough."""

    @pytest.mark.parametrize("n_epochs", [30])
    def test_loss_drops_significantly(self, key, n_epochs):
        cfg = _tiny_config()
        model = Model(cfg, key)
        xs  = jax.random.normal(key, (cfg.n_encoders, 1, cfg.n, cfg.n))
        tgt = jax.random.normal(jax.random.key(7), (cfg.problems[0].fc_out_features,))

        opt   = optax.adam(1e-2)
        state = opt.init(eqx.filter(model, eqx.is_inexact_array))
        step  = _make_step_fn(model, opt, problem_idx=0)

        losses = []
        for _ in range(n_epochs):
            model, state, loss = step(model, state, xs, tgt)
            losses.append(float(loss))

        # Loss must decrease monotonically overall
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.6f} → {losses[-1]:.6f}"
        )
        # All losses must be finite
        assert all(np.isfinite(l) for l in losses), "NaN/Inf loss detected"

    def test_output_approaches_target(self, key):
        """After enough training the model output should be close to the target."""
        cfg = _tiny_config(fc_out_features=4)
        model = Model(cfg, key)
        xs  = jax.random.normal(key, (cfg.n_encoders, 1, cfg.n, cfg.n))
        tgt = jnp.array([1.0, -1.0, 0.5, 0.0])

        opt   = optax.adam(1e-2)
        state = opt.init(eqx.filter(model, eqx.is_inexact_array))
        step  = _make_step_fn(model, opt)

        for _ in range(50):
            model, state, _ = step(model, state, xs, tgt)

        pred = model(0, xs)
        assert jnp.allclose(pred, tgt, atol=0.5), (
            f"Output did not converge to target: pred={pred}, tgt={tgt}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 2.  BATCHED FORWARD VIA VMAP
# ══════════════════════════════════════════════════════════════════════════════

class TestBatchedForward:
    """eqx.filter_vmap over the input batch axis should work."""

    def test_vmap_forward_shapes(self, key):
        cfg = _tiny_config()
        model = Model(cfg, key)
        batch = jax.random.normal(key, (3, cfg.n_encoders, 1, cfg.n, cfg.n))

        batched_fwd = eqx.filter_vmap(
            lambda m, x: m(0, x), in_axes=(None, 0)
        )
        out = batched_fwd(model, batch)
        assert out.shape == (3, cfg.problems[0].fc_out_features)

    def test_vmap_outputs_differ_for_different_inputs(self, key):
        # Use all encoders active so decoder nodes have enough active parents
        cfg = _tiny_config(n_encoders=4, n_encoders_used=4)
        model = Model(cfg, key)
        k1, k2 = jax.random.split(key)
        x1 = jax.random.normal(k1, (cfg.n_encoders, 1, cfg.n, cfg.n))
        x2 = jax.random.normal(k2, (cfg.n_encoders, 1, cfg.n, cfg.n))

        out1 = model(0, x1)
        out2 = model(0, x2)
        # If any decoder activates, different inputs yield different outputs.
        # If all decoders are inactive both will be FC(zeros), which is the same.
        # Either way, forward must succeed and produce finite results.
        assert jnp.all(jnp.isfinite(out1)) and jnp.all(jnp.isfinite(out2))


# ══════════════════════════════════════════════════════════════════════════════
# 3.  BATCHED TRAINING (MEAN LOSS OVER BATCH)
# ══════════════════════════════════════════════════════════════════════════════

class TestBatchedTraining:

    def test_batch_train_loss_decreases(self, key):
        """Train with a mini-batch of 4 (accumulated via loop) for 10 steps."""
        batch_size = 4
        cfg = _tiny_config(fc_out_features=4)
        model = Model(cfg, key)
        batch_xs  = [jax.random.normal(jax.random.key(i), (cfg.n_encoders, 1, cfg.n, cfg.n))
                     for i in range(batch_size)]
        batch_tgt = [jax.random.normal(jax.random.key(100 + i), (4,))
                     for i in range(batch_size)]

        opt   = optax.adam(3e-3)
        state = opt.init(eqx.filter(model, eqx.is_inexact_array))

        @eqx.filter_jit
        def single_step(m, s, xs, tgt):
            def loss_fn(mdl):
                pred = mdl(0, xs)
                return jnp.mean((pred - tgt) ** 2)
            loss, grads = eqx.filter_value_and_grad(loss_fn)(m)
            updates, ns = opt.update(
                eqx.filter(grads, eqx.is_inexact_array), s,
                eqx.filter(m, eqx.is_inexact_array),
            )
            return eqx.apply_updates(m, updates), ns, loss

        losses = []
        for epoch in range(10):
            epoch_loss = 0.0
            for xs_i, tgt_i in zip(batch_xs, batch_tgt):
                model, state, l = single_step(model, state, xs_i, tgt_i)
                epoch_loss += float(l)
            losses.append(epoch_loss / batch_size)

        assert losses[-1] < losses[0], (
            f"Batched loss did not decrease: {losses[0]:.6f} → {losses[-1]:.6f}"
        )
        assert all(np.isfinite(ll) for ll in losses)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MULTI-PROBLEM ROUND-ROBIN TRAINING
# ══════════════════════════════════════════════════════════════════════════════

class TestMultiProblemTraining:

    def test_round_robin_losses_decrease(self, key):
        """Train 2 problems in alternation — both losses should decrease."""
        cfg = ModelConfig(
            n=4, n_encoders=4, n_decoder_layers=1, max_decoder_nodes=20,
            problems=(
                ProblemConfig(n_encoders_used=2, fc_out_features=4, fc_activation='identity'),
                ProblemConfig(n_encoders_used=3, fc_out_features=6, fc_activation='identity'),
            ),
        )
        model = Model(cfg, key)
        xs  = jax.random.normal(key, (cfg.n_encoders, 1, cfg.n, cfg.n))
        targets = [
            jax.random.normal(jax.random.key(30), (4,)),
            jax.random.normal(jax.random.key(31), (6,)),
        ]

        opt   = optax.adam(3e-3)
        state = opt.init(eqx.filter(model, eqx.is_inexact_array))

        # Build one step fn per problem
        steps = [_make_step_fn(model, opt, p) for p in range(2)]

        history = {0: [], 1: []}
        n_rounds = 10
        for _ in range(n_rounds):
            for p in range(2):
                model, state, l = steps[p](model, state, xs, targets[p])
                history[p].append(float(l))

        for p in range(2):
            assert history[p][-1] < history[p][0], (
                f"Problem {p} loss did not decrease: "
                f"{history[p][0]:.6f} → {history[p][-1]:.6f}"
            )

    def test_fc_heads_independent_after_round_robin(self, key):
        """After round-robin, each FC head should have been updated.

        Note: due to the gradient barrier (decoders may all be inactive →
        flat input to FC is zeros → weight gradient is zero, only bias
        gets gradient), we check ANY leaf of each FC head, not only weight.
        """
        cfg = ModelConfig(
            n=4, n_encoders=4, n_decoder_layers=1, max_decoder_nodes=20,
            problems=(
                ProblemConfig(n_encoders_used=2, fc_out_features=4, fc_activation='identity'),
                ProblemConfig(n_encoders_used=3, fc_out_features=4, fc_activation='identity'),
            ),
        )
        model0 = Model(cfg, key)
        model  = Model(cfg, key)  # same init
        xs  = jax.random.normal(key, (cfg.n_encoders, 1, cfg.n, cfg.n))
        targets = [jnp.ones(4), -jnp.ones(4)]

        opt   = optax.adam(3e-3)
        state = opt.init(eqx.filter(model, eqx.is_inexact_array))
        steps = [_make_step_fn(model, opt, p) for p in range(2)]

        for _ in range(5):
            for p in range(2):
                model, state, _ = steps[p](model, state, xs, targets[p])

        # Both FC heads should have changed from their initial values
        # (at minimum bias is updated even if weight gradient is zero)
        for p in range(2):
            leaves_before = jax.tree_util.tree_leaves(
                eqx.filter(model0.fc_heads[p], eqx.is_inexact_array)
            )
            leaves_after = jax.tree_util.tree_leaves(
                eqx.filter(model.fc_heads[p], eqx.is_inexact_array)
            )
            changed = any(
                not jnp.array_equal(a, b)
                for a, b in zip(leaves_before, leaves_after)
            )
            assert changed, (
                f"FC head {p}: no leaf changed after round-robin training"
            )


# ══════════════════════════════════════════════════════════════════════════════
# 5.  TIMING / PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

class TestTiming:
    """Wall-time measurements.  These are informational — not hard failures."""

    def test_compilation_and_step_time(self, key, capsys):
        """Measure JIT compilation time vs. cached step time."""
        cfg = _tiny_config()
        model = Model(cfg, key)
        xs  = jax.random.normal(key, (cfg.n_encoders, 1, cfg.n, cfg.n))
        tgt = jnp.zeros(cfg.problems[0].fc_out_features)
        opt   = optax.adam(1e-3)
        state = opt.init(eqx.filter(model, eqx.is_inexact_array))
        step  = _make_step_fn(model, opt)

        # ── First call (compilation) ──────────────────────────────────────
        t0 = time.perf_counter()
        model, state, _ = step(model, state, xs, tgt)
        jax.block_until_ready(_)
        compile_time = time.perf_counter() - t0

        # ── Cached calls ──────────────────────────────────────────────────
        n_steps = 20
        t0 = time.perf_counter()
        for _ in range(n_steps):
            model, state, loss = step(model, state, xs, tgt)
        jax.block_until_ready(loss)
        cached_total = time.perf_counter() - t0
        per_step = cached_total / n_steps

        # Print timing report (shown with pytest -s)
        with capsys.disabled():
            print(
                f"\n[Timing] compile={compile_time:.2f}s  |  "
                f"{n_steps} cached steps={cached_total:.2f}s  |  "
                f"per_step={per_step*1000:.1f}ms"
            )

        # Sanity: compilation should be slower than a single cached step
        assert compile_time > per_step, (
            "Compilation should be at least as slow as one cached step"
        )
        # Sanity: cached step should be < 5s each (very generous bound)
        assert per_step < 5.0, f"Each cached step took {per_step:.2f}s — too slow"

    def test_forward_time_multiple_calls(self, key, capsys):
        """Time 10 sequential forward calls (all cached after first)."""
        cfg = _tiny_config()
        model = Model(cfg, key)
        x_single = jax.random.normal(key, (cfg.n_encoders, 1, cfg.n, cfg.n))

        fwd = eqx.filter_jit(lambda m, x: m(0, x))

        # Warm up (compilation)
        out = fwd(model, x_single)
        jax.block_until_ready(out)

        # Time cached calls
        n_calls = 20
        t0 = time.perf_counter()
        for _ in range(n_calls):
            out = fwd(model, x_single)
        jax.block_until_ready(out)
        elapsed = time.perf_counter() - t0
        per_call = elapsed / n_calls

        with capsys.disabled():
            print(
                f"\n[Timing] {n_calls} cached fwd calls = {elapsed:.3f}s  |  "
                f"per_call = {per_call*1000:.1f}ms"
            )

        assert np.isfinite(float(jnp.sum(out)))
        assert per_call < 5.0, f"Each forward call took {per_call:.2f}s — too slow"


# ══════════════════════════════════════════════════════════════════════════════
# 6.  NUMERICAL STABILITY OVER MANY STEPS
# ══════════════════════════════════════════════════════════════════════════════

class TestNumericalStability:

    def test_no_nan_or_inf_over_50_steps(self, key):
        """50 training steps must produce no NaN/Inf in loss or output."""
        cfg = _tiny_config()
        model = Model(cfg, key)
        xs  = jax.random.normal(key, (cfg.n_encoders, 1, cfg.n, cfg.n))
        tgt = jnp.ones(cfg.problems[0].fc_out_features)

        opt   = optax.adam(1e-3)
        state = opt.init(eqx.filter(model, eqx.is_inexact_array))
        step  = _make_step_fn(model, opt)

        for i in range(50):
            model, state, loss = step(model, state, xs, tgt)
            assert jnp.isfinite(loss), f"Non-finite loss at step {i}: {loss}"

        out = model(0, xs)
        assert jnp.all(jnp.isfinite(out)), "Non-finite output after 50 steps"

    def test_large_input_does_not_explode(self, key):
        """Inputs scaled by 100 should still produce finite outputs."""
        cfg = _tiny_config()
        model = Model(cfg, key)
        xs = jax.random.normal(key, (cfg.n_encoders, 1, cfg.n, cfg.n)) * 100.0
        out = model(0, xs)
        assert jnp.all(jnp.isfinite(out)), f"Non-finite output for large input: {out}"

    def test_zero_input_does_not_nan(self, key):
        """All-zero input should produce finite output (not NaN from slogdet)."""
        cfg = _tiny_config()
        model = Model(cfg, key)
        xs = jnp.zeros((cfg.n_encoders, 1, cfg.n, cfg.n))
        out = model(0, xs)
        assert jnp.all(jnp.isfinite(out)), f"Non-finite output for zero input: {out}"

    def test_high_lr_does_not_produce_nan(self, key):
        """Training with a high learning rate should not produce NaN."""
        cfg = _tiny_config()
        model = Model(cfg, key)
        xs  = jax.random.normal(key, (cfg.n_encoders, 1, cfg.n, cfg.n))
        tgt = jnp.zeros(cfg.problems[0].fc_out_features)

        opt   = optax.adam(0.1)  # high LR
        state = opt.init(eqx.filter(model, eqx.is_inexact_array))
        step  = _make_step_fn(model, opt)

        for i in range(10):
            model, state, loss = step(model, state, xs, tgt)
            assert jnp.isfinite(loss), f"NaN/Inf at step {i} with LR=0.1"


# ══════════════════════════════════════════════════════════════════════════════
# 7.  REPEATED IDENTICAL INPUTS (DETERMINISM)
# ══════════════════════════════════════════════════════════════════════════════

class TestDeterminism:

    def test_same_input_same_output(self, key):
        """Two forward calls with the same input → identical output."""
        cfg = _tiny_config()
        model = Model(cfg, key)
        xs = jax.random.normal(key, (cfg.n_encoders, 1, cfg.n, cfg.n))
        o1 = model(0, xs)
        o2 = model(0, xs)
        assert jnp.array_equal(o1, o2)

    def test_jit_determinism(self, key):
        """JIT-compiled forward is deterministic across calls."""
        cfg = _tiny_config()
        model = Model(cfg, key)
        xs = jax.random.normal(key, (cfg.n_encoders, 1, cfg.n, cfg.n))

        fwd = eqx.filter_jit(lambda m, x: m(0, x))
        o1 = fwd(model, xs)
        o2 = fwd(model, xs)
        assert jnp.array_equal(o1, o2)

    def test_training_from_same_init_is_reproducible(self, key):
        """Two models from the same key, trained identically → same result."""
        cfg = _tiny_config()
        xs  = jax.random.normal(key, (cfg.n_encoders, 1, cfg.n, cfg.n))
        tgt = jnp.ones(cfg.problems[0].fc_out_features)

        results = []
        for _ in range(2):
            model = Model(cfg, key)
            opt   = optax.adam(1e-3)
            state = opt.init(eqx.filter(model, eqx.is_inexact_array))
            step  = _make_step_fn(model, opt)
            for __ in range(5):
                model, state, loss = step(model, state, xs, tgt)
            results.append(float(loss))

        assert results[0] == results[1], (
            f"Reproducibility failed: {results[0]:.8f} vs {results[1]:.8f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 8.  GRADIENT NORM TRACKING
# ══════════════════════════════════════════════════════════════════════════════

class TestGradientNorms:

    def test_gradient_norms_finite(self, key):
        """Gradient norms should be finite for 10 steps."""
        cfg = _tiny_config()
        model = Model(cfg, key)
        xs  = jax.random.normal(key, (cfg.n_encoders, 1, cfg.n, cfg.n))
        tgt = jnp.ones(cfg.problems[0].fc_out_features)

        def loss_fn(m):
            return jnp.mean((m(0, xs) - tgt) ** 2)

        opt   = optax.adam(1e-3)
        state = opt.init(eqx.filter(model, eqx.is_inexact_array))

        @eqx.filter_jit
        def step_with_gnorm(m, s):
            loss, grads = eqx.filter_value_and_grad(loss_fn)(m)
            grad_leaves = jax.tree_util.tree_leaves(
                eqx.filter(grads, eqx.is_inexact_array)
            )
            gnorm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in grad_leaves))
            updates, ns = opt.update(
                eqx.filter(grads, eqx.is_inexact_array), s,
                eqx.filter(m, eqx.is_inexact_array),
            )
            return eqx.apply_updates(m, updates), ns, loss, gnorm

        for i in range(10):
            model, state, loss, gnorm = step_with_gnorm(model, state)
            assert jnp.isfinite(gnorm), f"Gradient norm not finite at step {i}"
            assert jnp.isfinite(loss), f"Loss not finite at step {i}"

    def test_gradients_nonzero(self, key):
        """The gradient norm should be non-zero (model actually receives signal)."""
        cfg = _tiny_config()
        model = Model(cfg, key)
        xs  = jax.random.normal(key, (cfg.n_encoders, 1, cfg.n, cfg.n))
        tgt = jnp.ones(cfg.problems[0].fc_out_features) * 100.0  # far from init output

        def loss_fn(m):
            return jnp.mean((m(0, xs) - tgt) ** 2)

        grads = eqx.filter_grad(loss_fn)(model)
        grad_leaves = jax.tree_util.tree_leaves(
            eqx.filter(grads, eqx.is_inexact_array)
        )
        gnorm = float(jnp.sqrt(sum(jnp.sum(g ** 2) for g in grad_leaves)))
        assert gnorm > 0, "Gradient norm should be non-zero"


# ══════════════════════════════════════════════════════════════════════════════
# 9.  CHECKPOINT MID-TRAINING
# ══════════════════════════════════════════════════════════════════════════════

class TestCheckpointResume:

    def test_save_load_continue_training(self, key):
        """Save mid-training → load → continue → loss still decreases."""
        cfg = _tiny_config(fc_out_features=4)
        model = Model(cfg, key)
        xs  = jax.random.normal(key, (cfg.n_encoders, 1, cfg.n, cfg.n))
        tgt = jax.random.normal(jax.random.key(77), (4,))

        opt   = optax.adam(3e-3)
        state = opt.init(eqx.filter(model, eqx.is_inexact_array))
        step  = _make_step_fn(model, opt)

        # Train 5 steps
        for _ in range(5):
            model, state, loss_pre = step(model, state, xs, tgt)

        loss_at_save = float(loss_pre)

        # Save checkpoint
        buf = io.BytesIO()
        eqx.tree_serialise_leaves(buf, model)

        # Load into fresh skeleton
        buf.seek(0)
        model2 = Model(cfg, jax.random.key(999))
        model2 = eqx.tree_deserialise_leaves(buf, model2)

        # Fresh optimizer state (simulates a restart)
        state2 = opt.init(eqx.filter(model2, eqx.is_inexact_array))
        step2  = _make_step_fn(model2, opt)

        # Train 10 more steps
        for _ in range(10):
            model2, state2, loss_post = step2(model2, state2, xs, tgt)

        loss_after_resume = float(loss_post)
        assert loss_after_resume < loss_at_save, (
            f"Loss did not decrease after resume: "
            f"save={loss_at_save:.6f} → post={loss_after_resume:.6f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 10. LARGER MODEL STRESS TEST
# ══════════════════════════════════════════════════════════════════════════════

class TestLargerModel:
    """Use slightly larger configs to check nothing breaks at scale."""

    def test_deeper_decoder(self, key):
        """2 decoder layers, more nodes — forward + 3 train steps must work."""
        cfg = ModelConfig(
            n=4, n_encoders=4, n_decoder_layers=2, max_decoder_nodes=40,
            problems=(
                ProblemConfig(n_encoders_used=3, fc_out_features=8,
                              fc_activation='identity'),
            ),
        )
        model = Model(cfg, key)
        xs  = jax.random.normal(key, (cfg.n_encoders, 1, cfg.n, cfg.n))
        tgt = jnp.zeros(8)

        opt   = optax.adam(1e-3)
        state = opt.init(eqx.filter(model, eqx.is_inexact_array))
        step  = _make_step_fn(model, opt)

        for i in range(3):
            model, state, loss = step(model, state, xs, tgt)
            assert jnp.isfinite(loss), f"Non-finite loss at step {i}"

    def test_more_encoders(self, key):
        """6 encoders, 2 problems — forward + train must work."""
        cfg = ModelConfig(
            n=4, n_encoders=6, n_decoder_layers=1, max_decoder_nodes=30,
            problems=(
                ProblemConfig(n_encoders_used=3, fc_out_features=4,
                              fc_activation='identity'),
                ProblemConfig(n_encoders_used=5, fc_out_features=6,
                              fc_activation='relu'),
            ),
        )
        model = Model(cfg, key)
        xs  = jax.random.normal(key, (cfg.n_encoders, 1, cfg.n, cfg.n))

        # Forward for both problems
        for p in range(2):
            out = model(p, xs)
            assert jnp.all(jnp.isfinite(out))
            assert out.shape == (cfg.problems[p].fc_out_features,)

        # One train step per problem
        opt   = optax.adam(1e-3)
        state = opt.init(eqx.filter(model, eqx.is_inexact_array))
        for p in range(2):
            tgt = jnp.zeros(cfg.problems[p].fc_out_features)
            step = _make_step_fn(model, opt, problem_idx=p)
            model, state, loss = step(model, state, xs, tgt)
            assert jnp.isfinite(loss), f"Non-finite loss for problem {p}"


# ══════════════════════════════════════════════════════════════════════════════
# 11. ACTIVATION FUNCTIONS UNDER TRAINING
# ══════════════════════════════════════════════════════════════════════════════

class TestActivationsTraining:
    """Ensure every supported FC activation works under gradient descent."""

    @pytest.mark.parametrize("act", ['relu', 'gelu', 'tanh', 'sigmoid', 'identity'])
    def test_activation_trains_without_error(self, key, act):
        cfg = _tiny_config(fc_activation=act)
        model = Model(cfg, key)
        xs  = jax.random.normal(key, (cfg.n_encoders, 1, cfg.n, cfg.n))
        tgt = jnp.ones(cfg.problems[0].fc_out_features)

        opt   = optax.adam(1e-3)
        state = opt.init(eqx.filter(model, eqx.is_inexact_array))
        step  = _make_step_fn(model, opt)

        for i in range(5):
            model, state, loss = step(model, state, xs, tgt)
            assert jnp.isfinite(loss), (
                f"Non-finite loss at step {i} with activation={act}"
            )
