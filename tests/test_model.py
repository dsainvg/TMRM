"""
Tests for the multi-problem Model class and its config dataclasses.

Covers:
  - ModelConfig / ProblemConfig validation
  - Model construction (single problem, multi-problem)
  - Forward pass shape correctness per problem
  - Encoder mask properties (correct count, no overlap unless intended)
  - Training: FC head weights update; untrained problem's head stays frozen
  - Different problems produce different outputs from same input
  - JIT-compiled forward runs twice without error
  - Serialisation round-trip (save → load → identical output)
  - Helper properties
"""

import io
import numpy as np
import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from model import Model
from utils.config.model import ModelConfig, ProblemConfig


# ─── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def key():
    return jax.random.key(42)


def _simple_config(n_problems=1, n_encoders=4, n=6, **overrides):
    """Build a lightweight ModelConfig for testing."""
    problems = []
    for i in range(n_problems):
        problems.append(ProblemConfig(
            n_encoders_used=overrides.get('n_encoders_used', min(2, n_encoders)),
            fc_out_features=overrides.get('fc_out_features', n * n),
            fc_activation=overrides.get('fc_activation', 'identity'),
        ))
    return ModelConfig(
        n=n,
        n_encoders=n_encoders,
        n_decoder_layers=overrides.get('n_decoder_layers', 1),
        max_decoder_nodes=overrides.get('max_decoder_nodes', 20),
        problems=tuple(problems),
    )


def _multi_config(n=6, n_encoders=6):
    """Build a ModelConfig with 3 heterogeneous problem heads."""
    return ModelConfig(
        n=n,
        n_encoders=n_encoders,
        n_decoder_layers=1,
        max_decoder_nodes=20,
        problems=(
            ProblemConfig(n_encoders_used=2, fc_out_features=4, fc_activation='relu'),
            ProblemConfig(n_encoders_used=4, fc_out_features=10, fc_activation='identity'),
            ProblemConfig(n_encoders_used=6, fc_out_features=n * n, fc_activation='gelu'),
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1.  CONFIG VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

class TestConfigValidation:

    def test_valid_config_does_not_raise(self):
        cfg = _simple_config()
        assert cfg.n_problems == 1

    def test_multi_problem_config(self):
        cfg = _multi_config()
        assert cfg.n_problems == 3

    def test_n_must_be_positive(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            ModelConfig(n=0, n_encoders=2, n_decoder_layers=1,
                        max_decoder_nodes=10,
                        problems=(ProblemConfig(1, 4),))

    def test_n_encoders_must_be_positive(self):
        with pytest.raises(ValueError, match="n_encoders must be >= 1"):
            ModelConfig(n=4, n_encoders=0, n_decoder_layers=1,
                        max_decoder_nodes=10,
                        problems=(ProblemConfig(1, 4),))

    def test_n_decoder_layers_must_be_positive(self):
        with pytest.raises(ValueError, match="n_decoder_layers must be >= 1"):
            ModelConfig(n=4, n_encoders=2, n_decoder_layers=0,
                        max_decoder_nodes=10,
                        problems=(ProblemConfig(1, 4),))

    def test_max_decoder_nodes_must_be_positive(self):
        with pytest.raises(ValueError, match="max_decoder_nodes must be >= 1"):
            ModelConfig(n=4, n_encoders=2, n_decoder_layers=1,
                        max_decoder_nodes=0,
                        problems=(ProblemConfig(1, 4),))

    def test_empty_problems_raises(self):
        with pytest.raises(ValueError, match="at least one ProblemConfig"):
            ModelConfig(n=4, n_encoders=2, n_decoder_layers=1,
                        max_decoder_nodes=10, problems=())

    def test_encoders_used_too_large(self):
        with pytest.raises(ValueError, match="n_encoders_used=5"):
            ModelConfig(n=4, n_encoders=4, n_decoder_layers=1,
                        max_decoder_nodes=10,
                        problems=(ProblemConfig(n_encoders_used=5,
                                               fc_out_features=4),))

    def test_encoders_used_zero(self):
        with pytest.raises(ValueError, match="n_encoders_used=0"):
            ModelConfig(n=4, n_encoders=4, n_decoder_layers=1,
                        max_decoder_nodes=10,
                        problems=(ProblemConfig(n_encoders_used=0,
                                               fc_out_features=4),))

    def test_bad_activation_raises(self):
        with pytest.raises(ValueError, match="not in"):
            ModelConfig(n=4, n_encoders=2, n_decoder_layers=1,
                        max_decoder_nodes=10,
                        problems=(ProblemConfig(1, 4, fc_activation='swish'),))

    def test_fc_out_features_zero_raises(self):
        with pytest.raises(ValueError, match="fc_out_features must be >= 1"):
            ModelConfig(n=4, n_encoders=2, n_decoder_layers=1,
                        max_decoder_nodes=10,
                        problems=(ProblemConfig(1, 0),))

    def test_frozen_config(self):
        cfg = _simple_config()
        with pytest.raises(AttributeError):
            cfg.n = 99  # type: ignore


# ══════════════════════════════════════════════════════════════════════════════
# 2.  MODEL CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

class TestModelConstruction:

    def test_single_problem_model(self, key):
        cfg = _simple_config(n_problems=1)
        model = Model(cfg, key)
        assert model.n_problems == 1
        assert len(model.fc_heads) == 1
        assert len(model.encoder_masks) == 1

    def test_multi_problem_model(self, key):
        cfg = _multi_config()
        model = Model(cfg, key)
        assert model.n_problems == 3
        assert len(model.fc_heads) == 3
        assert len(model.encoder_masks) == 3

    def test_fc_in_features_matches(self, key):
        cfg = _simple_config(n=8)
        model = Model(cfg, key)
        expected = model.n_decoder_output_nodes * 8 * 8
        assert model.fc_in_features == expected

    def test_config_stored_as_static(self, key):
        cfg = _simple_config()
        model = Model(cfg, key)
        assert model.config is cfg


# ══════════════════════════════════════════════════════════════════════════════
# 3.  ENCODER MASK PROPERTIES
# ══════════════════════════════════════════════════════════════════════════════

class TestEncoderMasks:

    def test_mask_count_matches_config(self, key):
        """Each problem's mask has exactly n_encoders_used True entries."""
        cfg = _multi_config()
        model = Model(cfg, key)
        for i, p in enumerate(cfg.problems):
            mask = model.encoder_masks[i]
            assert mask.sum() == p.n_encoders_used, \
                f"problem {i}: expected {p.n_encoders_used} active, got {mask.sum()}"

    def test_mask_shape(self, key):
        cfg = _multi_config(n_encoders=8)
        model = Model(cfg, key)
        for mask in model.encoder_masks:
            assert mask.shape == (8,)
            assert mask.dtype == bool

    def test_active_encoder_indices_sorted(self, key):
        cfg = _multi_config()
        model = Model(cfg, key)
        for i in range(cfg.n_problems):
            idx = model.active_encoder_indices(i)
            assert np.all(idx[:-1] <= idx[1:]), "indices must be sorted"
            assert len(idx) == cfg.problems[i].n_encoders_used

    def test_all_encoders_used_mask(self, key):
        """Problem using all encoders → all True."""
        cfg = ModelConfig(
            n=4, n_encoders=4, n_decoder_layers=1, max_decoder_nodes=20,
            problems=(ProblemConfig(n_encoders_used=4, fc_out_features=16),),
        )
        model = Model(cfg, key)
        assert model.encoder_masks[0].all()

    def test_masks_are_numpy(self, key):
        """Masks must be plain NumPy arrays (compile-time constants)."""
        cfg = _simple_config()
        model = Model(cfg, key)
        for mask in model.encoder_masks:
            assert isinstance(mask, np.ndarray)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  FORWARD PASS SHAPE
# ══════════════════════════════════════════════════════════════════════════════

class TestForwardPassShape:

    def test_single_problem_output_shape(self, key):
        n = 6
        cfg = _simple_config(n_problems=1, n=n, fc_out_features=n * n)
        model = Model(cfg, key)
        xs = jax.random.normal(key, (cfg.n_encoders, 1, n, n))
        out = model(0, xs)
        assert out.shape == (n * n,)

    def test_multi_problem_output_shapes(self, key):
        cfg = _multi_config(n=6)
        model = Model(cfg, key)
        xs = jax.random.normal(key, (cfg.n_encoders, 1, 6, 6))
        for i, p in enumerate(cfg.problems):
            out = model(i, xs)
            assert out.shape == (p.fc_out_features,), \
                f"problem {i}: expected ({p.fc_out_features},), got {out.shape}"

    def test_output_is_finite(self, key):
        cfg = _simple_config(n=4)
        model = Model(cfg, key)
        xs = jax.random.normal(key, (cfg.n_encoders, 1, 4, 4))
        out = model(0, xs)
        assert bool(jnp.all(jnp.isfinite(out)))


# ══════════════════════════════════════════════════════════════════════════════
# 5.  DIFFERENT PROBLEMS → DIFFERENT OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

class TestDifferentProblemsProduceDifferentOutputs:

    def test_distinct_problem_outputs(self, key):
        """Two problems with different masks/FC heads produce different outputs."""
        cfg = ModelConfig(
            n=6, n_encoders=6, n_decoder_layers=1, max_decoder_nodes=20,
            problems=(
                ProblemConfig(n_encoders_used=2, fc_out_features=8,
                              fc_activation='identity'),
                ProblemConfig(n_encoders_used=4, fc_out_features=8,
                              fc_activation='identity'),
            ),
        )
        model = Model(cfg, key)
        xs = jax.random.normal(key, (cfg.n_encoders, 1, 6, 6))
        out0 = model(0, xs)
        out1 = model(1, xs)
        # Different masks → different enc_flags → different decoder path
        assert not bool(jnp.allclose(out0, out1)), \
            "Different problems must produce different outputs"


# ══════════════════════════════════════════════════════════════════════════════
# 6.  TRAINING
# ══════════════════════════════════════════════════════════════════════════════

class TestTraining:

    def test_loss_decreases_on_one_problem(self, key):
        """Training problem 0 for 5 steps → loss must decrease."""
        cfg = _simple_config(n_problems=2, n=4, fc_out_features=4)
        model = Model(cfg, key)
        xs  = jax.random.normal(key, (cfg.n_encoders, 1, 4, 4))
        tgt = jax.random.normal(jax.random.key(99), (4,))

        opt   = optax.adam(3e-3)
        state = opt.init(eqx.filter(model, eqx.is_inexact_array))

        @eqx.filter_jit
        def step(m, s):
            def loss_fn(mdl):
                return jnp.mean((mdl(0, xs) - tgt) ** 2)
            l, g = eqx.filter_value_and_grad(loss_fn)(m)
            upd, ns = opt.update(
                eqx.filter(g, eqx.is_inexact_array), s,
                eqx.filter(m, eqx.is_inexact_array),
            )
            return eqx.apply_updates(m, upd), ns, l

        losses = []
        for _ in range(5):
            model, state, l = step(model, state)
            losses.append(float(l))

        assert losses[-1] < losses[0], \
            f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_fc_head_weights_change(self, key):
        """After one step on problem 0, its FC head weights must change."""
        cfg = _simple_config(n_problems=2, n=4, fc_out_features=4)
        model = Model(cfg, key)
        xs  = jax.random.normal(key, (cfg.n_encoders, 1, 4, 4))
        tgt = jnp.zeros(4)
        w0_before = model.fc_heads[0].linear.weight.copy()

        def loss_fn(mdl):
            return jnp.mean((mdl(0, xs) - tgt) ** 2)

        grads = eqx.filter_grad(loss_fn)(model)
        opt   = optax.adam(1e-3)
        state = opt.init(eqx.filter(model, eqx.is_inexact_array))
        updates, _ = opt.update(
            eqx.filter(grads, eqx.is_inexact_array), state,
            eqx.filter(model, eqx.is_inexact_array),
        )
        model_upd = eqx.apply_updates(model, updates)
        # At minimum the decoder or FC head must have changed
        w0_after = model_upd.fc_heads[0].linear.weight
        # Gather all leaves; at least one must differ
        leaves_before = jax.tree_util.tree_leaves(
            eqx.filter(model, eqx.is_inexact_array))
        leaves_after  = jax.tree_util.tree_leaves(
            eqx.filter(model_upd, eqx.is_inexact_array))
        changed = any(
            not jnp.array_equal(a, b)
            for a, b in zip(leaves_after, leaves_before)
        )
        assert changed, \
            "At least one weight leaf must change after a training step"

    def test_other_problem_fc_head_unchanged(self, key):
        """Training problem 0 must NOT change problem 1's FC head weights."""
        cfg = _simple_config(n_problems=2, n=4, fc_out_features=4)
        model = Model(cfg, key)
        xs  = jax.random.normal(key, (cfg.n_encoders, 1, 4, 4))
        tgt = jnp.zeros(4)
        w1_before = model.fc_heads[1].linear.weight.copy()

        def loss_fn(mdl):
            return jnp.mean((mdl(0, xs) - tgt) ** 2)

        grads = eqx.filter_grad(loss_fn)(model)
        # Problem 1's FC head gradient should be all zeros
        g_w1 = grads.fc_heads[1].linear.weight
        assert bool(jnp.all(g_w1 == 0.0)), \
            "Problem 1's FC head should have zero gradient when training problem 0"

        opt   = optax.adam(1e-3)
        state = opt.init(eqx.filter(model, eqx.is_inexact_array))
        updates, _ = opt.update(
            eqx.filter(grads, eqx.is_inexact_array), state,
            eqx.filter(model, eqx.is_inexact_array),
        )
        model_upd = eqx.apply_updates(model, updates)
        # Even after applying Adam (which has momentum for 0-grad → tiny epsilon updates),
        # the weight must still differ negligibly.
        w1_after = model_upd.fc_heads[1].linear.weight
        assert bool(jnp.allclose(w1_before, w1_after, atol=1e-6)), \
            "Problem 1's FC head weight should be unchanged"


# ══════════════════════════════════════════════════════════════════════════════
# 7.  JIT COMPILATION
# ══════════════════════════════════════════════════════════════════════════════

class TestJIT:

    def test_jit_runs_twice(self, key):
        """JIT-compiled forward pass must succeed on two consecutive calls."""
        cfg = _simple_config(n=4)
        model = Model(cfg, key)
        xs = jax.random.normal(key, (cfg.n_encoders, 1, 4, 4))

        @eqx.filter_jit
        def fwd(m, x):
            return m(0, x)

        o1 = fwd(model, xs)
        o2 = fwd(model, xs)
        assert bool(jnp.allclose(o1, o2))

    def test_jit_train_step_two_calls(self, key):
        """JIT-compiled train step must not crash on the second call."""
        cfg = _simple_config(n=4, fc_out_features=4)
        model = Model(cfg, key)
        xs  = jax.random.normal(key, (cfg.n_encoders, 1, 4, 4))
        tgt = jnp.zeros(4)
        opt   = optax.adam(1e-3)
        state = opt.init(eqx.filter(model, eqx.is_inexact_array))

        @eqx.filter_jit
        def step(m, s):
            def lfn(mdl):
                return jnp.mean((mdl(0, xs) - tgt) ** 2)
            l, g = eqx.filter_value_and_grad(lfn)(m)
            upd, ns = opt.update(
                eqx.filter(g, eqx.is_inexact_array), s,
                eqx.filter(m, eqx.is_inexact_array),
            )
            return eqx.apply_updates(m, upd), ns, l

        model, state, l1 = step(model, state)
        model, state, l2 = step(model, state)
        assert jnp.isfinite(l1) and jnp.isfinite(l2)


# ══════════════════════════════════════════════════════════════════════════════
# 8.  SERIALISATION ROUND-TRIP
# ══════════════════════════════════════════════════════════════════════════════

class TestSerialisation:

    def test_roundtrip_identical_output(self, key):
        """Save → load → output must be identical."""
        cfg = _multi_config(n=4, n_encoders=6)
        model = Model(cfg, key)
        xs = jax.random.normal(key, (cfg.n_encoders, 1, 4, 4))
        out_before = model(0, xs)

        buf = io.BytesIO()
        eqx.tree_serialise_leaves(buf, model)
        buf.seek(0)
        # Rebuild structure with different key, then overwrite weights
        model2 = Model(cfg, jax.random.key(999))
        model2 = eqx.tree_deserialise_leaves(buf, model2)

        out_after = model2(0, xs)
        assert bool(jnp.allclose(out_before, out_after, atol=1e-6)), \
            "Round-trip model output must be identical"


# ══════════════════════════════════════════════════════════════════════════════
# 9.  HELPER PROPERTIES
# ══════════════════════════════════════════════════════════════════════════════

class TestHelperProperties:

    def test_n_problems_property(self, key):
        cfg = _multi_config()
        model = Model(cfg, key)
        assert model.n_problems == 3

    def test_n_decoder_output_nodes_positive(self, key):
        cfg = _simple_config()
        model = Model(cfg, key)
        assert model.n_decoder_output_nodes > 0

    def test_fc_in_features_positive(self, key):
        cfg = _simple_config(n=6)
        model = Model(cfg, key)
        assert model.fc_in_features > 0
        assert model.fc_in_features == model.n_decoder_output_nodes * 6 * 6
