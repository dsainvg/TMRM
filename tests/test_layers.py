"""
Tests for the three TMRM layer types:
    - EncoderLayer  (N independent 2-encoder stacks)
    - DecoderLayer  (K Decoder nodes with static adjacency wiring)
    - FCLayer       (Linear + activation terminal block)
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from utils.encoder_layer import EncoderLayer
from utils.decoder_layer import DecoderLayer
from utils.fc_layer import FCLayer


# ─── Shared fixture ──────────────────────────────────────────────────────────

@pytest.fixture
def key():
    return jax.random.key(42)


# ─── EncoderLayer ────────────────────────────────────────────────────────────

class TestEncoderLayer:

    def test_output_shape_all_active(self, key):
        """N=2 active inputs -> (128, 8, 8) and 128 True flags."""
        layer = EncoderLayer(n_inputs=2, key=key)
        xs    = jnp.ones((2, 1, 8, 8))
        flags = jnp.array([True, True])

        out, acts = layer(xs, flags)

        assert out.shape == (128, 8, 8)
        assert acts.shape == (128,)
        assert bool(jnp.all(acts))

    def test_output_shape_all_inactive(self, key):
        """All inactive -> (128, 8, 8) zeros, all 128 flags False."""
        layer = EncoderLayer(n_inputs=2, key=key)
        xs    = jnp.ones((2, 1, 8, 8))
        flags = jnp.array([False, False])

        out, acts = layer(xs, flags)

        assert out.shape == (128, 8, 8)
        assert acts.shape == (128,)
        assert bool(jnp.all(~acts))
        assert bool(jnp.all(out == 0.0))

    def test_mixed_active_flags(self, key):
        """Mixed flags: inactive stack channels are zeros; flags reflect stack activity."""
        layer = EncoderLayer(n_inputs=3, key=key)
        xs    = jnp.ones((3, 1, 8, 8))
        flags = jnp.array([True, False, True])

        out, acts = layer(xs, flags)

        # (3*64, 8, 8) = (192, 8, 8)
        assert out.shape == (192, 8, 8)
        assert acts.shape == (192,)
        # First 64 channels: stack 0 active
        assert bool(jnp.all(acts[:64]))
        # Channels 64-127: stack 1 inactive
        assert bool(jnp.all(~acts[64:128]))
        assert bool(jnp.all(out[64:128] == 0.0))
        # Last 64 channels: stack 2 active
        assert bool(jnp.all(acts[128:]))

    def test_jit_compiles(self, key):
        """Layer must compile cleanly under eqx.filter_jit."""
        layer = EncoderLayer(n_inputs=2, key=key)
        xs    = jnp.ones((2, 1, 8, 8))
        flags = jnp.array([True, True])

        out, acts = eqx.filter_jit(layer)(xs, flags)

        assert out.shape == (128, 8, 8)
        assert acts.shape == (128,)

    def test_larger_n_inputs(self, key):
        """Scale test: N=5 inputs, spatial size 16x16 -> (320, 16, 16)."""
        layer = EncoderLayer(n_inputs=5, key=key)
        xs    = jnp.ones((5, 1, 16, 16))
        flags = jnp.ones((5,), dtype=bool)

        out, acts = layer(xs, flags)

        assert out.shape == (320, 16, 16)
        assert acts.shape == (320,)
        assert bool(jnp.all(acts))

    # ── Tree-structure-specific tests ────────────────────────────────────────

    def test_tree_stage2_enc_batch_dims(self, key):
        """stage2_encs must carry a (N, 8) leading batch — not just (N,)."""
        N = 3
        layer = EncoderLayer(n_inputs=N, key=key)
        # Equinox vmaps pytrees: check conv1 weight leading dims are (N, 8, ...)
        w = layer.stage2_encs.conv1.weight
        assert w.shape[0] == N,  f"outer vmap dim: expected {N}, got {w.shape[0]}"
        assert w.shape[1] == 8,  f"inner vmap dim: expected 8 leaves, got {w.shape[1]}"

    def test_tree_nine_nodes_total_per_input(self, key):
        """Each input has exactly 9 Encoder nodes: 1 root + 8 leaves."""
        N = 2
        layer = EncoderLayer(n_inputs=N, key=key)
        # stage1_encs: (N, ...) — N roots
        assert layer.stage1_encs.conv1.weight.shape[0] == N
        # stage2_encs: (N, 8, ...) — N × 8 leaves
        assert layer.stage2_encs.conv1.weight.shape[0] == N
        assert layer.stage2_encs.conv1.weight.shape[1] == 8

    def test_tree_per_leaf_flags(self, key):
        """Active stack produces 8 per-leaf flags, each covering 8 channels."""
        layer = EncoderLayer(n_inputs=1, key=key)
        xs    = jnp.ones((1, 1, 8, 8))
        flags = jnp.array([True])

        out, acts = layer(xs, flags)

        # 1 stack × 64 channels total
        assert out.shape  == (64, 8, 8)
        assert acts.shape == (64,)
        # All 64 flags True (all 8 leaves × 8 channels active)
        assert bool(jnp.all(acts))

    def test_tree_inactive_root_silences_all_leaves(self, key):
        """Inactive root flag zeroes all 8 leaves -> 64 zero channels."""
        layer = EncoderLayer(n_inputs=1, key=key)
        xs    = jnp.ones((1, 1, 8, 8))
        flags = jnp.array([False])

        out, acts = layer(xs, flags)

        assert bool(jnp.all(~acts)),          "All 64 leaf flags must be False"
        assert bool(jnp.all(out == 0.0)),     "All 64 channels must be zero"

    def test_tree_different_inputs_produce_different_outputs(self, key):
        """Two distinct inputs to the same layer must yield distinct outputs."""
        layer = EncoderLayer(n_inputs=2, key=key)
        k1, k2 = jax.random.split(key)
        xs    = jnp.stack([
            jax.random.normal(k1, (1, 8, 8)),
            jax.random.normal(k2, (1, 8, 8)),
        ])
        flags = jnp.array([True, True])

        out, _ = layer(xs, flags)

        # First 64 channels vs last 64 channels must differ
        assert not bool(jnp.allclose(out[:64], out[64:]))

    def test_tree_output_nonzero_for_active_input(self, key):
        """Active input with non-zero values must produce non-zero output."""
        layer = EncoderLayer(n_inputs=1, key=key)
        xs    = jax.random.normal(key, (1, 1, 8, 8))
        flags = jnp.array([True])

        out, _ = layer(xs, flags)

        assert bool(jnp.any(out != 0.0))


# ─── DecoderLayer ────────────────────────────────────────────────────────────

class TestDecoderLayer:

    def _make_layer(self, k, n_prev, key):
        """Helper: build a DecoderLayer with a simple sequential parent_indices."""
        # Tile prev indices to fill k*16 slots, then reshape
        total = k * 16
        pool  = np.tile(np.arange(n_prev, dtype=np.int32),
                        int(np.ceil(total / n_prev)))[:total]
        pi = pool.reshape(k, 16)
        return DecoderLayer(parent_indices=pi, key=key)

    def test_output_shape_all_active(self, key):
        """All 16 prev outputs active -> all K decoders should activate (sum=16>=12)."""
        layer      = self._make_layer(4, 16, key)
        prev_out   = jnp.ones((16, 8, 8))
        prev_flags = jnp.ones((16,), dtype=bool)

        out, acts = layer(prev_out, prev_flags)

        assert out.shape == (4, 8, 8)
        assert bool(jnp.all(acts))

    def test_output_shape_all_inactive(self, key):
        """No active prev outputs -> all decoders inactive, outputs are zeros."""
        layer      = self._make_layer(4, 16, key)
        prev_out   = jnp.ones((16, 8, 8))
        prev_flags = jnp.zeros((16,), dtype=bool)

        out, acts = layer(prev_out, prev_flags)

        assert out.shape == (4, 8, 8)
        assert bool(jnp.all(~acts))
        assert bool(jnp.all(out == 0.0))

    def test_parent_indices_shape(self, key):
        """Adjacency array must be (K, 16)."""
        layer = self._make_layer(6, 20, key)
        assert layer.parent_indices.shape == (6, 16)

    def test_parent_indices_in_range(self, key):
        """All wired parent indices must be valid prev-layer indices."""
        n_prev = 10
        layer  = self._make_layer(4, n_prev, key)
        assert bool(jnp.all(layer.parent_indices >= 0))
        assert bool(jnp.all(layer.parent_indices < n_prev))

    def test_jit_compiles(self, key):
        """Layer must compile cleanly under eqx.filter_jit."""
        layer      = self._make_layer(4, 16, key)
        prev_out   = jnp.ones((16, 8, 8))
        prev_flags = jnp.ones((16,), dtype=bool)

        out, acts = eqx.filter_jit(layer)(prev_out, prev_flags)

        assert out.shape == (4, 8, 8)

    def test_larger_k_nodes(self, key):
        """Scale test: K=8, M=24 prev outputs, spatial 12x12."""
        layer      = self._make_layer(8, 24, key)
        prev_out   = jnp.ones((24, 12, 12))
        prev_flags = jnp.ones((24,), dtype=bool)

        out, acts = layer(prev_out, prev_flags)

        assert out.shape == (8, 12, 12)


# ─── FCLayer ─────────────────────────────────────────────────────────────────

class TestFCLayer:

    def test_output_shape(self, key):
        layer = FCLayer(in_features=256, out_features=128, key=key)
        x     = jnp.ones((256,))

        out = layer(x)

        assert out.shape == (128,)

    def test_relu_output_nonnegative(self, key):
        """After ReLU, all output values must be >= 0."""
        layer = FCLayer(in_features=8, out_features=8, key=key, activation='relu')
        x     = jnp.array([-3.0, -1.0, 0.0, 1.0, 2.0, -5.0, 3.0, 4.0])

        out = layer(x)

        assert bool(jnp.all(out >= 0.0))

    def test_gelu_activation(self, key):
        """GELU activation computes without error and returns correct shape."""
        layer = FCLayer(in_features=16, out_features=8, key=key, activation='gelu')
        x     = jnp.ones((16,))

        out = layer(x)

        assert out.shape == (8,)

    def test_sigmoid_activation(self, key):
        """Sigmoid output must lie in (0, 1)."""
        layer = FCLayer(in_features=16, out_features=8, key=key, activation='sigmoid')
        x     = jnp.ones((16,))

        out = layer(x)

        assert out.shape == (8,)
        assert bool(jnp.all(out > 0.0))
        assert bool(jnp.all(out < 1.0))

    def test_identity_activation(self, key):
        """Identity activation: output == linear(x)."""
        layer  = FCLayer(in_features=4, out_features=4, key=key, activation='identity')
        linear = layer.linear
        x      = jnp.array([1.0, 2.0, 3.0, 4.0])

        out = layer(x)

        assert bool(jnp.allclose(out, linear(x)))

    def test_unknown_activation_raises(self, key):
        with pytest.raises((KeyError, ValueError)):
            FCLayer(in_features=4, out_features=4, key=key, activation='swish')

    def test_jit_compiles(self, key):
        """Layer must compile cleanly under eqx.filter_jit."""
        layer = FCLayer(in_features=256, out_features=128, key=key)
        x     = jnp.ones((256,))

        out = eqx.filter_jit(layer)(x)

        assert out.shape == (128,)

    def test_stacked_fc_layers(self, key):
        """Two FC layers chained together: 512 -> 256 -> 128."""
        k1, k2 = jax.random.split(key)
        l1 = FCLayer(in_features=512, out_features=256, key=k1)
        l2 = FCLayer(in_features=256, out_features=128, key=k2)
        x  = jnp.ones((512,))

        out = l2(l1(x))

        assert out.shape == (128,)
