"""
Tests for the three TMRM layer types:
    - EncoderLayer  (N independent 2-encoder stacks)
    - DecoderLayer  (K Decoder nodes with static adjacency wiring)
    - FCLayer       (Linear + activation terminal block)
"""

import pytest
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


# ─── DecoderLayer ────────────────────────────────────────────────────────────

class TestDecoderLayer:

    def test_output_shape_all_active(self, key):
        """All 16 prev outputs active -> all K decoders should activate (sum=16>=12)."""
        layer      = DecoderLayer(k_nodes=4, n_prev_outputs=16, key=key)
        prev_out   = jnp.ones((16, 8, 8))
        prev_flags = jnp.ones((16,), dtype=bool)

        out, acts = layer(prev_out, prev_flags)

        assert out.shape == (4, 1, 8, 8)
        assert bool(jnp.all(acts))

    def test_output_shape_all_inactive(self, key):
        """No active prev outputs -> all decoders inactive, outputs are zeros."""
        layer      = DecoderLayer(k_nodes=4, n_prev_outputs=16, key=key)
        prev_out   = jnp.ones((16, 8, 8))
        prev_flags = jnp.zeros((16,), dtype=bool)

        out, acts = layer(prev_out, prev_flags)

        assert out.shape == (4, 1, 8, 8)
        assert bool(jnp.all(~acts))
        assert bool(jnp.all(out == 0.0))

    def test_parent_indices_shape(self, key):
        """Adjacency array must be (K, 16)."""
        layer = DecoderLayer(k_nodes=6, n_prev_outputs=20, key=key)
        assert layer.parent_indices.shape == (6, 16)

    def test_parent_indices_in_range(self, key):
        """All wired parent indices must be valid prev-layer indices."""
        n_prev = 10
        layer  = DecoderLayer(k_nodes=4, n_prev_outputs=n_prev, key=key)
        assert bool(jnp.all(layer.parent_indices >= 0))
        assert bool(jnp.all(layer.parent_indices < n_prev))

    def test_jit_compiles(self, key):
        """Layer must compile cleanly under eqx.filter_jit."""
        layer      = DecoderLayer(k_nodes=4, n_prev_outputs=16, key=key)
        prev_out   = jnp.ones((16, 8, 8))
        prev_flags = jnp.ones((16,), dtype=bool)

        out, acts = eqx.filter_jit(layer)(prev_out, prev_flags)

        assert out.shape == (4, 1, 8, 8)

    def test_larger_k_nodes(self, key):
        """Scale test: K=8, M=24 prev outputs, spatial 12x12."""
        layer      = DecoderLayer(k_nodes=8, n_prev_outputs=24, key=key)
        prev_out   = jnp.ones((24, 12, 12))
        prev_flags = jnp.ones((24,), dtype=bool)

        out, acts = layer(prev_out, prev_flags)

        assert out.shape == (8, 1, 12, 12)


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
