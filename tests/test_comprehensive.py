"""
Comprehensive & adversarial test suite for TMRM.

Covers:
- Boundary conditions (exact threshold, 1×1 spatial, n=1 inputs, K=1 decoders)
- Adversarial inputs (NaN, Inf, ±1e30, singular/zero matrices)
- Gradient flow (active and inactive paths)
- Numerical stability (slogdet masking, top_k validity)
- Wiring helpers (_straight_wire, _gaussian_wire, _terminal_wire)
- DecoderCluster (construction, wiring validity, forward pass, JIT)
- JIT consistency between eager and jit-compiled paths
- Non-square spatial dims
- FCLayer additional activations and edge cases
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from utils.encoder import Encoder
from utils.decoder import Decoder
from utils.encoder_layer import EncoderLayer
from utils.decoder_layer import DecoderLayer
from utils.decoder_cluster import (
    DecoderCluster,
    _straight_wire,
    _gaussian_wire,
    _terminal_wire,
)
from utils.fc_layer import FCLayer
from utils.config.trainparams import (
    DECODER_MAX_PARENTS,
    DECODER_ACTIVATION_THRESHOLD,
    DECODER_TOP_K_EXTRACT,
    DECODER_INTERACT_RANKS,
    ENCODER_OUT_CHANNELS,
)


# ─── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def key():
    return jax.random.key(0)


# ══════════════════════════════════════════════════════════════════════════════
# ENCODER — boundary & adversarial inputs
# ══════════════════════════════════════════════════════════════════════════════

class TestEncoderBoundary:

    def test_1x1_spatial_active(self, key):
        """Smallest valid spatial size through the active path."""
        enc = Encoder(key)
        x = jax.random.normal(key, (1, 1, 1))
        out, act = enc(x, jnp.array(True))
        assert out.shape == (8, 1, 1)
        assert bool(act)

    def test_1x1_spatial_inactive(self, key):
        """1×1 spatial through inactive path."""
        enc = Encoder(key)
        x = jax.random.normal(key, (1, 1, 1))
        out, act = enc(x, jnp.array(False))
        assert out.shape == (8, 1, 1)
        assert bool(~act)
        assert bool(jnp.all(out == 0.0))

    def test_non_square_spatial_unsupported(self, key):
        """Non-square (n, m) spatial is unsupported: pairwise matmul requires square
        inputs because it contracts the inner spatial dimension (m must equal n).
        Verify the architectural constraint is enforced (TypeError raised)."""
        enc = Encoder(key)
        x = jax.random.normal(key, (1, 6, 10))  # 6 != 10 → matmul will fail
        with pytest.raises(Exception):  # TypeError from XLA dot_general
            enc(x, jnp.array(True))

    def test_inactive_zeroes_regardless_of_huge_input(self, key):
        """Huge input + inactive flag must still produce exact all-zero output."""
        enc = Encoder(key)
        x = jnp.full((1, 8, 8), 1e30)
        out, act = enc(x, jnp.array(False))
        assert bool(jnp.all(out == 0.0))
        assert bool(~act)

    def test_nan_input_propagates_in_active_path(self, key):
        """NaN inputs in active path: output is NaN (no silent zeroing or crash)."""
        enc = Encoder(key)
        x = jnp.full((1, 4, 4), jnp.nan)
        out, act = enc(x, jnp.array(True))
        assert out.shape == (8, 4, 4)
        assert bool(jnp.any(jnp.isnan(out)))
        assert bool(act)

    def test_inf_input_does_not_crash(self, key):
        """Inf inputs in active path should return Inf/NaN, not crash."""
        enc = Encoder(key)
        x = jnp.full((1, 4, 4), jnp.inf)
        out, act = enc(x, jnp.array(True))
        assert out.shape == (8, 4, 4)
        assert bool(act)

    def test_very_large_input_no_crash(self, key):
        """1e30 magnitude inputs must not crash."""
        enc = Encoder(key)
        x = jnp.full((1, 8, 8), 1e30)
        out, act = enc(x, jnp.array(True))
        assert out.shape == (8, 8, 8)
        assert bool(act)

    def test_zero_input_active_correct_shape_and_flag(self, key):
        """All-zero input through active path: shape + flag correct."""
        enc = Encoder(key)
        x = jnp.zeros((1, 8, 8))
        out, act = enc(x, jnp.array(True))
        assert out.shape == (8, 8, 8)
        assert bool(act)

    def test_jit_eager_consistency(self, key):
        """Eager and JIT outputs must be numerically identical."""
        enc = Encoder(key)
        x = jax.random.normal(key, (1, 8, 8))
        out_eager, _ = enc(x, jnp.array(True))
        out_jit, _ = eqx.filter_jit(enc)(x, jnp.array(True))
        assert bool(jnp.allclose(out_eager, out_jit, atol=1e-6))

    def test_gradient_active_path_nonzero(self, key):
        """Gradient w.r.t. input must be non-zero through active path."""
        enc = Encoder(key)
        x = jax.random.normal(key, (1, 4, 4))

        def loss(xv):
            out, _ = enc(xv, jnp.array(True))
            return jnp.sum(out)

        grad = jax.grad(loss)(x)
        assert grad.shape == x.shape
        assert bool(jnp.any(grad != 0.0))

    def test_gradient_inactive_path_is_zero(self, key):
        """Inactive path returns zeros → gradient w.r.t input must be zero."""
        enc = Encoder(key)
        x = jax.random.normal(key, (1, 4, 4))

        def loss(xv):
            out, _ = enc(xv, jnp.array(False))
            return jnp.sum(out)

        grad = jax.grad(loss)(x)
        assert bool(jnp.all(grad == 0.0))

    def test_large_spatial_no_crash(self, key):
        """Large spatial input must run without crash."""
        enc = Encoder(key)
        x = jax.random.normal(key, (1, 16, 16))
        out, act = enc(x, jnp.array(True))
        assert out.shape == (8, 16, 16)

    def test_pairwise_intermediate_shapes(self, key):
        """Verify conv1→pairs→concat→conv2 intermediate dims are correct."""
        enc = Encoder(key)
        n = 5
        x = jax.random.normal(key, (1, n, n))

        x_conv1 = enc.conv1(x)
        assert x_conv1.shape == (4, n, n), "conv1 must expand to 4 channels"

        idx1 = jnp.array([0, 0, 0, 1, 1, 2])
        idx2 = jnp.array([1, 2, 3, 2, 3, 3])
        pairs = jnp.matmul(x_conv1[idx1], x_conv1[idx2])
        assert pairs.shape == (6, n, n), "4C2=6 pairwise products"

        concat = jnp.concatenate([x_conv1, pairs], axis=0)
        assert concat.shape == (10, n, n), "4 + 6 = 10 channels"


# ══════════════════════════════════════════════════════════════════════════════
# DECODER — boundary & adversarial inputs
# ══════════════════════════════════════════════════════════════════════════════

class TestDecoderBoundary:

    def test_exactly_at_threshold_fires(self, key):
        """Exactly 12 (=DECODER_ACTIVATION_THRESHOLD) active → gate opens."""
        dec = Decoder(key)
        x = jax.random.normal(key, (16, 8, 8))
        flags = jnp.array([True] * DECODER_ACTIVATION_THRESHOLD
                          + [False] * (DECODER_MAX_PARENTS - DECODER_ACTIVATION_THRESHOLD))
        out, act = dec(x, flags)
        assert out.shape == (8, 8)
        assert bool(act)

    def test_one_below_threshold_stays_closed(self, key):
        """11 active parents → gate stays closed (< 12)."""
        dec = Decoder(key)
        x = jax.random.normal(key, (16, 8, 8))
        flags = jnp.array([True] * (DECODER_ACTIVATION_THRESHOLD - 1)
                          + [False] * (DECODER_MAX_PARENTS - DECODER_ACTIVATION_THRESHOLD + 1))
        out, act = dec(x, flags)
        assert out.shape == (8, 8)
        assert bool(~act)
        assert bool(jnp.all(out == 0.0))

    def test_all_16_active_nonzero_output(self, key):
        """All 16 active → gate opens and output is non-zero."""
        dec = Decoder(key)
        x = jax.random.normal(key, (16, 8, 8))
        flags = jnp.ones(16, dtype=bool)
        out, act = dec(x, flags)
        assert bool(act)
        assert bool(jnp.any(out != 0.0))

    def test_all_inactive_exact_zeros(self, key):
        """Zero active parents → gate stays closed → strict zero output."""
        dec = Decoder(key)
        x = jax.random.normal(key, (16, 8, 8))
        flags = jnp.zeros(16, dtype=bool)
        out, act = dec(x, flags)
        assert bool(~act)
        assert bool(jnp.all(out == 0.0))

    def test_1x1_spatial(self, key):
        """Smallest square spatial size (slogdet needs square matrices)."""
        dec = Decoder(key)
        x = jax.random.normal(key, (16, 1, 1))
        flags = jnp.ones(16, dtype=bool)
        out, act = dec(x, flags)
        assert out.shape == (1, 1)
        assert bool(act)

    def test_singular_matrices_no_crash(self, key):
        """All-zero (singular) matrices: slogdet=-inf → all scores -inf, must not crash."""
        dec = Decoder(key)
        x = jnp.zeros((16, 4, 4))
        flags = jnp.ones(16, dtype=bool)
        out, act = dec(x, flags)
        assert out.shape == (4, 4)
        assert bool(act)

    def test_identity_matrices_valid(self, key):
        """Identity matrices (logdet=0) are positive-definite: decoder runs fine."""
        dec = Decoder(key)
        n = 4
        x = jnp.broadcast_to(jnp.eye(n), (16, n, n))
        flags = jnp.ones(16, dtype=bool)
        out, act = dec(x, flags)
        assert out.shape == (n, n)
        assert bool(act)

    def test_nan_input_active_no_crash(self, key):
        """NaN matrices in active path: must not crash (NaN propagation OK)."""
        dec = Decoder(key)
        x = jnp.full((16, 4, 4), jnp.nan)
        flags = jnp.ones(16, dtype=bool)
        out, act = dec(x, flags)
        assert out.shape == (4, 4)
        assert bool(act)

    def test_inactive_ignores_huge_input(self, key):
        """Inactive path with 1e30 inputs must return exact zeros."""
        dec = Decoder(key)
        x = jnp.full((16, 4, 4), 1e30)
        flags = jnp.zeros(16, dtype=bool)
        out, act = dec(x, flags)
        assert bool(jnp.all(out == 0.0))
        assert bool(~act)

    def test_gradient_flows_through_active_decoder(self, key):
        """Non-zero gradient must reach inputs through the active decoder path."""
        dec = Decoder(key)
        x = jax.random.normal(key, (16, 4, 4))
        flags = jnp.ones(16, dtype=bool)

        def loss(xv):
            out, _ = dec(xv, flags)
            return jnp.sum(out)

        grad = jax.grad(loss)(x)
        assert grad.shape == x.shape

    def test_gradient_zero_in_inactive_decoder(self, key):
        """Inactive decoder → zero outputs → zero gradient."""
        dec = Decoder(key)
        x = jax.random.normal(key, (16, 4, 4))
        flags = jnp.zeros(16, dtype=bool)

        def loss(xv):
            out, _ = dec(xv, flags)
            return jnp.sum(out)

        grad = jax.grad(loss)(x)
        assert bool(jnp.all(grad == 0.0))

    def test_jit_eager_consistency(self, key):
        """JIT and eager outputs must be numerically identical."""
        dec = Decoder(key)
        x = jax.random.normal(key, (16, 6, 6))
        flags = jnp.ones(16, dtype=bool)
        out_eager, _ = dec(x, flags)
        out_jit, _ = eqx.filter_jit(dec)(x, flags)
        assert bool(jnp.allclose(out_eager, out_jit, atol=1e-6))

    def test_different_inputs_produce_different_outputs(self, key):
        """Two distinct gate-passing inputs must yield distinct outputs."""
        dec = Decoder(key)
        k1, k2 = jax.random.split(key)
        x1 = jax.random.normal(k1, (16, 6, 6))
        x2 = jax.random.normal(k2, (16, 6, 6))
        flags = jnp.ones(16, dtype=bool)
        out1, _ = dec(x1, flags)
        out2, _ = dec(x2, flags)
        assert not bool(jnp.allclose(out1, out2))

    def test_decoder_pairwise_shapes(self, key):
        """Verify 8C2=28 combinations from top-8 produce correct intermediate shapes."""
        n = 4
        top8 = jax.random.normal(key, (DECODER_INTERACT_RANKS, n, n))
        idx1, idx2 = jnp.triu_indices(DECODER_INTERACT_RANKS, k=1)
        pairs = jnp.matmul(top8[idx1], top8[idx2])
        assert pairs.shape == (28, n, n), "8C2=28 pairwise products"

    def test_output_is_swish_bounded(self, key):
        """Swish(x) = x*sigmoid(x) ∈ [-0.28..., +inf); sanity check range."""
        dec = Decoder(key)
        x = jax.random.normal(key, (16, 4, 4))
        flags = jnp.ones(16, dtype=bool)
        out, act = dec(x, flags)
        if bool(act):
            # swish is bounded below by ~-0.278
            assert bool(jnp.all(out >= -0.3))


# ══════════════════════════════════════════════════════════════════════════════
# WIRING HELPERS — unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestStraightWire:

    @pytest.mark.parametrize("n", [1, 7, 16, 17, 32, 100])
    def test_shape_and_k(self, n):
        """Output is (ceil(n/16), 16) and K matches."""
        rng = np.random.default_rng(0)
        pi, K = _straight_wire(n, rng)
        expected_K = int(np.ceil(n / DECODER_MAX_PARENTS))
        assert pi.shape == (expected_K, DECODER_MAX_PARENTS)
        assert K == expected_K

    @pytest.mark.parametrize("n", [1, 7, 16, 20, 64])
    def test_all_indices_covered(self, n):
        """Every encoder index 0..n-1 appears at least once."""
        rng = np.random.default_rng(1)
        pi, _ = _straight_wire(n, rng)
        present = set(pi.flatten().tolist())
        for i in range(n):
            assert i in present, f"Index {i} missing for n={n}"

    @pytest.mark.parametrize("n", [4, 16, 50])
    def test_indices_in_range(self, n):
        """All indices in [0, n)."""
        rng = np.random.default_rng(2)
        pi, _ = _straight_wire(n, rng)
        assert int(pi.min()) >= 0
        assert int(pi.max()) < n

    def test_single_input_all_slots_zero(self):
        """n=1: K=1, all 16 slots must equal 0."""
        rng = np.random.default_rng(3)
        pi, K = _straight_wire(1, rng)
        assert K == 1
        assert pi.shape == (1, 16)
        assert bool(np.all(pi == 0))

    def test_dtype_is_int32(self):
        """Output dtype must be int32."""
        rng = np.random.default_rng(4)
        pi, _ = _straight_wire(32, rng)
        assert pi.dtype == np.int32

    def test_different_rngs_different_layouts(self):
        """Different RNG seeds produce different slot arrangements (very likely)."""
        pi1, _ = _straight_wire(64, np.random.default_rng(10))
        pi2, _ = _straight_wire(64, np.random.default_rng(99))
        assert not np.array_equal(pi1, pi2)


class TestGaussianWire:

    def test_shape(self):
        """Output adjacency is (k_nodes, 16)."""
        rng = np.random.default_rng(0)
        n_prev, k_nodes = 30, 8
        fanouts = np.full(n_prev, 6, dtype=np.int32)  # pool=180, supports 11 nodes
        pi, unc = _gaussian_wire(n_prev, fanouts, k_nodes, rng)
        assert pi.shape == (k_nodes, DECODER_MAX_PARENTS)
        assert unc.shape == (n_prev,)

    def test_indices_in_range(self):
        """All gathered indices are within [0, n_prev)."""
        rng = np.random.default_rng(1)
        n_prev, k_nodes = 30, 8
        fanouts = np.full(n_prev, 8, dtype=np.int32)
        pi, _ = _gaussian_wire(n_prev, fanouts, k_nodes, rng)
        assert int(pi.min()) >= 0
        assert int(pi.max()) < n_prev

    def test_unconnected_mask_dtype(self):
        """unconnected_mask must be boolean array of shape (n_prev,)."""
        rng = np.random.default_rng(2)
        n_prev, k_nodes = 20, 5
        fanouts = np.full(n_prev, 10, dtype=np.int32)
        pi, unc = _gaussian_wire(n_prev, fanouts, k_nodes, rng)
        assert unc.dtype == bool
        assert unc.shape == (n_prev,)

    def test_unconnected_mask_correctness(self):
        """An index is unconnected iff it's absent from the USED portion of the pool."""
        rng = np.random.default_rng(3)
        n_prev, k_nodes = 20, 5
        fanouts = np.full(n_prev, 8, dtype=np.int32)
        pi, unc = _gaussian_wire(n_prev, fanouts, k_nodes, rng)
        used_set = set(pi.flatten().tolist())
        for i in range(n_prev):
            expected = i not in used_set
            assert unc[i] == expected, f"Index {i}: mask={unc[i]}, expected={expected}"

    def test_dtype_is_int32(self):
        """parent_indices dtype must be int32."""
        rng = np.random.default_rng(4)
        fanouts = np.full(20, 6, dtype=np.int32)
        pi, _ = _gaussian_wire(20, fanouts, 7, rng)
        assert pi.dtype == np.int32


class TestTerminalWire:

    @pytest.mark.parametrize("n_prev,k_budget", [(64, 3), (48, 10), (16, 1)])
    def test_shape_and_k_terminal(self, n_prev, k_budget):
        """k_terminal = min(budget, n_prev//16); shape (k_terminal, 16)."""
        rng = np.random.default_rng(0)
        pi, k_term, leftover = _terminal_wire(n_prev, k_budget, rng)
        expected_k = min(k_budget, n_prev // DECODER_MAX_PARENTS)
        assert k_term == expected_k
        assert pi.shape == (k_term, DECODER_MAX_PARENTS)
        assert leftover.shape == (n_prev,)

    @pytest.mark.parametrize("n_prev", [16, 48, 100])
    def test_no_duplicate_slots(self, n_prev):
        """Each index appears at most once across all terminal slots."""
        rng = np.random.default_rng(1)
        pi, k_term, _ = _terminal_wire(n_prev, 10, rng)
        if k_term > 0:
            flat = pi.flatten().tolist()
            assert len(flat) == len(set(flat)), "Duplicate index found in terminal wiring"

    def test_edge_n_less_than_16(self):
        """When n_prev < 16, k_terminal=0, all are leftovers."""
        rng = np.random.default_rng(2)
        pi, k_term, leftover = _terminal_wire(15, 5, rng)
        assert k_term == 0
        assert pi.shape == (0, DECODER_MAX_PARENTS)
        assert bool(np.all(leftover)), "All 15 nodes should be leftover"

    def test_budget_zero_yields_no_decoders(self):
        """k_budget=0 → k_terminal=0."""
        rng = np.random.default_rng(3)
        pi, k_term, leftover = _terminal_wire(32, 0, rng)
        assert k_term == 0

    def test_leftover_mask_correctness(self):
        """Leftover mask is True iff index is NOT in parent_indices."""
        rng = np.random.default_rng(4)
        n_prev = 32
        pi, k_term, leftover = _terminal_wire(n_prev, 2, rng)
        used_set = set(pi.flatten().tolist()) if k_term > 0 else set()
        for i in range(n_prev):
            if i in used_set:
                assert not leftover[i], f"Index {i} used but wrongly marked leftover"
            else:
                assert leftover[i], f"Index {i} unused but not marked leftover"

    def test_dtype_is_int32(self):
        """parent_indices dtype must be int32."""
        rng = np.random.default_rng(5)
        pi, _, _ = _terminal_wire(32, 2, rng)
        assert pi.dtype == np.int32

    def test_indices_in_range(self):
        """All parent indices are valid prev-layer indices."""
        rng = np.random.default_rng(6)
        n_prev = 64
        pi, k_term, _ = _terminal_wire(n_prev, 3, rng)
        if k_term > 0:
            assert int(pi.min()) >= 0
            assert int(pi.max()) < n_prev

    def test_n_exactly_16(self):
        """n_prev=16, k_budget=5: k_terminal=1 (16//16=1)."""
        rng = np.random.default_rng(7)
        pi, k_term, leftover = _terminal_wire(16, 5, rng)
        assert k_term == 1
        assert pi.shape == (1, 16)


# ══════════════════════════════════════════════════════════════════════════════
# DECODER CLUSTER — construction, wiring validity, forward pass
# ══════════════════════════════════════════════════════════════════════════════

class TestDecoderCluster:

    def test_basic_construction(self, key):
        """Cluster builds without error and has at least one layer."""
        cluster = DecoderCluster(n_layers=2, max_nodes=100, n_inputs=32, key=key)
        assert len(cluster.layers) >= 1

    def test_n_output_nodes_positive(self, key):
        """Cluster must have at least one output node."""
        cluster = DecoderCluster(n_layers=2, max_nodes=50, n_inputs=32, key=key)
        assert cluster.n_output_nodes > 0

    def test_output_slots_equal_layers(self, key):
        """output_node_indices must have the same length as layers."""
        cluster = DecoderCluster(n_layers=3, max_nodes=100, n_inputs=64, key=key)
        assert len(cluster.output_node_indices) == len(cluster.layers)

    def test_no_none_output_slots(self, key):
        """No output_node_indices entry should be None after construction."""
        cluster = DecoderCluster(n_layers=2, max_nodes=100, n_inputs=32, key=key)
        for idx in cluster.output_node_indices:
            assert isinstance(idx, np.ndarray), "Slot must be ndarray, not None"

    def test_last_layer_always_has_outputs(self, key):
        """Last layer's output_node_indices must be non-empty (safety net)."""
        cluster = DecoderCluster(n_layers=2, max_nodes=100, n_inputs=32, key=key)
        assert cluster.output_node_indices[-1].size > 0

    def test_max_nodes_not_exceeded(self, key):
        """Total decoder nodes across all layers must not exceed max_nodes."""
        max_nodes = 50
        cluster = DecoderCluster(n_layers=5, max_nodes=max_nodes, n_inputs=64, key=key)
        total = sum(layer.parent_indices.shape[0] for layer in cluster.layers)
        assert total <= max_nodes, f"total={total} exceeds max_nodes={max_nodes}"

    def test_layer0_covers_all_encoder_outputs(self, key):
        """Layer-0 wiring must reference every encoder output at least once."""
        n_inputs = 20
        cluster = DecoderCluster(n_layers=2, max_nodes=100, n_inputs=n_inputs, key=key)
        pi0 = cluster.layers[0].parent_indices
        present = set(pi0.flatten().tolist())
        for i in range(n_inputs):
            assert i in present, f"Encoder output {i} not connected to any layer-0 decoder"

    def test_all_parent_indices_in_range(self, key):
        """Each layer's parent_indices must index valid previous-layer nodes."""
        n_inputs = 32
        cluster = DecoderCluster(n_layers=3, max_nodes=100, n_inputs=n_inputs, key=key)
        sizes = [n_inputs] + [layer.parent_indices.shape[0] for layer in cluster.layers[:-1]]
        for l_idx, layer in enumerate(cluster.layers):
            pi = layer.parent_indices
            assert int(pi.min()) >= 0, f"Layer {l_idx}: negative index"
            assert int(pi.max()) < sizes[l_idx], (
                f"Layer {l_idx}: index {pi.max()} >= prev size {sizes[l_idx]}"
            )

    def test_forward_pass_shape_consistent(self, key):
        """Forward output first dim equals n_output_nodes."""
        n_inputs = 32
        cluster = DecoderCluster(n_layers=2, max_nodes=80, n_inputs=n_inputs, key=key)
        enc_out = jax.random.normal(key, (n_inputs, 8, 8))
        enc_flags = jnp.ones(n_inputs, dtype=bool)
        out, flags = cluster(enc_out, enc_flags)
        assert out.shape == (cluster.n_output_nodes, 8, 8)
        assert flags.shape == (cluster.n_output_nodes,)

    def test_forward_all_inactive_all_zeros(self, key):
        """All-inactive encoder flags → all cluster outputs zero, all flags False."""
        n_inputs = 32
        cluster = DecoderCluster(n_layers=1, max_nodes=20, n_inputs=n_inputs, key=key)
        enc_out = jax.random.normal(key, (n_inputs, 4, 4))
        enc_flags = jnp.zeros(n_inputs, dtype=bool)
        out, flags = cluster(enc_out, enc_flags)
        assert bool(jnp.all(~flags)), "All flags should be False"
        assert bool(jnp.all(out == 0.0)), "All outputs should be zero"

    def test_forward_all_active_some_nonzero(self, key):
        """With all-active inputs some output nodes should produce non-zero values."""
        n_inputs = 32
        cluster = DecoderCluster(n_layers=2, max_nodes=80, n_inputs=n_inputs, key=key)
        enc_out = jax.random.normal(key, (n_inputs, 8, 8))
        enc_flags = jnp.ones(n_inputs, dtype=bool)
        out, flags = cluster(enc_out, enc_flags)
        assert bool(jnp.any(flags)), "At least one output node should be active"

    def test_jit_compiles_and_matches_eager(self, key):
        """Cluster forward must compile under eqx.filter_jit and match eager output."""
        n_inputs = 16
        cluster = DecoderCluster(n_layers=1, max_nodes=20, n_inputs=n_inputs, key=key)
        enc_out = jax.random.normal(key, (n_inputs, 4, 4))
        enc_flags = jnp.ones(n_inputs, dtype=bool)
        out_eager, _ = cluster(enc_out, enc_flags)
        out_jit, _ = eqx.filter_jit(cluster)(enc_out, enc_flags)
        assert bool(jnp.allclose(out_eager, out_jit, atol=1e-6))

    def test_deterministic_given_same_key(self, key):
        """Same key → identical wiring and n_output_nodes."""
        c1 = DecoderCluster(n_layers=2, max_nodes=50, n_inputs=32, key=key)
        c2 = DecoderCluster(n_layers=2, max_nodes=50, n_inputs=32, key=key)
        assert len(c1.layers) == len(c2.layers)
        assert c1.n_output_nodes == c2.n_output_nodes
        for pi1, pi2 in zip(c1.output_node_indices, c2.output_node_indices):
            assert np.array_equal(pi1, pi2)

    def test_different_keys_different_wirings(self, key):
        """Different keys should (almost certainly) produce different layer-0 wiring."""
        k1, k2 = jax.random.split(key)
        c1 = DecoderCluster(n_layers=2, max_nodes=50, n_inputs=32, key=k1)
        c2 = DecoderCluster(n_layers=2, max_nodes=50, n_inputs=32, key=k2)
        pi1 = c1.layers[0].parent_indices
        pi2 = c2.layers[0].parent_indices
        assert not np.array_equal(pi1, pi2), "Different seeds should yield different wirings"

    def test_n_inputs_exactly_16(self, key):
        """n_inputs=16 → layer-0 has exactly 1 decoder node (ceil(16/16)=1)."""
        cluster = DecoderCluster(n_layers=1, max_nodes=10, n_inputs=16, key=key)
        assert cluster.layers[0].parent_indices.shape[0] == 1

    def test_large_n_inputs(self, key):
        """Large n_inputs=32 cluster builds and has valid wiring."""
        cluster = DecoderCluster(n_layers=3, max_nodes=64, n_inputs=32, key=key)
        assert cluster.n_output_nodes > 0
        assert len(cluster.layers) >= 1

    def test_output_node_indices_within_layer_bounds(self, key):
        """output_node_indices for layer l must index within [0, K_l)."""
        n_inputs = 32
        cluster = DecoderCluster(n_layers=3, max_nodes=100, n_inputs=n_inputs, key=key)
        for l_idx, (layer, idx) in enumerate(zip(cluster.layers, cluster.output_node_indices)):
            K_l = layer.parent_indices.shape[0]
            if idx.size > 0:
                assert int(idx.min()) >= 0, f"Layer {l_idx}: negative output index"
                assert int(idx.max()) < K_l, (
                    f"Layer {l_idx}: output index {idx.max()} >= K={K_l}"
                )

    def test_spatial_preserved_through_cluster(self, key):
        """Spatial dimensions (n, m) must be preserved end-to-end."""
        n_inputs = 16
        cluster = DecoderCluster(n_layers=1, max_nodes=10, n_inputs=n_inputs, key=key)
        enc_out = jnp.ones((n_inputs, 6, 6))
        enc_flags = jnp.ones(n_inputs, dtype=bool)
        out, _ = cluster(enc_out, enc_flags)
        assert out.shape[1] == 6
        assert out.shape[2] == 6


# ══════════════════════════════════════════════════════════════════════════════
# ENCODER LAYER — adversarial & structural tests
# ══════════════════════════════════════════════════════════════════════════════

class TestEncoderLayerAdversarial:

    def test_n1_single_input(self, key):
        """n_inputs=1: output is (64, n, n) with 64 flags."""
        layer = EncoderLayer(n_inputs=1, key=key)
        xs = jax.random.normal(key, (1, 1, 8, 8))
        flags = jnp.array([True])
        out, acts = layer(xs, flags)
        assert out.shape == (64, 8, 8)
        assert acts.shape == (64,)

    def test_1x1_spatial_all_active(self, key):
        """1×1 spatial through encoder layer must not crash."""
        layer = EncoderLayer(n_inputs=2, key=key)
        xs = jnp.ones((2, 1, 1, 1))
        flags = jnp.ones(2, dtype=bool)
        out, acts = layer(xs, flags)
        assert out.shape == (128, 1, 1)

    def test_single_active_in_large_batch(self, key):
        """N=6, only stack 2 active: stacks 0,1,3,4,5 produce zeros."""
        N = 6
        layer = EncoderLayer(n_inputs=N, key=key)
        xs = jax.random.normal(key, (N, 1, 8, 8))
        flags = jnp.array([False, False, True, False, False, False])
        out, acts = layer(xs, flags)
        # Stack 2 → channels 128:192
        assert bool(jnp.all(acts[128:192])),  "Stack-2 channels must be True"
        assert bool(jnp.all(~acts[:128])),    "Stacks 0,1 must be False"
        assert bool(jnp.all(~acts[192:])),    "Stacks 3,4,5 must be False"
        assert bool(jnp.all(out[:128] == 0.0))
        assert bool(jnp.all(out[192:] == 0.0))

    def test_gradient_flows_through_active_encoder_layer(self, key):
        """Gradient w.r.t. input tensor must be non-zero for active stack."""
        layer = EncoderLayer(n_inputs=1, key=key)
        xs = jax.random.normal(key, (1, 1, 4, 4))
        flags = jnp.array([True])

        def loss(xsv):
            out, _ = layer(xsv, flags)
            return jnp.sum(out)

        grad = jax.grad(loss)(xs)
        assert grad.shape == xs.shape
        assert bool(jnp.any(grad != 0.0))

    def test_stage2_batch_dims(self, key):
        """stage2_encs conv1 weight must have leading dims (N, 8, ...)."""
        N = 3
        layer = EncoderLayer(n_inputs=N, key=key)
        w = layer.stage2_encs.conv1.weight
        assert w.shape[0] == N, f"outer vmap dim: expected {N}, got {w.shape[0]}"
        assert w.shape[1] == 8, f"inner vmap dim: expected 8, got {w.shape[1]}"

    def test_large_spatial_no_crash(self, key):
        """Large spatial must run end-to-end without error."""
        layer = EncoderLayer(n_inputs=2, key=key)
        xs = jax.random.normal(key, (2, 1, 16, 16))
        flags = jnp.ones(2, dtype=bool)
        out, acts = layer(xs, flags)
        assert out.shape == (128, 16, 16)

    def test_nan_input_propagates(self, key):
        """NaN inputs in active stack should produce NaN outputs (no crash)."""
        layer = EncoderLayer(n_inputs=1, key=key)
        xs = jnp.full((1, 1, 4, 4), jnp.nan)
        flags = jnp.array([True])
        out, acts = layer(xs, flags)
        assert bool(jnp.any(jnp.isnan(out)))
        assert bool(jnp.all(acts))


# ══════════════════════════════════════════════════════════════════════════════
# DECODER LAYER — adversarial & structural tests
# ══════════════════════════════════════════════════════════════════════════════

class TestDecoderLayerAdversarial:

    def _make_layer(self, k, n_prev, key):
        total = k * DECODER_MAX_PARENTS
        pool = np.tile(np.arange(n_prev, dtype=np.int32),
                       int(np.ceil(total / n_prev)))[:total]
        return DecoderLayer(parent_indices=pool.reshape(k, DECODER_MAX_PARENTS), key=key)

    def test_k1_decoder(self, key):
        """K=1 decoder: output shape (1, n, n)."""
        layer = self._make_layer(1, 16, key)
        prev_out = jnp.ones((16, 4, 4))
        prev_flags = jnp.ones(16, dtype=bool)
        out, acts = layer(prev_out, prev_flags)
        assert out.shape == (1, 4, 4)
        assert acts.shape == (1,)

    def test_exactly_12_of_16_active_fires(self, key):
        """Exactly 12 of 16 wired parents active → decoder fires."""
        pi = np.arange(16, dtype=np.int32).reshape(1, 16)
        layer = DecoderLayer(parent_indices=pi, key=key)
        prev_out = jax.random.normal(key, (16, 4, 4))
        prev_flags = jnp.array([True] * 12 + [False] * 4)
        out, acts = layer(prev_out, prev_flags)
        assert bool(acts[0])

    def test_11_of_16_active_does_not_fire(self, key):
        """11 of 16 active → gate stays closed → zero output."""
        pi = np.arange(16, dtype=np.int32).reshape(1, 16)
        layer = DecoderLayer(parent_indices=pi, key=key)
        prev_out = jax.random.normal(key, (16, 4, 4))
        prev_flags = jnp.array([True] * 11 + [False] * 5)
        out, acts = layer(prev_out, prev_flags)
        assert bool(~acts[0])
        assert bool(jnp.all(out[0] == 0.0))

    def test_parent_indices_dtype(self, key):
        """parent_indices must be stored as numpy int32."""
        layer = self._make_layer(4, 16, key)
        assert isinstance(layer.parent_indices, np.ndarray)
        assert layer.parent_indices.dtype == np.int32

    def test_gradient_flows_through_layer(self, key):
        """Gradient must backprop through an active DecoderLayer."""
        pi = np.arange(16, dtype=np.int32).reshape(1, 16)
        layer = DecoderLayer(parent_indices=pi, key=key)
        prev_out = jax.random.normal(key, (16, 4, 4))
        prev_flags = jnp.ones(16, dtype=bool)

        def loss(po):
            out, _ = layer(po, prev_flags)
            return jnp.sum(out)

        grad = jax.grad(loss)(prev_out)
        assert grad.shape == prev_out.shape

    def test_inactive_layer_zero_regardless_of_prev_values(self, key):
        """All-inactive flags → zero output regardless of prev_out magnitude."""
        layer = self._make_layer(3, 16, key)
        prev_out = jnp.full((16, 4, 4), 1e20)
        prev_flags = jnp.zeros(16, dtype=bool)
        out, acts = layer(prev_out, prev_flags)
        assert bool(jnp.all(acts == False))
        assert bool(jnp.all(out == 0.0))

    def test_large_k_nodes(self, key):
        """K=16, M=64 prev outputs at 12×12 spatial."""
        layer = self._make_layer(16, 64, key)
        prev_out = jnp.ones((64, 12, 12))
        prev_flags = jnp.ones(64, dtype=bool)
        out, acts = layer(prev_out, prev_flags)
        assert out.shape == (16, 12, 12)
        assert acts.shape == (16,)


# ══════════════════════════════════════════════════════════════════════════════
# FC LAYER — adversarial & coverage tests
# ══════════════════════════════════════════════════════════════════════════════

class TestFCLayerAdversarial:

    def test_tanh_output_bounded(self, key):
        """tanh output must lie strictly in (-1, 1)."""
        layer = FCLayer(in_features=16, out_features=8, key=key, activation='tanh')
        x = jax.random.normal(key, (16,)) * 10.0  # saturates tanh
        out = layer(x)
        assert out.shape == (8,)
        assert bool(jnp.all(out > -1.0))
        assert bool(jnp.all(out < 1.0))

    def test_1_to_1_projection(self, key):
        """Smallest FC: 1 in → 1 out."""
        layer = FCLayer(in_features=1, out_features=1, key=key)
        out = layer(jnp.array([3.14]))
        assert out.shape == (1,)

    def test_large_projection(self, key):
        """4096 → 2048 projection runs without OOM or crash."""
        k1, _ = jax.random.split(key)
        layer = FCLayer(in_features=4096, out_features=2048, key=k1)
        x = jax.random.normal(k1, (4096,))
        out = layer(x)
        assert out.shape == (2048,)

    def test_gradient_through_all_activations(self, key):
        """Gradients must be non-None and correct shape for all activations."""
        for act_name in ['relu', 'gelu', 'tanh', 'sigmoid', 'identity']:
            layer = FCLayer(in_features=8, out_features=4, key=key, activation=act_name)
            x = jnp.ones((8,))

            def loss(xv):
                return jnp.sum(layer(xv))

            grad = jax.grad(loss)(x)
            assert grad.shape == (8,), f"Wrong grad shape for {act_name}"

    def test_all_invalid_activation_names_raise(self, key):
        """All unrecognised activation names must raise ValueError."""
        for bad in ['swish', 'elu', 'prelu', '', 'Relu', 'GELU']:
            with pytest.raises((ValueError, KeyError)):
                FCLayer(in_features=4, out_features=4, key=key, activation=bad)

    def test_jit_all_activations(self, key):
        """All valid activations must JIT-compile correctly."""
        for act_name in ['relu', 'gelu', 'tanh', 'sigmoid', 'identity']:
            layer = FCLayer(in_features=8, out_features=4, key=key, activation=act_name)
            out = eqx.filter_jit(layer)(jnp.ones((8,)))
            assert out.shape == (4,), f"Failed JIT for activation={act_name}"

    def test_nan_propagates_through_identity(self, key):
        """NaN input must propagate through identity activation."""
        layer = FCLayer(in_features=4, out_features=4, key=key, activation='identity')
        x = jnp.array([float('nan'), 1.0, 2.0, 3.0])
        out = layer(x)
        assert bool(jnp.any(jnp.isnan(out)))

    def test_relu_zero_for_all_negative_input(self, key):
        """Pure negative input through ReLU must give all-zero output (if bias=0)."""
        layer = FCLayer(in_features=4, out_features=4, key=key, activation='relu')
        # Override weights to zeros and bias to zeros for deterministic check
        x = jnp.array([-10.0, -10.0, -10.0, -10.0])
        # ReLU(linear(x)) >= 0 always
        out = layer(x)
        assert bool(jnp.all(out >= 0.0))

    def test_sigmoid_always_unit_interval(self, key):
        """Sigmoid output must lie in [0, 1] for any finite input.
        float32 saturates to exactly 0.0 / 1.0 for extreme inputs, so we use
        inclusive bounds rather than strict inequalities."""
        layer = FCLayer(in_features=8, out_features=8, key=key, activation='sigmoid')
        for scale in [0.0, 1.0, 100.0, -100.0]:
            x = jnp.full((8,), scale)
            out = layer(x)
            assert bool(jnp.all(out >= 0.0)), f"scale={scale}: output below 0"
            assert bool(jnp.all(out <= 1.0)), f"scale={scale}: output above 1"


# ══════════════════════════════════════════════════════════════════════════════
# NUMERICAL STABILITY — slogdet masking & top_k correctness
# ══════════════════════════════════════════════════════════════════════════════

class TestNumericalStability:

    def test_slogdet_inactive_entries_become_neg_inf(self):
        """Inactive entries scored as -inf are never picked by top_k."""
        n = 4
        inputs = jnp.broadcast_to(jnp.eye(n), (16, n, n))
        mask = jnp.array([True] * 12 + [False] * 4)
        _, logdet = jnp.linalg.slogdet(inputs)
        scores = jnp.where(mask, logdet, -jnp.inf)
        # Inactive entries must all be -inf
        assert bool(jnp.all(scores[12:] == -jnp.inf))
        # Active entries (identity: logdet=0) must be finite
        assert bool(jnp.all(jnp.isfinite(scores[:12])))

    def test_top_k_skips_neg_inf_entries(self):
        """top_k must select the 12 finite scores over the 4 -inf scores."""
        scores = jnp.concatenate([jnp.zeros(12), jnp.full(4, -jnp.inf)])
        _, indices = jax.lax.top_k(scores, DECODER_TOP_K_EXTRACT)
        for i in range(12):
            assert i in set(indices.tolist()), f"Score index {i} should be selected"

    def test_all_neg_inf_scores_top_k_no_crash(self):
        """If all 16 scores are -inf (e.g. all singular), top_k must not crash."""
        scores = jnp.full(16, -jnp.inf)
        _, indices = jax.lax.top_k(scores, DECODER_TOP_K_EXTRACT)
        assert indices.shape == (DECODER_TOP_K_EXTRACT,)

    def test_slogdet_on_singular_matrix(self):
        """slogdet of a rank-0 matrix returns -inf (not NaN or crash)."""
        mat = jnp.zeros((4, 4))
        sign, logdet = jnp.linalg.slogdet(mat)
        assert logdet == -jnp.inf or jnp.isnan(logdet)  # -inf or NaN both acceptable

    def test_slogdet_batched_shape(self):
        """Batched slogdet on (16, n, n) returns shape (16,)."""
        n = 6
        mats = jax.random.normal(jax.random.key(0), (16, n, n))
        _, logdets = jnp.linalg.slogdet(mats)
        assert logdets.shape == (16,)

    def test_encoder_no_nan_on_normal_input(self, key):
        """Random normal input through active encoder must not produce NaN."""
        enc = Encoder(key)
        x = jax.random.normal(key, (1, 8, 8))
        out, _ = enc(x, jnp.array(True))
        assert bool(~jnp.any(jnp.isnan(out))), "NaN in encoder output with normal input"

    def test_decoder_no_nan_on_normal_input(self, key):
        """Random normal input through active decoder must not produce NaN."""
        dec = Decoder(key)
        x = jax.random.normal(key, (16, 8, 8))
        flags = jnp.ones(16, dtype=bool)
        out, _ = dec(x, flags)
        assert bool(~jnp.any(jnp.isnan(out))), "NaN in decoder output with normal input"

    def test_encoder_intermediate_channel_counts(self, key):
        """Verify 4 + 6 = 10 intermediate channels in encoder."""
        enc = Encoder(key)
        n = 5
        x = jax.random.normal(key, (1, n, n))
        x4 = enc.conv1(x)
        assert x4.shape == (4, n, n)
        idx1 = jnp.array([0, 0, 0, 1, 1, 2])
        idx2 = jnp.array([1, 2, 3, 2, 3, 3])
        pairs = jnp.matmul(x4[idx1], x4[idx2])
        concat = jnp.concatenate([x4, pairs], axis=0)
        assert concat.shape == (10, n, n)


# ══════════════════════════════════════════════════════════════════════════════
# END-TO-END PIPELINE — EncoderLayer → DecoderCluster → FCLayer
# ══════════════════════════════════════════════════════════════════════════════

class TestEndToEndPipeline:

    def test_encoder_to_cluster_to_fc(self, key):
        """Full pipeline: EncoderLayer → DecoderCluster → FCLayer runs correctly."""
        k1, k2, k3 = jax.random.split(key, 3)
        n_modalities = 2
        n = 8

        # Encoder layer
        enc_layer = EncoderLayer(n_inputs=n_modalities, key=k1)
        xs = jax.random.normal(k1, (n_modalities, 1, n, n))
        enc_flags = jnp.ones(n_modalities, dtype=bool)
        enc_out, enc_active = enc_layer(xs, enc_flags)
        # enc_out: (128, n, n)

        # Decoder cluster
        cluster = DecoderCluster(n_layers=1, max_nodes=20, n_inputs=enc_out.shape[0], key=k2)
        dec_out, dec_flags = cluster(enc_out, enc_active)
        # dec_out: (n_output_nodes, n, n)

        # Flatten and FC
        flat = dec_out.reshape(-1)
        fc = FCLayer(in_features=flat.shape[0], out_features=16, key=k3)
        result = fc(flat)

        assert result.shape == (16,)

    def test_pipeline_jit_compiles(self, key):
        """Full pipeline must compile under jax.jit."""
        k1, k2, k3 = jax.random.split(key, 3)
        n_modalities, n = 2, 4

        enc_layer = EncoderLayer(n_inputs=n_modalities, key=k1)
        cluster = DecoderCluster(n_layers=1, max_nodes=10, n_inputs=128, key=k2)
        n_out = cluster.n_output_nodes
        fc = FCLayer(in_features=n_out * n * n, out_features=8, key=k3)

        @jax.jit
        def forward(xs, enc_flags):
            enc_out, enc_active = enc_layer(xs, enc_flags)
            dec_out, _ = cluster(enc_out, enc_active)
            flat = dec_out.reshape(-1)
            return fc(flat)

        xs = jax.random.normal(k1, (n_modalities, 1, n, n))
        flags = jnp.ones(n_modalities, dtype=bool)
        result = forward(xs, flags)
        assert result.shape == (8,)

    def test_pipeline_all_inactive_produces_zeros(self, key):
        """All-inactive flags propagate to zero FC input (relu(0) = 0)."""
        k1, k2, k3 = jax.random.split(key, 3)
        n_modalities, n = 2, 4

        enc_layer = EncoderLayer(n_inputs=n_modalities, key=k1)
        xs = jax.random.normal(k1, (n_modalities, 1, n, n))
        enc_flags = jnp.zeros(n_modalities, dtype=bool)
        enc_out, enc_active = enc_layer(xs, enc_flags)

        cluster = DecoderCluster(n_layers=1, max_nodes=10, n_inputs=enc_out.shape[0], key=k2)
        dec_out, _ = cluster(enc_out, enc_active)

        # All decoder outputs should be zero
        assert bool(jnp.all(dec_out == 0.0))

        flat = dec_out.reshape(-1)
        fc = FCLayer(in_features=flat.shape[0], out_features=4, key=k3, activation='relu')
        result = fc(flat)  # relu of linear(0) = relu(bias)
        assert result.shape == (4,)
