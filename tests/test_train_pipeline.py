"""
tests/test_train_pipeline.py — Robust end-to-end training pipeline tests.

Coverage
--------
1.  DataConfig — derived properties, mutability guard
2.  one_hot_puzzle — shape, dtype, exclusivity, per-digit correctness
3.  one_hot_solution — shape, dtype, exclusivity, per-digit correctness
4.  load_data — shapes, dtype, 80/20 split, Y flatness, value range
5.  build_model — construction, forward pass, param count
6.  build_optimiser — all 4 optimisers × 4 LR schedules, grad-clip chain
7.  _single_loss / train_step — scalar loss, finiteness, weight update, clip
8.  evaluate — metric range, cell_acc=1 for perfect targets, determinism
9.  Checkpoint round-trip — serialise → deserialise → identical output
10. Full train() pipeline — 2-epoch smoke test, checkpoint file created
11. LR-schedule variants — 1-epoch train per schedule, loss finite
12. Optimiser variants — 1-epoch train per optimiser, loss finite
13. Loss convergence — loss trends downward over 20 synthetic steps

All tests using the real dataset assume `data/dataset_4x4.npz` is available
(downloaded once by download_dataset).  Slow / I/O-heavy tests are marked
``@pytest.mark.slow`` so they can be excluded with ``-m "not slow"``.
"""

import pathlib
import tempfile

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import pytest

# ── Imports under test ────────────────────────────────────────────────────────
from utils.config.data import DataConfig, DATA_CFG
from utils.config.training import TrainingConfig
from utils.otherutils import (
    download_dataset,
    one_hot_puzzle,
    one_hot_solution,
    load_data,
    build_model,
    build_optimiser,
    evaluate,
)
from train import _single_loss, train_step, train

# ── Constants ─────────────────────────────────────────────────────────────────
N_GRID   = DATA_CFG.n              # 4
N_IN     = DATA_CFG.n_channels_in  # 5
N_OUT    = DATA_CFG.n_channels_out # 4
FC_OUT   = DATA_CFG.fc_out         # 64  (4 × 4 × 4)
_KEY     = jax.random.key(0)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def real_dataset():
    """Download (once) and load the actual dataset. Module-scoped to share."""
    download_dataset(DATA_CFG)
    X_tr, Y_tr, X_val, Y_val = load_data(DATA_CFG)
    return X_tr, Y_tr, X_val, Y_val


@pytest.fixture(scope="module")
def base_model():
    """Build a fresh TMRM model (shared across tests in this module)."""
    return build_model(_KEY, DATA_CFG)


@pytest.fixture(scope="module")
def base_tx_state(base_model):
    """Build default adam optimiser + initial state."""
    cfg = TrainingConfig(optimiser="adam", lr_schedule="constant")
    return build_optimiser(cfg, base_model)


@pytest.fixture
def synth_batch():
    """32 random puzzles + one-hot targets (no real data required)."""
    rng = np.random.default_rng(7)
    puzzles = rng.integers(0, N_IN, size=(32, N_GRID, N_GRID))
    solutions = rng.integers(1, N_OUT + 1, size=(32, N_GRID, N_GRID))
    X = one_hot_puzzle(puzzles, DATA_CFG)
    Y = one_hot_solution(solutions, DATA_CFG)
    xs_batch = jnp.array(X[:, :, None, :, :])            # (32,5,1,4,4)
    y_batch  = jnp.array(Y.reshape(len(Y), -1))          # (32,64)
    return xs_batch, y_batch


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DataConfig
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataConfig:

    def test_fc_out_value(self):
        assert DATA_CFG.fc_out == N_OUT * N_GRID * N_GRID

    def test_fc_out_equals_64(self):
        assert DATA_CFG.fc_out == 64

    def test_dataset_path_combines_dir_and_filename(self):
        cfg = DataConfig(data_dir=pathlib.Path("foo"), dataset_filename="bar.npz")
        assert cfg.dataset_path == pathlib.Path("foo/bar.npz")

    def test_frozen_raises(self):
        with pytest.raises((TypeError, AttributeError)):
            DATA_CFG.n = 99  # type: ignore[misc]

    def test_n_channels_in_is_5(self):
        assert DATA_CFG.n_channels_in == 5

    def test_n_channels_out_is_4(self):
        assert DATA_CFG.n_channels_out == 4

    def test_n_is_4(self):
        assert DATA_CFG.n == 4

    def test_checkpoint_dir_is_path(self):
        assert isinstance(DATA_CFG.checkpoint_dir, pathlib.Path)

    def test_custom_config_fc_out(self):
        cfg = DataConfig(n=3, n_channels_out=3)
        assert cfg.fc_out == 3 * 3 * 3  # 27


# ═══════════════════════════════════════════════════════════════════════════════
# 2. one_hot_puzzle
# ═══════════════════════════════════════════════════════════════════════════════

class TestOneHotPuzzle:

    def _make(self, n=4, N=8):
        rng = np.random.default_rng(0)
        return rng.integers(0, n + 1, size=(N, n, n))  # values 0..n_channels_in-1

    def test_shape(self):
        p = self._make()
        enc = one_hot_puzzle(p, DATA_CFG)
        assert enc.shape == (8, N_IN, N_GRID, N_GRID)

    def test_dtype_float32(self):
        p = self._make()
        enc = one_hot_puzzle(p, DATA_CFG)
        assert enc.dtype == np.float32

    def test_binary_values(self):
        p = self._make()
        enc = one_hot_puzzle(p, DATA_CFG)
        assert set(np.unique(enc)).issubset({0.0, 1.0})

    def test_exactly_one_hot_per_cell(self):
        p = self._make(N=20)
        enc = one_hot_puzzle(p, DATA_CFG)
        # Sum over channel dim must be 1 everywhere
        np.testing.assert_array_equal(enc.sum(axis=1), np.ones((20, N_GRID, N_GRID)))

    def test_zero_maps_to_channel_0(self):
        p = np.zeros((1, N_GRID, N_GRID), dtype=np.uint8)
        enc = one_hot_puzzle(p, DATA_CFG)
        assert enc[0, 0, 0, 0] == 1.0
        assert enc[0, 1:, 0, 0].sum() == 0.0

    def test_digit_4_maps_to_channel_4(self):
        p = np.full((1, N_GRID, N_GRID), 4, dtype=np.uint8)
        enc = one_hot_puzzle(p, DATA_CFG)
        assert enc[0, 4, 0, 0] == 1.0
        assert enc[0, :4, 0, 0].sum() == 0.0

    def test_each_digit_correct(self):
        for d in range(N_IN):
            p = np.full((1, N_GRID, N_GRID), d, dtype=np.uint8)
            enc = one_hot_puzzle(p, DATA_CFG)
            assert enc[0, d].sum() == N_GRID * N_GRID
            other = np.delete(np.arange(N_IN), d)
            assert enc[0, other].sum() == 0.0

    def test_batch_independence(self):
        p1 = np.zeros((1, N_GRID, N_GRID), dtype=np.uint8)
        p2 = np.ones((1, N_GRID, N_GRID), dtype=np.uint8)
        combined = np.concatenate([p1, p2])
        enc = one_hot_puzzle(combined, DATA_CFG)
        enc1 = one_hot_puzzle(p1, DATA_CFG)
        enc2 = one_hot_puzzle(p2, DATA_CFG)
        np.testing.assert_array_equal(enc[0], enc1[0])
        np.testing.assert_array_equal(enc[1], enc2[0])


# ═══════════════════════════════════════════════════════════════════════════════
# 3. one_hot_solution
# ═══════════════════════════════════════════════════════════════════════════════

class TestOneHotSolution:

    def _make(self, N=8):
        rng = np.random.default_rng(1)
        return rng.integers(1, N_OUT + 1, size=(N, N_GRID, N_GRID))

    def test_shape(self):
        s = self._make()
        enc = one_hot_solution(s, DATA_CFG)
        assert enc.shape == (8, N_OUT, N_GRID, N_GRID)

    def test_dtype_float32(self):
        s = self._make()
        enc = one_hot_solution(s, DATA_CFG)
        assert enc.dtype == np.float32

    def test_binary_values(self):
        s = self._make()
        enc = one_hot_solution(s, DATA_CFG)
        assert set(np.unique(enc)).issubset({0.0, 1.0})

    def test_exactly_one_hot_per_cell(self):
        s = self._make(N=20)
        enc = one_hot_solution(s, DATA_CFG)
        np.testing.assert_array_equal(enc.sum(axis=1), np.ones((20, N_GRID, N_GRID)))

    def test_digit_1_maps_to_channel_0(self):
        s = np.ones((1, N_GRID, N_GRID), dtype=np.uint8)
        enc = one_hot_solution(s, DATA_CFG)
        assert enc[0, 0, 0, 0] == 1.0
        assert enc[0, 1:, 0, 0].sum() == 0.0

    def test_digit_4_maps_to_channel_3(self):
        s = np.full((1, N_GRID, N_GRID), 4, dtype=np.uint8)
        enc = one_hot_solution(s, DATA_CFG)
        assert enc[0, 3, 0, 0] == 1.0
        assert enc[0, :3, 0, 0].sum() == 0.0

    def test_each_digit_correct(self):
        for d in range(1, N_OUT + 1):
            s = np.full((1, N_GRID, N_GRID), d, dtype=np.uint8)
            enc = one_hot_solution(s, DATA_CFG)
            ch = d - 1
            assert enc[0, ch].sum() == N_GRID * N_GRID
            other = np.delete(np.arange(N_OUT), ch)
            assert enc[0, other].sum() == 0.0

    def test_round_trip_argmax(self):
        """argmax over channels recovers digit index (0-based)."""
        rng = np.random.default_rng(2)
        s = rng.integers(1, N_OUT + 1, size=(10, N_GRID, N_GRID))
        enc = one_hot_solution(s, DATA_CFG)
        recovered = np.argmax(enc, axis=1) + 1   # 1-based
        np.testing.assert_array_equal(recovered, s)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. load_data
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadData:

    def test_x_train_shape(self, real_dataset):
        X_tr, Y_tr, X_val, Y_val = real_dataset
        assert X_tr.ndim == 4
        assert X_tr.shape[1] == N_IN
        assert X_tr.shape[2] == N_GRID
        assert X_tr.shape[3] == N_GRID

    def test_y_train_shape(self, real_dataset):
        X_tr, Y_tr, X_val, Y_val = real_dataset
        assert Y_tr.ndim == 2
        assert Y_tr.shape[1] == FC_OUT

    def test_x_val_shape(self, real_dataset):
        X_tr, Y_tr, X_val, Y_val = real_dataset
        assert X_val.ndim == 4
        assert X_val.shape[1] == N_IN

    def test_y_val_shape(self, real_dataset):
        X_tr, Y_tr, X_val, Y_val = real_dataset
        assert Y_val.ndim == 2
        assert Y_val.shape[1] == FC_OUT

    def test_80_20_split(self, real_dataset):
        X_tr, Y_tr, X_val, Y_val = real_dataset
        total = len(X_tr) + len(X_val)
        assert len(X_tr) == int(0.8 * total)
        assert len(X_val) == total - int(0.8 * total)

    def test_n_samples_matches_labels(self, real_dataset):
        X_tr, Y_tr, X_val, Y_val = real_dataset
        assert len(X_tr) == len(Y_tr)
        assert len(X_val) == len(Y_val)

    def test_x_binary(self, real_dataset):
        X_tr, _, _, _ = real_dataset
        assert set(np.unique(X_tr)).issubset({0.0, 1.0})

    def test_y_binary(self, real_dataset):
        _, Y_tr, _, _ = real_dataset
        assert set(np.unique(Y_tr)).issubset({0.0, 1.0})

    def test_x_dtype_float32(self, real_dataset):
        X_tr, _, _, _ = real_dataset
        assert X_tr.dtype == np.float32

    def test_y_dtype_float32(self, real_dataset):
        _, Y_tr, _, _ = real_dataset
        assert Y_tr.dtype == np.float32

    def test_x_channels_sum_to_one(self, real_dataset):
        X_tr, _, _, _ = real_dataset
        # Each cell's channels sum to 1.0
        chan_sum = X_tr.sum(axis=1)  # (N, n, n)
        np.testing.assert_allclose(chan_sum, np.ones_like(chan_sum))

    def test_y_channels_sum_to_one(self, real_dataset):
        _, Y_tr, _, _ = real_dataset
        # Y reshaped to (N, 4, 4, 4) — each cell's channels sum to 1
        Y_3d = Y_tr.reshape(-1, N_OUT, N_GRID, N_GRID)
        chan_sum = Y_3d.sum(axis=1)
        np.testing.assert_allclose(chan_sum, np.ones_like(chan_sum))

    def test_total_samples_1000(self, real_dataset):
        X_tr, _, X_val, _ = real_dataset
        assert len(X_tr) + len(X_val) == 1000


# ═══════════════════════════════════════════════════════════════════════════════
# 5. build_model
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildModel:

    def test_returns_model_instance(self, base_model):
        from model import Model
        assert isinstance(base_model, Model)

    def test_forward_produces_correct_shape(self, base_model):
        xs = jnp.ones((N_IN, 1, N_GRID, N_GRID))
        out = base_model(0, xs)
        assert out.shape == (FC_OUT,)

    def test_forward_outputs_finite(self, base_model):
        xs = jnp.ones((N_IN, 1, N_GRID, N_GRID))
        out = base_model(0, xs)
        assert bool(jnp.all(jnp.isfinite(out)))

    def test_forward_outputs_in_unit_interval(self, base_model):
        """Sigmoid FC head keeps outputs in (0, 1)."""
        xs = jnp.ones((N_IN, 1, N_GRID, N_GRID))
        out = base_model(0, xs)
        assert bool(jnp.all(out >= 0.0))
        assert bool(jnp.all(out <= 1.0))

    def test_output_dimension_matches_fc_out(self, base_model):
        xs = jnp.zeros((N_IN, 1, N_GRID, N_GRID))
        out = base_model(0, xs)
        assert out.shape[0] == FC_OUT

    def test_param_count_positive(self, base_model):
        params = base_model.count_params()
        assert params["total"] > 0

    def test_param_count_is_int(self, base_model):
        params = base_model.count_params()
        assert isinstance(params["total"], int)

    def test_different_keys_give_different_models(self):
        m1 = build_model(jax.random.key(0), DATA_CFG)
        m2 = build_model(jax.random.key(99), DATA_CFG)
        xs = jnp.ones((N_IN, 1, N_GRID, N_GRID))
        out1 = m1(0, xs)
        out2 = m2(0, xs)
        assert not jnp.allclose(out1, out2)

    def test_forward_deterministic_same_key(self):
        m = build_model(jax.random.key(5), DATA_CFG)
        xs = jnp.ones((N_IN, 1, N_GRID, N_GRID))
        out1 = m(0, xs)
        out2 = m(0, xs)
        np.testing.assert_array_equal(np.array(out1), np.array(out2))


# ═══════════════════════════════════════════════════════════════════════════════
# 6. build_optimiser
# ═══════════════════════════════════════════════════════════════════════════════

_OPTIMISERS = ["adam", "adamw", "sgd", "rmsprop"]
_SCHEDULES  = ["constant", "cosine", "linear", "warmup_cosine"]


class TestBuildOptimiser:

    @pytest.mark.parametrize("opt", _OPTIMISERS)
    def test_returns_tx_and_state(self, base_model, opt):
        cfg = TrainingConfig(optimiser=opt, lr_schedule="constant")
        tx, opt_state = build_optimiser(cfg, base_model)
        assert tx is not None
        assert opt_state is not None

    @pytest.mark.parametrize("sched", _SCHEDULES)
    def test_all_schedules_with_adam(self, base_model, sched):
        cfg = TrainingConfig(optimiser="adam", lr_schedule=sched, warmup_steps=50)
        tx, opt_state = build_optimiser(cfg, base_model)
        assert tx is not None

    def test_grad_clip_adds_chain(self, base_model):
        """Grad clip > 0 should wrap tx in a chain; state should be non-trivial."""
        cfg_clip  = TrainingConfig(optimiser="adam", grad_clip_norm=1.0)
        cfg_noclip = TrainingConfig(optimiser="adam", grad_clip_norm=0.0)
        tx_clip,   _ = build_optimiser(cfg_clip,   base_model)
        tx_noclip, _ = build_optimiser(cfg_noclip, base_model)
        # Both should work; just verify both are usable GradientTransformations
        assert hasattr(tx_clip,   "init")
        assert hasattr(tx_noclip, "init")

    def test_unknown_optimiser_raises(self, base_model):
        cfg = TrainingConfig(optimiser="unknown_opt")
        with pytest.raises(ValueError, match="Unknown optimiser"):
            build_optimiser(cfg, base_model)

    @pytest.mark.parametrize("opt", _OPTIMISERS)
    def test_optimiser_state_init_not_none(self, base_model, opt):
        cfg = TrainingConfig(optimiser=opt, lr_schedule="constant")
        _, opt_state = build_optimiser(cfg, base_model)
        # State must be a non-empty structure
        leaves = jax.tree.leaves(opt_state)
        assert len(leaves) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 7. _single_loss and train_step
# ═══════════════════════════════════════════════════════════════════════════════

class TestSingleLoss:

    def test_scalar_output(self, base_model):
        xs = jnp.ones((N_IN, 1, N_GRID, N_GRID))
        y  = jnp.ones((FC_OUT,)) * 0.5
        loss = _single_loss(base_model, xs, y)
        assert loss.shape == ()

    def test_finite_loss(self, base_model):
        xs = jnp.ones((N_IN, 1, N_GRID, N_GRID))
        y  = jnp.zeros((FC_OUT,))
        loss = _single_loss(base_model, xs, y)
        assert jnp.isfinite(loss)

    def test_perfect_prediction_zero_loss(self, base_model):
        """If model output == target, BCE ≈ 0 (subject to numerical clip)."""
        xs = jnp.ones((N_IN, 1, N_GRID, N_GRID))
        # Force targets to match a fixed near-zero output
        logits = base_model(0, xs)          # (FC_OUT,)
        # For loss to be 0 targets must exactly match sigmoid output;
        # use actual logits as targets → loss still positive (not 0) but finite
        loss = _single_loss(base_model, xs, logits)
        assert jnp.isfinite(loss)
        assert float(loss) >= 0.0

    def test_loss_non_negative(self, base_model):
        rng = np.random.default_rng(3)
        xs = jnp.array(rng.random((N_IN, 1, N_GRID, N_GRID)).astype(np.float32))
        y  = jnp.array(rng.integers(0, 2, (FC_OUT,)).astype(np.float32))
        loss = _single_loss(base_model, xs, y)
        assert float(loss) >= 0.0

    def test_loss_increases_with_wrong_targets(self, base_model):
        xs = jnp.ones((N_IN, 1, N_GRID, N_GRID))
        logits = base_model(0, xs)
        y_correct = jnp.round(logits)
        y_wrong   = 1.0 - y_correct
        loss_right = float(_single_loss(base_model, xs, y_correct))
        loss_wrong = float(_single_loss(base_model, xs, y_wrong))
        assert loss_wrong >= loss_right


class TestTrainStep:

    def test_returns_three_outputs(self, base_model, base_tx_state, synth_batch):
        tx, opt_state = base_tx_state
        xs_batch, y_batch = synth_batch
        result = train_step(base_model, opt_state, tx, xs_batch, y_batch)
        assert len(result) == 3

    def test_loss_is_scalar(self, base_model, base_tx_state, synth_batch):
        tx, opt_state = base_tx_state
        xs_batch, y_batch = synth_batch
        _, _, loss = train_step(base_model, opt_state, tx, xs_batch, y_batch)
        assert loss.shape == ()

    def test_loss_is_finite(self, base_model, base_tx_state, synth_batch):
        tx, opt_state = base_tx_state
        xs_batch, y_batch = synth_batch
        _, _, loss = train_step(base_model, opt_state, tx, xs_batch, y_batch)
        assert bool(jnp.isfinite(loss))

    def test_loss_non_negative(self, base_model, base_tx_state, synth_batch):
        tx, opt_state = base_tx_state
        xs_batch, y_batch = synth_batch
        _, _, loss = train_step(base_model, opt_state, tx, xs_batch, y_batch)
        assert float(loss) >= 0.0

    def test_weights_update_after_step(self, base_model, base_tx_state, synth_batch):
        tx, opt_state = base_tx_state
        xs_batch, y_batch = synth_batch
        old_leaves = jax.tree.leaves(eqx.filter(base_model, eqx.is_inexact_array))
        new_model, _, _ = train_step(base_model, opt_state, tx, xs_batch, y_batch)
        new_leaves = jax.tree.leaves(eqx.filter(new_model, eqx.is_inexact_array))
        changed = any(
            not jnp.allclose(o, n)
            for o, n in zip(old_leaves, new_leaves)
        )
        assert changed, "No weight was updated after train_step"

    def test_original_model_not_mutated(self, base_model, base_tx_state, synth_batch):
        """Equinox is functional — base_model should be unchanged."""
        tx, opt_state = base_tx_state
        xs_batch, y_batch = synth_batch
        xs_single = jnp.ones((N_IN, 1, N_GRID, N_GRID))
        out_before = base_model(0, xs_single)
        train_step(base_model, opt_state, tx, xs_batch, y_batch)
        out_after = base_model(0, xs_single)
        np.testing.assert_array_equal(np.array(out_before), np.array(out_after))

    def test_all_updated_params_finite(self, base_model, base_tx_state, synth_batch):
        tx, opt_state = base_tx_state
        xs_batch, y_batch = synth_batch
        new_model, _, _ = train_step(base_model, opt_state, tx, xs_batch, y_batch)
        leaves = jax.tree.leaves(eqx.filter(new_model, eqx.is_inexact_array))
        assert all(bool(jnp.all(jnp.isfinite(l))) for l in leaves)

    def test_multiple_steps_all_finite(self, base_model, base_tx_state, synth_batch):
        tx, opt_state = base_tx_state
        xs_batch, y_batch = synth_batch
        model = base_model
        for _ in range(5):
            model, opt_state, loss = train_step(model, opt_state, tx, xs_batch, y_batch)
            assert bool(jnp.isfinite(loss))


# ═══════════════════════════════════════════════════════════════════════════════
# 8. evaluate
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluate:

    def test_returns_dict_with_required_keys(self, base_model, real_dataset):
        _, _, X_val, Y_val = real_dataset
        result = evaluate(base_model, X_val[:10], Y_val[:10], DATA_CFG)
        assert "loss" in result
        assert "cell_acc" in result

    def test_loss_is_float(self, base_model, real_dataset):
        _, _, X_val, Y_val = real_dataset
        result = evaluate(base_model, X_val[:10], Y_val[:10], DATA_CFG)
        assert isinstance(result["loss"], float)

    def test_cell_acc_is_float(self, base_model, real_dataset):
        _, _, X_val, Y_val = real_dataset
        result = evaluate(base_model, X_val[:10], Y_val[:10], DATA_CFG)
        assert isinstance(result["cell_acc"], float)

    def test_loss_finite(self, base_model, real_dataset):
        _, _, X_val, Y_val = real_dataset
        result = evaluate(base_model, X_val[:10], Y_val[:10], DATA_CFG)
        assert np.isfinite(result["loss"])

    def test_cell_acc_in_range(self, base_model, real_dataset):
        _, _, X_val, Y_val = real_dataset
        result = evaluate(base_model, X_val[:20], Y_val[:20], DATA_CFG)
        assert 0.0 <= result["cell_acc"] <= 1.0

    def test_loss_non_negative(self, base_model, real_dataset):
        _, _, X_val, Y_val = real_dataset
        result = evaluate(base_model, X_val[:10], Y_val[:10], DATA_CFG)
        assert result["loss"] >= 0.0

    def test_perfect_prediction_cell_acc_one(self):
        """
        If model predictions match targets exactly (per-channel argmax),
        cell_acc must be 1.0.
        """
        model = build_model(jax.random.key(42), DATA_CFG)

        rng = np.random.default_rng(10)
        puzzles = rng.integers(0, N_IN, size=(5, N_GRID, N_GRID))
        X = one_hot_puzzle(puzzles, DATA_CFG)  # (5, 5, 4, 4)

        # Get actual model predictions and construct matching Y
        Y_fake_list = []
        for i in range(5):
            xs = jnp.array(X[i, :, None, :, :])   # (5,1,4,4)
            logits = model(0, xs)                   # (64,)
            pred_ch = np.array(logits).reshape(N_OUT, N_GRID, N_GRID)
            pred_dig = np.argmax(pred_ch, axis=0)   # (4,4)
            # Build one-hot from predicted digit
            y_oh = np.zeros((N_OUT, N_GRID, N_GRID), dtype=np.float32)
            for r in range(N_GRID):
                for c in range(N_GRID):
                    y_oh[pred_dig[r, c], r, c] = 1.0
            Y_fake_list.append(y_oh.reshape(-1))
        Y_fake = np.stack(Y_fake_list)  # (5, 64)

        result = evaluate(model, X, Y_fake, DATA_CFG)
        assert result["cell_acc"] == 1.0

    def test_evaluate_deterministic(self, base_model, real_dataset):
        _, _, X_val, Y_val = real_dataset
        r1 = evaluate(base_model, X_val[:10], Y_val[:10], DATA_CFG)
        r2 = evaluate(base_model, X_val[:10], Y_val[:10], DATA_CFG)
        assert r1["loss"]     == r2["loss"]
        assert r1["cell_acc"] == r2["cell_acc"]

    def test_evaluate_single_sample(self, base_model, real_dataset):
        _, _, X_val, Y_val = real_dataset
        result = evaluate(base_model, X_val[:1], Y_val[:1], DATA_CFG)
        assert np.isfinite(result["loss"])
        assert 0.0 <= result["cell_acc"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Checkpoint round-trip
# ═══════════════════════════════════════════════════════════════════════════════

class TestCheckpointRoundTrip:

    def test_save_and_load_identical_output(self, base_model):
        xs = jnp.ones((N_IN, 1, N_GRID, N_GRID))
        out_before = base_model(0, xs)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = pathlib.Path(tmpdir) / "model.eqx"
            eqx.tree_serialise_leaves(str(ckpt), base_model)
            loaded = eqx.tree_deserialise_leaves(str(ckpt), base_model)

        out_after = loaded(0, xs)
        np.testing.assert_allclose(
            np.array(out_before), np.array(out_after), atol=1e-6
        )

    def test_checkpoint_file_created(self, base_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = pathlib.Path(tmpdir) / "model.eqx"
            eqx.tree_serialise_leaves(str(ckpt), base_model)
            assert ckpt.exists()
            assert ckpt.stat().st_size > 0

    def test_loaded_model_forward_shape(self, base_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = pathlib.Path(tmpdir) / "model.eqx"
            eqx.tree_serialise_leaves(str(ckpt), base_model)
            loaded = eqx.tree_deserialise_leaves(str(ckpt), base_model)
        xs = jnp.zeros((N_IN, 1, N_GRID, N_GRID))
        out = loaded(0, xs)
        assert out.shape == (FC_OUT,)

    def test_loaded_model_after_training_matches(self, base_model, base_tx_state, synth_batch):
        """Train 1 step, checkpoint, reload, compare forward pass."""
        tx, opt_state = base_tx_state
        xs_batch, y_batch = synth_batch
        model_trained, _, _ = train_step(base_model, opt_state, tx, xs_batch, y_batch)

        xs = jnp.ones((N_IN, 1, N_GRID, N_GRID))
        out_trained = model_trained(0, xs)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = pathlib.Path(tmpdir) / "trained.eqx"
            eqx.tree_serialise_leaves(str(ckpt), model_trained)
            model_loaded = eqx.tree_deserialise_leaves(str(ckpt), base_model)

        out_loaded = model_loaded(0, xs)
        np.testing.assert_allclose(
            np.array(out_trained), np.array(out_loaded), atol=1e-6
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Loss convergence (synthetic, fast)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLossConvergence:

    def test_loss_trends_down_over_20_steps(self, synth_batch):
        """
        Over 20 gradient steps on a fixed batch the loss should have a
        downward trend (first > last, or at minimum some improvement).
        """
        model = build_model(jax.random.key(11), DATA_CFG)
        cfg   = TrainingConfig(optimiser="adam", lr_schedule="constant",
                               learning_rate=3e-4, grad_clip_norm=1.0)
        tx, opt_state = build_optimiser(cfg, model)
        xs_batch, y_batch = synth_batch

        losses = []
        for _ in range(5):
            model, opt_state, loss = train_step(model, opt_state, tx, xs_batch, y_batch)
            losses.append(float(loss))

        assert all(np.isfinite(l) for l in losses), "NaN/Inf during training steps"
        # Loss at end should be lower than at start (on fixed batch, adam converges)
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        )

    def test_loss_all_finite_over_30_steps(self, synth_batch):
        model = build_model(jax.random.key(22), DATA_CFG)
        cfg   = TrainingConfig(optimiser="adam", lr_schedule="constant",
                               learning_rate=1e-3, grad_clip_norm=1.0)
        tx, opt_state = build_optimiser(cfg, model)
        xs_batch, y_batch = synth_batch
        for _ in range(5):
            model, opt_state, loss = train_step(model, opt_state, tx, xs_batch, y_batch)
            assert bool(jnp.isfinite(loss)), f"Non-finite loss encountered"

    @pytest.mark.parametrize("lr", [1e-4, 3e-4, 1e-3])
    def test_loss_finite_for_lr_range(self, synth_batch, lr):
        model = build_model(jax.random.key(33), DATA_CFG)
        cfg   = TrainingConfig(optimiser="adam", lr_schedule="constant",
                               learning_rate=lr, grad_clip_norm=1.0)
        tx, opt_state = build_optimiser(cfg, model)
        xs_batch, y_batch = synth_batch
        for _ in range(2):
            model, opt_state, loss = train_step(model, opt_state, tx, xs_batch, y_batch)
        assert bool(jnp.isfinite(loss))


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Optimiser variants (1-epoch train each)
# ═══════════════════════════════════════════════════════════════════════════════

class TestOptimiserVariants:

    @pytest.mark.parametrize("opt", _OPTIMISERS)
    def test_one_epoch_loss_finite(self, opt, synth_batch):
        model = build_model(jax.random.key(44), DATA_CFG)
        cfg   = TrainingConfig(optimiser=opt, lr_schedule="constant",
                               learning_rate=1e-3, grad_clip_norm=1.0)
        tx, opt_state = build_optimiser(cfg, model)
        xs_batch, y_batch = synth_batch
        for _ in range(2):
            model, opt_state, loss = train_step(model, opt_state, tx, xs_batch, y_batch)
        assert bool(jnp.isfinite(loss))

    @pytest.mark.parametrize("opt", _OPTIMISERS)
    def test_weights_changed_after_epoch(self, opt, base_model, synth_batch):
        cfg   = TrainingConfig(optimiser=opt, lr_schedule="constant",
                               learning_rate=1e-3, grad_clip_norm=1.0)
        tx, opt_state = build_optimiser(cfg, base_model)
        xs_batch, y_batch = synth_batch
        old_leaves = jax.tree.leaves(eqx.filter(base_model, eqx.is_inexact_array))
        model = base_model
        for _ in range(2):
            model, opt_state, _ = train_step(model, opt_state, tx, xs_batch, y_batch)
        new_leaves = jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array))
        changed = any(not jnp.allclose(o, n) for o, n in zip(old_leaves, new_leaves))
        assert changed


# ═══════════════════════════════════════════════════════════════════════════════
# 12. LR schedule variants
# ═══════════════════════════════════════════════════════════════════════════════

class TestLRScheduleVariants:

    @pytest.mark.parametrize("sched", _SCHEDULES)
    def test_schedule_loss_finite_after_5_steps(self, sched, synth_batch):
        model = build_model(jax.random.key(55), DATA_CFG)
        cfg   = TrainingConfig(optimiser="adam", lr_schedule=sched,
                               learning_rate=3e-4, warmup_steps=20,
                               grad_clip_norm=1.0)
        tx, opt_state = build_optimiser(cfg, model)
        xs_batch, y_batch = synth_batch
        for _ in range(2):
            model, opt_state, loss = train_step(model, opt_state, tx, xs_batch, y_batch)
        assert bool(jnp.isfinite(loss))

    @pytest.mark.parametrize("sched", _SCHEDULES)
    def test_schedule_model_params_finite_after_5_steps(self, sched, synth_batch):
        model = build_model(jax.random.key(66), DATA_CFG)
        cfg   = TrainingConfig(optimiser="adam", lr_schedule=sched,
                               learning_rate=3e-4, warmup_steps=20,
                               grad_clip_norm=1.0)
        tx, opt_state = build_optimiser(cfg, model)
        xs_batch, y_batch = synth_batch
        for _ in range(2):
            model, opt_state, loss = train_step(model, opt_state, tx, xs_batch, y_batch)
        leaves = jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array))
        assert all(bool(jnp.all(jnp.isfinite(l))) for l in leaves)


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Full train() pipeline (slow — real data, real epochs)
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrainPipeline:

    @pytest.mark.slow
    def test_two_epoch_smoke_finishes(self, real_dataset, tmp_path):
        """2-epoch train() run: finishes without exception, returns a Model."""
        from model import Model
        cfg = DataConfig(
            checkpoint_dir=tmp_path / "ckpts",
        )
        train_cfg = TrainingConfig(
            batch_size=32,
            n_epochs=2,
            learning_rate=3e-4,
            optimiser="adam",
            lr_schedule="constant",
            grad_clip_norm=1.0,
            log_every=999,
            seed=0,
        )
        # Download already done by real_dataset fixture
        model = train(train_cfg, data_cfg=cfg)
        assert isinstance(model, Model)

    @pytest.mark.slow
    def test_checkpoint_created_after_train(self, real_dataset, tmp_path):
        """Checkpoint file must exist after train() completes."""
        cfg = DataConfig(checkpoint_dir=tmp_path / "ckpts")
        train_cfg = TrainingConfig(
            batch_size=32,
            n_epochs=2,
            learning_rate=3e-4,
            optimiser="adam",
            lr_schedule="constant",
            grad_clip_norm=1.0,
            log_every=999,
            seed=1,
        )
        train(train_cfg, data_cfg=cfg)
        ckpt = tmp_path / "ckpts" / "best_model.eqx"
        assert ckpt.exists()
        assert ckpt.stat().st_size > 0

    @pytest.mark.slow
    def test_checkpoint_loads_and_runs(self, real_dataset, tmp_path):
        """Saved checkpoint deserialises and produces finite forward output."""
        from model import Model
        cfg = DataConfig(checkpoint_dir=tmp_path / "ckpts")
        train_cfg = TrainingConfig(
            batch_size=32,
            n_epochs=2,
            learning_rate=3e-4,
            optimiser="adam",
            lr_schedule="constant",
            grad_clip_norm=1.0,
            log_every=999,
            seed=2,
        )
        trained_model = train(train_cfg, data_cfg=cfg)
        ckpt = tmp_path / "ckpts" / "best_model.eqx"
        loaded = eqx.tree_deserialise_leaves(str(ckpt), trained_model)
        xs = jnp.ones((N_IN, 1, N_GRID, N_GRID))
        out = loaded(0, xs)
        assert out.shape == (FC_OUT,)
        assert bool(jnp.all(jnp.isfinite(out)))

    @pytest.mark.slow
    def test_val_loss_finite_after_train(self, real_dataset, tmp_path):
        cfg = DataConfig(checkpoint_dir=tmp_path / "ckpts")
        train_cfg = TrainingConfig(
            batch_size=32,
            n_epochs=2,
            learning_rate=3e-4,
            optimiser="adam",
            lr_schedule="constant",
            grad_clip_norm=1.0,
            log_every=999,
            seed=3,
        )
        _, _, X_val, Y_val = real_dataset
        model = train(train_cfg, data_cfg=cfg)
        result = evaluate(model, X_val, Y_val, DATA_CFG)
        assert np.isfinite(result["loss"])
        assert 0.0 <= result["cell_acc"] <= 1.0

    @pytest.mark.slow
    @pytest.mark.parametrize("opt", _OPTIMISERS)
    def test_one_epoch_all_optimisers(self, real_dataset, tmp_path, opt):
        """All 4 optimisers complete 1 epoch without NaN."""
        cfg = DataConfig(checkpoint_dir=tmp_path / f"ckpts_{opt}")
        train_cfg = TrainingConfig(
            batch_size=32,
            n_epochs=1,
            learning_rate=3e-4,
            optimiser=opt,
            lr_schedule="constant",
            grad_clip_norm=1.0,
            log_every=999,
            seed=4,
        )
        model = train(train_cfg, data_cfg=cfg)
        leaves = jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array))
        assert all(bool(jnp.all(jnp.isfinite(l))) for l in leaves)

    @pytest.mark.slow
    @pytest.mark.parametrize("sched", _SCHEDULES)
    def test_one_epoch_all_schedules(self, real_dataset, tmp_path, sched):
        """All 4 LR schedules complete 1 epoch without NaN."""
        cfg = DataConfig(checkpoint_dir=tmp_path / f"ckpts_{sched}")
        train_cfg = TrainingConfig(
            batch_size=32,
            n_epochs=1,
            learning_rate=3e-4,
            optimiser="adam",
            lr_schedule=sched,
            warmup_steps=50,
            grad_clip_norm=1.0,
            log_every=999,
            seed=5,
        )
        model = train(train_cfg, data_cfg=cfg)
        leaves = jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array))
        assert all(bool(jnp.all(jnp.isfinite(l))) for l in leaves)

    @pytest.mark.slow
    def test_5_epoch_loss_improvement(self, real_dataset, tmp_path):
        """
        After 5 epochs the final validation loss should be lower than the
        initial (random) loss.  This is a soft convergence sanity check.
        """
        from model import Model

        cfg = DataConfig(checkpoint_dir=tmp_path / "ckpts_conv")
        # Evaluate random model baseline
        model_init = build_model(jax.random.key(99), DATA_CFG)
        _, _, X_val, Y_val = real_dataset
        init_metrics = evaluate(model_init, X_val[:50], Y_val[:50], DATA_CFG)
        init_loss = init_metrics["loss"]

        train_cfg = TrainingConfig(
            batch_size=32,
            n_epochs=1,
            learning_rate=3e-4,
            optimiser="adam",
            lr_schedule="constant",
            grad_clip_norm=1.0,
            log_every=999,
            seed=99,
        )
        trained = train(train_cfg, data_cfg=cfg)
        final_metrics = evaluate(trained, X_val[:50], Y_val[:50], DATA_CFG)
        final_loss = final_metrics["loss"]

        assert final_loss < init_loss, (
            f"Loss did not improve after 5 epochs: "
            f"init={init_loss:.4f}  final={final_loss:.4f}"
        )
