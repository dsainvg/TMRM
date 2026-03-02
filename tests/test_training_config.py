"""
Tests for TrainingConfig validation and Model.count_params().

Covers:
  - TrainingConfig defaults
  - TrainingConfig custom values
  - All validation edges (batch_size, n_epochs, learning_rate, optimiser,
    weight_decay, lr_schedule, warmup_steps, grad_clip_norm, log_every, seed)
  - Frozen immutability
  - Supported optimiser and schedule names
  - count_params() structure and correctness
  - count_params() total consistency
"""

import pytest
import jax
import equinox as eqx

from utils.config.training import TrainingConfig, _VALID_OPTIMISERS, _VALID_SCHEDULES
from utils.config.model import ModelConfig, ProblemConfig
from model import Model


# ══════════════════════════════════════════════════════════════════════════════
# 1.  TrainingConfig — defaults & happy paths
# ══════════════════════════════════════════════════════════════════════════════

class TestTrainingConfigDefaults:

    def test_default_values(self):
        tc = TrainingConfig()
        assert tc.batch_size == 1
        assert tc.n_epochs == 50
        assert tc.learning_rate == 3e-4
        assert tc.optimiser == 'adam'
        assert tc.weight_decay == 0.0
        assert tc.lr_schedule == 'constant'
        assert tc.warmup_steps == 0
        assert tc.grad_clip_norm == 0.0
        assert tc.log_every == 10
        assert tc.seed == 0

    def test_custom_values(self):
        tc = TrainingConfig(
            batch_size=16,
            n_epochs=100,
            learning_rate=1e-3,
            optimiser='adamw',
            weight_decay=0.01,
            lr_schedule='cosine',
            warmup_steps=500,
            grad_clip_norm=1.0,
            log_every=5,
            seed=99,
        )
        assert tc.batch_size == 16
        assert tc.n_epochs == 100
        assert tc.learning_rate == 1e-3
        assert tc.optimiser == 'adamw'
        assert tc.weight_decay == 0.01
        assert tc.lr_schedule == 'cosine'
        assert tc.warmup_steps == 500
        assert tc.grad_clip_norm == 1.0
        assert tc.log_every == 5
        assert tc.seed == 99


class TestTrainingConfigFrozen:

    def test_is_frozen(self):
        tc = TrainingConfig()
        with pytest.raises(AttributeError):
            tc.batch_size = 4

    def test_cannot_add_field(self):
        tc = TrainingConfig()
        with pytest.raises(AttributeError):
            tc.new_field = "hello"


# ══════════════════════════════════════════════════════════════════════════════
# 2.  TrainingConfig — validation errors
# ══════════════════════════════════════════════════════════════════════════════

class TestTrainingConfigValidation:

    # ── batch_size ────────────────────────────────────────────────────────
    @pytest.mark.parametrize("val", [0, -1, -100])
    def test_batch_size_invalid(self, val):
        with pytest.raises(ValueError, match="batch_size"):
            TrainingConfig(batch_size=val)

    def test_batch_size_one_is_valid(self):
        tc = TrainingConfig(batch_size=1)
        assert tc.batch_size == 1

    # ── n_epochs ──────────────────────────────────────────────────────────
    @pytest.mark.parametrize("val", [0, -1])
    def test_n_epochs_invalid(self, val):
        with pytest.raises(ValueError, match="n_epochs"):
            TrainingConfig(n_epochs=val)

    # ── learning_rate ─────────────────────────────────────────────────────
    @pytest.mark.parametrize("val", [0.0, -1e-4])
    def test_lr_invalid(self, val):
        with pytest.raises(ValueError, match="learning_rate"):
            TrainingConfig(learning_rate=val)

    def test_lr_tiny_valid(self):
        tc = TrainingConfig(learning_rate=1e-10)
        assert tc.learning_rate == 1e-10

    # ── optimiser ─────────────────────────────────────────────────────────
    def test_invalid_optimiser(self):
        with pytest.raises(ValueError, match="optimiser"):
            TrainingConfig(optimiser='adagrad')

    @pytest.mark.parametrize("opt", sorted(_VALID_OPTIMISERS))
    def test_all_valid_optimisers(self, opt):
        tc = TrainingConfig(optimiser=opt)
        assert tc.optimiser == opt

    # ── weight_decay ──────────────────────────────────────────────────────
    def test_negative_weight_decay(self):
        with pytest.raises(ValueError, match="weight_decay"):
            TrainingConfig(weight_decay=-0.01)

    def test_zero_weight_decay_valid(self):
        tc = TrainingConfig(weight_decay=0.0)
        assert tc.weight_decay == 0.0

    # ── lr_schedule ───────────────────────────────────────────────────────
    def test_invalid_schedule(self):
        with pytest.raises(ValueError, match="lr_schedule"):
            TrainingConfig(lr_schedule='step')

    @pytest.mark.parametrize("sched", sorted(_VALID_SCHEDULES))
    def test_all_valid_schedules(self, sched):
        tc = TrainingConfig(lr_schedule=sched)
        assert tc.lr_schedule == sched

    # ── warmup_steps ──────────────────────────────────────────────────────
    def test_negative_warmup(self):
        with pytest.raises(ValueError, match="warmup_steps"):
            TrainingConfig(warmup_steps=-1)

    def test_zero_warmup_valid(self):
        tc = TrainingConfig(warmup_steps=0)
        assert tc.warmup_steps == 0

    # ── grad_clip_norm ────────────────────────────────────────────────────
    def test_negative_clip(self):
        with pytest.raises(ValueError, match="grad_clip_norm"):
            TrainingConfig(grad_clip_norm=-0.5)

    def test_zero_clip_valid(self):
        tc = TrainingConfig(grad_clip_norm=0.0)
        assert tc.grad_clip_norm == 0.0

    # ── log_every ─────────────────────────────────────────────────────────
    @pytest.mark.parametrize("val", [0, -1])
    def test_log_every_invalid(self, val):
        with pytest.raises(ValueError, match="log_every"):
            TrainingConfig(log_every=val)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  TrainingConfig — edge cases
# ══════════════════════════════════════════════════════════════════════════════

class TestTrainingConfigEdgeCases:

    def test_large_batch_size(self):
        tc = TrainingConfig(batch_size=1024)
        assert tc.batch_size == 1024

    def test_large_epochs(self):
        tc = TrainingConfig(n_epochs=10_000)
        assert tc.n_epochs == 10_000

    def test_very_high_lr(self):
        tc = TrainingConfig(learning_rate=100.0)
        assert tc.learning_rate == 100.0

    def test_very_high_grad_clip(self):
        tc = TrainingConfig(grad_clip_norm=1e6)
        assert tc.grad_clip_norm == 1e6

    def test_negative_seed_valid(self):
        """Negative seeds are fine — no validation constraint."""
        tc = TrainingConfig(seed=-42)
        assert tc.seed == -42


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Model.count_params()
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def tiny_model():
    cfg = ModelConfig(
        n=4,
        n_encoders=2,
        n_decoder_layers=1,
        max_decoder_nodes=10,
        problems=(
            ProblemConfig(n_encoders_used=1, fc_out_features=4, fc_activation='relu'),
            ProblemConfig(n_encoders_used=2, fc_out_features=8, fc_activation='identity'),
        ),
    )
    return Model(cfg, key=jax.random.key(0))


class TestCountParams:

    def test_returns_dict(self, tiny_model):
        result = tiny_model.count_params()
        assert isinstance(result, dict)

    def test_keys(self, tiny_model):
        result = tiny_model.count_params()
        assert set(result.keys()) == {
            'encoder_layer', 'decoder_cluster', 'fc_heads', 'total',
        }

    def test_fc_heads_list_length(self, tiny_model):
        result = tiny_model.count_params()
        assert isinstance(result['fc_heads'], list)
        assert len(result['fc_heads']) == tiny_model.n_problems

    def test_total_is_sum(self, tiny_model):
        result = tiny_model.count_params()
        expected = (
            result['encoder_layer']
            + result['decoder_cluster']
            + sum(result['fc_heads'])
        )
        assert result['total'] == expected

    def test_all_positive(self, tiny_model):
        result = tiny_model.count_params()
        assert result['encoder_layer'] > 0
        assert result['decoder_cluster'] > 0
        for fc in result['fc_heads']:
            assert fc > 0
        assert result['total'] > 0

    def test_all_int(self, tiny_model):
        result = tiny_model.count_params()
        assert isinstance(result['encoder_layer'], int)
        assert isinstance(result['decoder_cluster'], int)
        assert isinstance(result['total'], int)
        for fc in result['fc_heads']:
            assert isinstance(fc, int)

    def test_fc_head_sizes_differ_for_different_out_features(self, tiny_model):
        """Problem 0 has fc_out=4, problem 1 has fc_out=8 → different FC sizes."""
        result = tiny_model.count_params()
        assert result['fc_heads'][0] != result['fc_heads'][1]

    def test_encoder_params_match_manual_count(self, tiny_model):
        """Verify encoder count matches direct leaf enumeration."""
        expected = sum(
            x.size for x in jax.tree_util.tree_leaves(
                eqx.filter(tiny_model.encoder_layer, eqx.is_inexact_array)
            )
        )
        assert tiny_model.count_params()['encoder_layer'] == expected

    def test_decoder_params_match_manual_count(self, tiny_model):
        """Verify decoder count matches direct leaf enumeration."""
        expected = sum(
            x.size for x in jax.tree_util.tree_leaves(
                eqx.filter(tiny_model.decoder_cluster, eqx.is_inexact_array)
            )
        )
        assert tiny_model.count_params()['decoder_cluster'] == expected

    def test_deterministic(self, tiny_model):
        """Calling count_params twice gives identical results."""
        a = tiny_model.count_params()
        b = tiny_model.count_params()
        assert a == b

    def test_single_problem_model(self):
        cfg = ModelConfig(
            n=4,
            n_encoders=2,
            n_decoder_layers=1,
            max_decoder_nodes=10,
            problems=(
                ProblemConfig(n_encoders_used=1, fc_out_features=4),
            ),
        )
        m = Model(cfg, key=jax.random.key(7))
        result = m.count_params()
        assert len(result['fc_heads']) == 1
        assert result['total'] == (
            result['encoder_layer']
            + result['decoder_cluster']
            + result['fc_heads'][0]
        )


# ══════════════════════════════════════════════════════════════════════════════
# 5.  TrainingConfig import from package level
# ══════════════════════════════════════════════════════════════════════════════

class TestTrainingConfigImport:

    def test_import_from_utils_config(self):
        from utils.config import TrainingConfig as TC
        tc = TC()
        assert tc.batch_size == 1

    def test_in_all(self):
        import utils.config
        assert 'TrainingConfig' in utils.config.__all__
