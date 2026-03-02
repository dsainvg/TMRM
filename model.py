"""
Multi-problem TMRM Model.

Wraps a shared backbone (EncoderLayer + DecoderCluster) with per-problem
encoder masks and independent FC heads.  Each problem randomly selects a
subset of encoder slots at initialisation; at forward time a Python-int
``problem_idx`` picks the corresponding mask and FC head.

Usage
-----
>>> from utils.config.model import ModelConfig, ProblemConfig
>>> cfg = ModelConfig(
...     n=8,
...     n_encoders=4,
...     n_decoder_layers=2,
...     max_decoder_nodes=50,
...     problems=(
...         ProblemConfig(n_encoders_used=2, fc_out_features=10),
...         ProblemConfig(n_encoders_used=3, fc_out_features=64, fc_activation='identity'),
...     ),
... )
>>> model = Model(cfg, key=jax.random.key(0))
>>> xs = jax.random.normal(jax.random.key(1), (cfg.n_encoders, 1, cfg.n, cfg.n))
>>> out_p0 = model(0, xs)   # problem 0 → shape (10,)
>>> out_p1 = model(1, xs)   # problem 1 → shape (64,)
"""

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from utils.encoder_layer import EncoderLayer
from utils.decoder_cluster import DecoderCluster
from utils.fc_layer import FCLayer
from utils.config.model import ModelConfig
from utils.config.encoder_layer import ENCODER_STACK_OUT_CHANNELS


class Model(eqx.Module):
    """
    Multi-problem TMRM network.

    A single shared backbone (EncoderLayer → DecoderCluster) serves all
    problems.  Each problem has:
      - a random encoder mask selecting which of the ``n_encoders`` slots
        are active (determined once at init, frozen thereafter), and
      - its own independent FCLayer (own output dim + activation).

    ``problem_idx`` is a **Python int** (not a JAX-traced value).  Each
    distinct problem therefore traces its own XLA program — this is
    unavoidable because FC heads may have different output shapes.

    Parameters
    ----------
    config : ModelConfig
        Frozen dataclass holding all structural hyperparameters.
    key : jax.Array
        JAX PRNG key for weight initialisation and mask generation.
    """

    # ── static metadata (invisible to JAX gradient / pytree) ──────────────
    config: ModelConfig = eqx.field(static=True)

    # ── shared trainable backbone ─────────────────────────────────────────
    encoder_layer: EncoderLayer
    decoder_cluster: DecoderCluster

    # ── per-problem trainable heads ───────────────────────────────────────
    fc_heads: list  # list[FCLayer]

    # ── per-problem non-trainable masks (plain NumPy, compile-time const) ─
    encoder_masks: list  # list[np.ndarray]  — each (n_encoders,) bool

    def __init__(self, config: ModelConfig, key: jax.Array):
        self.config = config

        k_enc, k_dec, k_fc, k_mask = jax.random.split(key, 4)

        # ── Shared encoder layer ──────────────────────────────────────────
        self.encoder_layer = EncoderLayer(
            n_inputs=config.n_encoders, key=k_enc,
        )

        # ── Shared decoder cluster ────────────────────────────────────────
        n_encoder_outputs = config.n_encoders * ENCODER_STACK_OUT_CHANNELS
        self.decoder_cluster = DecoderCluster(
            n_layers=config.n_decoder_layers,
            max_nodes=config.max_decoder_nodes,
            n_inputs=n_encoder_outputs,
            key=k_dec,
        )

        # ── Per-problem FC heads ──────────────────────────────────────────
        fc_in = self.decoder_cluster.n_output_nodes * config.n * config.n
        fc_keys = jax.random.split(k_fc, config.n_problems)
        self.fc_heads = [
            FCLayer(
                in_features=fc_in,
                out_features=config.problems[i].fc_out_features,
                activation=config.problems[i].fc_activation,
                key=fc_keys[i],
            )
            for i in range(config.n_problems)
        ]

        # ── Per-problem encoder masks (random subset selection) ───────────
        seed = int(jax.random.randint(k_mask, (), 0, 2**31 - 1))
        rng = np.random.default_rng(seed)
        masks = []
        for p in config.problems:
            chosen = rng.choice(
                config.n_encoders, size=p.n_encoders_used, replace=False,
            )
            mask = np.zeros(config.n_encoders, dtype=bool)
            mask[chosen] = True
            masks.append(mask)
        self.encoder_masks = masks

    # ── forward pass ──────────────────────────────────────────────────────

    def __call__(self, problem_idx: int, xs: jax.Array) -> jax.Array:
        """
        Run the network for a single problem.

        Parameters
        ----------
        problem_idx : int
            **Python int** selecting the problem head (0-based).
            Each distinct value traces a separate XLA program.
        xs : jax.Array, shape ``(n_encoders, 1, n, n)``
            Full encoder input tensor.  Slots not in this problem's
            encoder mask are deactivated via the ``is_active`` flags.

        Returns
        -------
        jax.Array, shape ``(fc_out_features,)``
            Task-specific output produced by the selected FC head.
        """
        # Encoder mask → JAX bool array (literal at trace time)
        flags = jnp.array(self.encoder_masks[problem_idx])

        # Shared encoder
        enc_out, enc_flags = self.encoder_layer(xs, flags)

        # Shared decoder cluster
        dec_out, _ = self.decoder_cluster(enc_out, enc_flags)

        # Flatten all decoder output nodes
        flat = dec_out.reshape(-1)

        # Problem-specific FC head
        return self.fc_heads[problem_idx](flat)

    # ── helpers ────────────────────────────────────────────────────────────

    @property
    def n_problems(self) -> int:
        """Number of problem heads."""
        return self.config.n_problems

    @property
    def n_decoder_output_nodes(self) -> int:
        """Total output nodes across all DecoderCluster layers."""
        return self.decoder_cluster.n_output_nodes

    @property
    def fc_in_features(self) -> int:
        """Dimensionality of the flattened vector entering each FC head."""
        return self.n_decoder_output_nodes * self.config.n * self.config.n

    def active_encoder_indices(self, problem_idx: int) -> np.ndarray:
        """Return the encoder slot indices that are active for a problem.

        Useful for logging / debugging which encoders a problem uses.

        Returns
        -------
        np.ndarray of int
            Sorted array of active encoder indices.
        """
        return np.where(self.encoder_masks[problem_idx])[0]

    def count_params(self) -> dict:
        """Count trainable parameters broken down by component.

        Returns
        -------
        dict with keys:
            'encoder_layer'    : int
            'decoder_cluster'  : int
            'fc_heads'         : list[int]  (one per problem)
            'total'            : int
        """
        def _sum_leaves(module):
            return sum(
                x.size for x in jax.tree_util.tree_leaves(
                    eqx.filter(module, eqx.is_inexact_array)
                )
            )

        enc = _sum_leaves(self.encoder_layer)
        dec = _sum_leaves(self.decoder_cluster)
        fc  = [_sum_leaves(h) for h in self.fc_heads]
        return {
            'encoder_layer': enc,
            'decoder_cluster': dec,
            'fc_heads': fc,
            'total': enc + dec + sum(fc),
        }
