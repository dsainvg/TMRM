"""
otherutils.py — Training utilities for the TMRM 4×4 Sudoku task.

Contains everything that supports training (data loading, model and
optimiser construction, evaluation) but does not belong in the
core training loop (train.py).
"""

import urllib.request

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from model import Model
from utils.config.model import ModelConfig, ProblemConfig
from utils.config.data import DataConfig, DATA_CFG
from utils.config.training import TrainingConfig


# ── 1. Data download ──────────────────────────────────────────────────────────

def download_dataset(cfg: DataConfig = DATA_CFG) -> None:
    """Download the dataset .npz if not already present on disk."""
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    if cfg.dataset_path.exists():
        print(f"[data] Dataset already present at {cfg.dataset_path}")
        return
    print(f"[data] Downloading dataset from {cfg.dataset_url} …")
    urllib.request.urlretrieve(cfg.dataset_url, cfg.dataset_path)
    print(
        f"[data] Saved → {cfg.dataset_path} "
        f"({cfg.dataset_path.stat().st_size / 1024:.1f} KB)"
    )


# ── 2. Preprocessing ──────────────────────────────────────────────────────────

def one_hot_puzzle(
    puzzles: np.ndarray,
    cfg: DataConfig = DATA_CFG,
) -> np.ndarray:
    """
    Encode raw puzzle values as one-hot channels.

    Parameters
    ----------
    puzzles : (N_samples, n, n)  values in {0, …, n_channels_in-1}
    cfg     : DataConfig

    Returns
    -------
    (N_samples, n_channels_in, n, n)  float32 binary channels
    """
    n_vals = cfg.n_channels_in
    oh = puzzles[:, None, :, :] == np.arange(n_vals)[:, None, None]  # bool
    return oh.astype(np.float32)


def one_hot_solution(
    solutions: np.ndarray,
    cfg: DataConfig = DATA_CFG,
) -> np.ndarray:
    """
    Encode raw solution values as one-hot channels.

    Parameters
    ----------
    solutions : (N_samples, n, n)  values in {1, …, n_channels_out}
    cfg       : DataConfig

    Returns
    -------
    (N_samples, n_channels_out, n, n)  float32 binary channels
    """
    digits = np.arange(1, cfg.n_channels_out + 1)
    oh = solutions[:, None, :, :] == digits[:, None, None]  # bool
    return oh.astype(np.float32)


def load_data(cfg: DataConfig = DATA_CFG):
    """
    Load and preprocess the Sudoku dataset, returning a train/val split.

    Returns
    -------
    X_train, Y_train, X_val, Y_val
        X shapes : (N_split, n_channels_in, n, n)
        Y shapes : (N_split, fc_out)  — flattened one-hot targets
    """
    raw = np.load(cfg.dataset_path)
    puzzles_raw   = raw["puzzles"]    # (N, n, n)
    solutions_raw = raw["solutions"]  # (N, n, n)

    print(f"[data] Loaded {len(puzzles_raw)} samples.")
    print(f"       Puzzle   shape : {puzzles_raw.shape}  dtype={puzzles_raw.dtype}")
    print(f"       Solution shape : {solutions_raw.shape}  dtype={solutions_raw.dtype}")

    X = one_hot_puzzle(puzzles_raw, cfg)      # (N, n_channels_in, n, n)
    Y = one_hot_solution(solutions_raw, cfg)  # (N, n_channels_out, n, n)

    # Flatten targets to FC output space: (N, fc_out)
    Y_flat = Y.reshape(len(Y), -1)

    # 80/20 train-val split
    split = int(0.8 * len(X))
    return (
        X[:split],      Y_flat[:split],
        X[split:],      Y_flat[split:],
    )


# ── 3. Model construction ─────────────────────────────────────────────────────

def build_model(key: jax.Array, cfg: DataConfig = DATA_CFG) -> Model:
    """Construct the TMRM model for the given task configuration."""
    model_cfg = ModelConfig(
        n=cfg.n,
        n_encoders=cfg.n_channels_in,
        n_decoder_layers=2,
        max_decoder_nodes=20,
        problems=(
            ProblemConfig(
                n_encoders_used=cfg.n_channels_in,
                fc_out_features=cfg.fc_out,
                fc_activation="sigmoid",
            ),
        ),
    )
    return Model(model_cfg, key=key)


# ── 4. Optimiser construction ─────────────────────────────────────────────────

def _make_schedule(cfg: TrainingConfig):
    """Build an optax learning-rate schedule from a TrainingConfig."""
    if cfg.lr_schedule == "constant":
        return cfg.learning_rate
    if cfg.lr_schedule == "cosine":
        return optax.cosine_decay_schedule(cfg.learning_rate, decay_steps=1000)
    if cfg.lr_schedule == "linear":
        return optax.linear_schedule(cfg.learning_rate, 0.0, 1000)
    if cfg.lr_schedule == "warmup_cosine":
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=cfg.learning_rate,
            warmup_steps=cfg.warmup_steps,
            decay_steps=1000,
        )
    return cfg.learning_rate


def build_optimiser(train_cfg: TrainingConfig, model: Model):
    """Build an optax optimiser and its initial state.

    Returns
    -------
    tx        : optax.GradientTransformation
    opt_state : optax.OptState
    """
    schedule = _make_schedule(train_cfg)

    if train_cfg.optimiser == "adam":
        tx = optax.adam(schedule)
    elif train_cfg.optimiser == "adamw":
        tx = optax.adamw(schedule, weight_decay=train_cfg.weight_decay)
    elif train_cfg.optimiser == "sgd":
        tx = optax.sgd(schedule, momentum=0.9)
    elif train_cfg.optimiser == "rmsprop":
        tx = optax.rmsprop(schedule)
    else:
        raise ValueError(f"Unknown optimiser '{train_cfg.optimiser}'")

    if train_cfg.grad_clip_norm > 0.0:
        tx = optax.chain(optax.clip_by_global_norm(train_cfg.grad_clip_norm), tx)

    opt_state = tx.init(eqx.filter(model, eqx.is_inexact_array))
    return tx, opt_state


# ── 5. Evaluation ─────────────────────────────────────────────────────────────

def evaluate(
    model: Model,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    cfg: DataConfig = DATA_CFG,
) -> dict:
    """
    Compute validation loss and cell-level accuracy.

    A cell is correct when the predicted digit (argmax over output channels)
    matches the true digit.

    Returns
    -------
    dict with keys ``'loss'`` (float) and ``'cell_acc'`` (float in [0, 1]).
    """
    total_loss    = 0.0
    correct_cells = 0
    total_cells   = 0

    eps = 1e-7

    for i in range(len(X_val)):
        xs     = jnp.array(X_val[i][:, None, :, :])  # (n_channels_in, 1, n, n)
        y      = jnp.array(Y_val[i])                  # (fc_out,)
        logits = model(0, xs)                          # (fc_out,)

        p   = jnp.clip(logits, eps, 1.0 - eps)
        bce = -(y * jnp.log(p) + (1.0 - y) * jnp.log(1.0 - p))
        total_loss += float(jnp.mean(bce))

        pred_ch = logits.reshape(cfg.n_channels_out, cfg.n, cfg.n)
        true_ch = y.reshape(cfg.n_channels_out, cfg.n, cfg.n)
        pred_dig = jnp.argmax(pred_ch, axis=0)
        true_dig = jnp.argmax(true_ch, axis=0)
        correct_cells += int(jnp.sum(pred_dig == true_dig))
        total_cells   += cfg.n * cfg.n

    return {
        "loss":     total_loss / len(X_val),
        "cell_acc": correct_cells / total_cells,
    }
