"""
train.py — Core training loop for the TMRM 4×4 Sudoku model.

Responsibilities
----------------
* Define the per-sample loss function (_single_loss).
* Define the JIT-compiled gradient update step (train_step).
* Orchestrate the epoch loop (train).

Everything else (data loading, model/optimiser construction, evaluation,
path constants) lives in utils/otherutils.py and utils/config/data.py.
"""

import time

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from model import Model
from utils.config.data import DATA_CFG
from utils.config.training import TrainingConfig
from utils.otherutils import (
    download_dataset,
    load_data,
    build_model,
    build_optimiser,
    evaluate,
)


# ── Loss ──────────────────────────────────────────────────────────────────────

def _single_loss(model: Model, xs: jax.Array, y: jax.Array) -> jax.Array:
    """
    Binary cross-entropy loss for a single sample (used inside vmap).

    xs : (n_channels_in, 1, n, n)  one-hot channels for a single puzzle
    y  : (fc_out,)                 flattened one-hot solution (float32)
    """
    logits = model(0, xs)  # (64,) — sigmoid applied inside FC head
    eps = 1e-7
    p   = jnp.clip(logits, eps, 1.0 - eps)
    bce = -(y * jnp.log(p) + (1.0 - y) * jnp.log(1.0 - p))
    return jnp.mean(bce)


@eqx.filter_jit
def train_step(
    model:     Model,
    opt_state,
    tx,
    xs_batch:  jax.Array,   # (B, n_channels_in, 1, n, n)
    y_batch:   jax.Array,   # (B, fc_out)
):
    """
    Batched gradient update via jax.vmap over B samples.

    All B samples are evaluated in parallel as a single SIMD XLA kernel.
    Enabled by Encoder.__call__ using jnp.where (no nested-vmap conflict).
    """
    def batch_loss(model):
        per_sample = jax.vmap(
            lambda xs, y: _single_loss(model, xs, y)
        )(xs_batch, y_batch)   # (B,)
        return jnp.mean(per_sample)

    loss, grads = eqx.filter_value_and_grad(batch_loss)(model)
    updates, opt_state_new = tx.update(
        grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
    )
    model_new = eqx.apply_updates(model, updates)
    return model_new, opt_state_new, loss


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    train_cfg: TrainingConfig = TrainingConfig(
        batch_size=32,
        n_epochs=30,
        learning_rate=3e-4,
        optimiser="adam",
        lr_schedule="constant",
        grad_clip_norm=1.0,
        log_every=50,
        seed=42,
    ),
    data_cfg=DATA_CFG,
):
    print("=" * 60)
    print("TMRM — 4×4 Sudoku Training")
    print("=" * 60)

    # ── Data ──────────────────────────────────────────────────────────
    download_dataset(data_cfg)
    X_train, Y_train, X_val, Y_val = load_data(data_cfg)
    n_train = len(X_train)
    print(f"[data] Train: {n_train}  |  Val: {len(X_val)}")

    # ── Model ─────────────────────────────────────────────────────────
    key   = jax.random.key(train_cfg.seed)
    model = build_model(key, data_cfg)
    params = model.count_params()
    print(f"[model] Total params: {params['total']:,}")

    # ── Optimiser ─────────────────────────────────────────────────────
    tx, opt_state = build_optimiser(train_cfg, model)

    # ── Checkpoint dir ────────────────────────────────────────────────
    data_cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Training ──────────────────────────────────────────────────────
    rng = np.random.default_rng(train_cfg.seed)
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(1, train_cfg.n_epochs + 1):
        epoch_start = time.time()
        perm = rng.permutation(n_train)

        epoch_loss = 0.0
        n_batches  = 0
        for b_start in range(0, n_train, train_cfg.batch_size):
            idx = perm[b_start : b_start + train_cfg.batch_size]

            xs_batch = jnp.array(X_train[idx][:, :, None, :, :])   # (B, C, 1, n, n)
            y_batch  = jnp.array(Y_train[idx])                      # (B, fc_out)

            model, opt_state, loss = train_step(model, opt_state, tx, xs_batch, y_batch)
            epoch_loss += float(loss)
            n_batches  += 1
            global_step += 1

            if global_step % train_cfg.log_every == 0:
                print(
                    f"  step {global_step:5d} | "
                    f"train_loss={loss:.4f}"
                )

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed  = time.time() - epoch_start

        val_metrics = evaluate(model, X_val, Y_val, data_cfg)
        val_loss    = val_metrics["loss"]
        cell_acc    = val_metrics["cell_acc"]

        print(
            f"[epoch {epoch:3d}/{train_cfg.n_epochs}] "
            f"train_loss={avg_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"cell_acc={cell_acc:.2%}  "
            f"({elapsed:.1f}s)"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path     = data_cfg.checkpoint_dir / "best_model.eqx"
            eqx.tree_serialise_leaves(str(ckpt_path), model)
            print(f"  ✓ Checkpoint saved → {ckpt_path}  (val_loss={val_loss:.4f})")

    print("=" * 60)
    print(f"Training complete.  Best val_loss: {best_val_loss:.4f}")
    print(f"Best model saved at: {data_cfg.checkpoint_dir / 'best_model.eqx'}")
    return model


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trained_model = train()
