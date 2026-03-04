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
from utils.config.training import TrainingConfig
from utils.config.trainparams import TRAIN_CFG, DATA_CFG, FLOW_DATA_CFG, MODEL_CFG
from utils.otherutils import (
    download_dataset,
    load_data,
    build_model,
    build_optimiser,
    build_xs_batch,
)


# ── Loss ──────────────────────────────────────────────────────────────────────

def _single_loss(model: Model, xs: jax.Array, y: jax.Array, problem_idx: int) -> jax.Array:
    """
    Binary cross-entropy loss for a single sample (used inside vmap).

    xs          : (n_encoders, 1, n, n)  full encoder slot tensor (zeros in inactive slots)
    y           : (fc_out,)              flattened one-hot solution (float32)
    problem_idx : int                    which task head to use
    """
    logits = model(problem_idx, xs)  # (fc_out,) — sigmoid applied inside FC head
    eps = 1e-7
    p   = jnp.clip(logits, eps, 1.0 - eps)
    bce = -(y * jnp.log(p) + (1.0 - y) * jnp.log(1.0 - p))
    return jnp.mean(bce)


@eqx.filter_jit
def eval_step(
    model:       Model,
    xs_batch:    jax.Array,   # (B, n_encoders, 1, n, n)
    problem_idx: int,
) -> jax.Array:
    """
    JIT-compiled batched forward pass for evaluation.

    Returns logits (B, fc_out).  Compiled once per unique
    (batch_shape, problem_idx) pair — reused every epoch.
    """
    return jax.vmap(lambda xs: model(problem_idx, xs))(xs_batch)


@eqx.filter_jit
def train_step(
    model:       Model,
    opt_state,
    tx,
    xs_batch:    jax.Array,   # (B, n_encoders, 1, n, n)
    y_batch:     jax.Array,   # (B, fc_out)
    problem_idx: int,         # Python int — traced as a compile-time constant
):
    """
    Batched gradient update via jax.vmap over B samples.

    ``problem_idx`` selects which encoder mask and FC head to use.  It is a
    plain Python int so each distinct value produces its own XLA program
    (same as the model forward pass itself).
    """
    def batch_loss(model):
        per_sample = jax.vmap(
            lambda xs, y: _single_loss(model, xs, y, problem_idx)
        )(xs_batch, y_batch)   # (B,)
        return jnp.mean(per_sample)

    loss, grads = eqx.filter_value_and_grad(batch_loss)(model)
    updates, opt_state_new = tx.update(
        grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
    )
    model_new = eqx.apply_updates(model, updates)
    return model_new, opt_state_new, loss


# ── Batched evaluation (no Python loop, single JIT call per task) ─────────────

def _evaluate_batched(
    model,
    X_val:       np.ndarray,    # (N_val, n_channels_in, n, n)
    Y_val:       np.ndarray,    # (N_val, fc_out)
    slot_indices: np.ndarray,
    n_encoders:  int,
    n:           int,
    n_channels_out: int,
    problem_idx: int,
    xs_val:      jax.Array = None,  # optional pre-built device array
    y_val_jax:   jax.Array = None,
) -> dict:
    """
    Evaluate in one batched JIT call (no Python loop over samples).
    Pass pre-built ``xs_val`` / ``y_val_jax`` to avoid repeated host→device
    transfers across epochs.
    """
    if xs_val is None:
        xs_val   = jnp.array(build_xs_batch(X_val, slot_indices, n_encoders, n))
    if y_val_jax is None:
        y_val_jax = jnp.array(Y_val)

    logits = eval_step(model, xs_val, problem_idx)    # (N_val, fc_out)

    eps  = 1e-7
    p    = jnp.clip(logits, eps, 1.0 - eps)
    bce  = -(y_val_jax * jnp.log(p) + (1.0 - y_val_jax) * jnp.log(1.0 - p))
    loss = float(jnp.mean(bce))

    pred_ch  = logits.reshape(-1, n_channels_out, n, n)
    true_ch  = y_val_jax.reshape(-1, n_channels_out, n, n)
    pred_dig = jnp.argmax(pred_ch, axis=1)             # (N_val, n, n)
    true_dig = jnp.argmax(true_ch, axis=1)             # (N_val, n, n)
    cell_acc = float(jnp.mean(pred_dig == true_dig))

    return {"loss": loss, "cell_acc": cell_acc}


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    train_cfg: TrainingConfig = TRAIN_CFG,
    data_cfg=DATA_CFG,
    flow_cfg=FLOW_DATA_CFG,
    model_cfg=MODEL_CFG,
):
    print("=" * 60)
    print("TMRM — Multi-Task Training  (Sudoku | Flow Free)")
    print("=" * 60)

    # ── Data (both tasks) ──────────────────────────────────────────────
    download_dataset(data_cfg)   # Task 0 — Sudoku
    download_dataset(flow_cfg)   # Task 1 — Flow Free

    print("[data] Task 0 — Sudoku:")
    X0_train, Y0_train, X0_val, Y0_val = load_data(data_cfg)
    print(f"       Train: {len(X0_train)}  |  Val: {len(X0_val)}")

    print("[data] Task 1 — Flow Free:")
    X1_train, Y1_train, X1_val, Y1_val = load_data(flow_cfg)
    print(f"       Train: {len(X1_train)}  |  Val: {len(X1_val)}")

    n0_train = len(X0_train)
    n1_train = len(X1_train)

    # ── Model ─────────────────────────────────────────────────────────
    key   = jax.random.key(train_cfg.seed)
    model = build_model(key, model_cfg)
    params = model.count_params()
    print(f"[model] n_encoders={model_cfg.n_encoders}  |  Total params: {params['total']:,}")

    # Encoder slot indices are fixed at model init — retrieve once here.
    slots0 = model.active_encoder_indices(0)   # (5,) slots used by Sudoku
    slots1 = model.active_encoder_indices(1)   # (5,) slots used by Flow Free
    print(f"[model] Sudoku    encoder slots : {sorted(slots0.tolist())}")
    print(f"[model] Flow Free encoder slots : {sorted(slots1.tolist())}")

    n_encoders = model_cfg.n_encoders
    n          = model_cfg.n

    # ── Optimiser ─────────────────────────────────────────────────────
    tx, opt_state = build_optimiser(train_cfg, model)

    # ── Checkpoint dir ────────────────────────────────────────────────
    data_cfg.checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    # ── Training ──────────────────────────────────────────────────────
    B   = train_cfg.batch_size
    rng = np.random.default_rng(train_cfg.seed)
    global_step   = 0
    best_val_loss = float("inf")

    # Pre-build full-size val tensors once (no scatter overhead per epoch)
    xs0_val = jnp.array(build_xs_batch(X0_val, slots0, n_encoders, n))  # (N0v, enc, 1, n, n)
    xs1_val = jnp.array(build_xs_batch(X1_val, slots1, n_encoders, n))  # (N1v, enc, 1, n, n)
    y0_val  = jnp.array(Y0_val)   # (N0v, fc_out)
    y1_val  = jnp.array(Y1_val)   # (N1v, fc_out)

    for epoch in range(1, train_cfg.n_epochs + 1):
        epoch_start = time.time()
        perm0 = rng.permutation(n0_train)
        perm1 = rng.permutation(n1_train)

        # Tile permutations so we always draw exactly B samples per batch
        # (avoids variable-shape XLA recompilation when dataset < B)
        perm0_tiled = np.resize(perm0, (len(perm0) + B - 1) // B * B)
        perm1_tiled = np.resize(perm1, (len(perm1) + B - 1) // B * B)

        n_batches_0 = max(1, len(perm0_tiled) // B)
        n_batches_1 = max(1, len(perm1_tiled) // B)
        n_iters     = max(n_batches_0, n_batches_1)

        epoch_loss_0 = 0.0
        epoch_loss_1 = 0.0
        nb0 = 0
        nb1 = 0

        for it in range(n_iters):
            # ── Task 0 (Sudoku) ───────────────────────────────────────
            b0   = (it % n_batches_0) * B
            idx0 = perm0_tiled[b0 : b0 + B]           # always exactly B indices
            xs0  = jnp.array(
                build_xs_batch(X0_train[idx0], slots0, n_encoders, n)
            )                                          # (B, n_encoders, 1, n, n)
            y0   = jnp.array(Y0_train[idx0])           # (B, fc_out)
            model, opt_state, loss0 = train_step(model, opt_state, tx, xs0, y0, 0)
            epoch_loss_0 += float(loss0)
            nb0 += 1

            # ── Task 1 (Flow Free) ────────────────────────────────────
            b1   = (it % n_batches_1) * B
            idx1 = perm1_tiled[b1 : b1 + B]           # always exactly B indices
            xs1  = jnp.array(
                build_xs_batch(X1_train[idx1], slots1, n_encoders, n)
            )                                          # (B, n_encoders, 1, n, n)
            y1   = jnp.array(Y1_train[idx1])           # (B, fc_out)
            model, opt_state, loss1 = train_step(model, opt_state, tx, xs1, y1, 1)
            epoch_loss_1 += float(loss1)
            nb1 += 1

            global_step += 1
            if global_step % train_cfg.log_every == 0:
                print(
                    f"  step {global_step:5d} | "
                    f"task0_loss={loss0:.4f}  task1_loss={loss1:.4f}"
                )

        avg_loss_0 = epoch_loss_0 / max(nb0, 1)
        avg_loss_1 = epoch_loss_1 / max(nb1, 1)
        elapsed    = time.time() - epoch_start

        # ── Validation (both tasks) — single batched JIT call each ──────
        val0 = _evaluate_batched(model, X0_val, Y0_val, slots0, n_encoders, n,
                                 data_cfg.n_channels_out, problem_idx=0,
                                 xs_val=xs0_val, y_val_jax=y0_val)
        val1 = _evaluate_batched(model, X1_val, Y1_val, slots1, n_encoders, n,
                                 flow_cfg.n_channels_out, problem_idx=1,
                                 xs_val=xs1_val, y_val_jax=y1_val)
        avg_val_loss = 0.5 * (val0["loss"] + val1["loss"])

        print(
            f"[epoch {epoch:3d}/{train_cfg.n_epochs}] "
            f"sudoku  train={avg_loss_0:.4f}  val={val0['loss']:.4f}  acc={val0['cell_acc']:.2%}  |  "
            f"flow    train={avg_loss_1:.4f}  val={val1['loss']:.4f}  acc={val1['cell_acc']:.2%}  "
            f"({elapsed:.1f}s)"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path     = data_cfg.checkpoint_dir_path / "best_model.eqx"
            eqx.tree_serialise_leaves(str(ckpt_path), model)
            print(
                f"  ✓ Checkpoint saved → {ckpt_path}  "
                f"(avg_val_loss={avg_val_loss:.4f})"
            )

    print("=" * 60)
    print(f"Training complete.  Best avg_val_loss: {best_val_loss:.4f}")
    print(f"Best model saved at: {data_cfg.checkpoint_dir_path / 'best_model.eqx'}")
    return model


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trained_model = train()
