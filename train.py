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

import sys
import time
from pathlib import Path

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


# ── Log tee (stdout -> terminal + file) ───────────────────────────────────────

class _Tee:
    """Duplicate every write to both the original stream and a log file."""
    def __init__(self, stream, filepath: Path):
        self._stream = stream
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(filepath, "a", encoding="utf-8", buffering=1)
        self._file.write(f"\n{'='*60}\n  Run started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*60}\n")

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def close(self):
        self._file.close()

    # pass through everything else (isatty, fileno, etc.)
    def __getattr__(self, name):
        return getattr(self._stream, name)


# ── Gradient statistics ──────────────────────────────────────────────────────

def _grad_norms(grads: Model) -> dict:
    """Compute L2-norm and max-abs gradient per named component (runs inside JIT)."""
    def _stats(module):
        leaves = jax.tree_util.tree_leaves(eqx.filter(module, eqx.is_inexact_array))
        if not leaves:
            return jnp.zeros(()), jnp.zeros(())
        flat = jnp.concatenate([x.ravel() for x in leaves])
        return jnp.linalg.norm(flat), jnp.max(jnp.abs(flat))

    enc_norm, enc_max = _stats(grads.encoder_layer)
    dec_norm, dec_max = _stats(grads.decoder_cluster)
    pa_stats = [_stats(h) for h in grads.port_adapters]

    all_leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
    all_flat   = jnp.concatenate([x.ravel() for x in all_leaves])
    total_norm = jnp.linalg.norm(all_flat)
    total_max  = jnp.max(jnp.abs(all_flat))

    stats = {
        "total_norm": total_norm, "total_max": total_max,
        "enc_norm":   enc_norm,   "enc_max":   enc_max,
        "dec_norm":   dec_norm,   "dec_max":   dec_max,
    }
    for i, (n_, mx) in enumerate(pa_stats):
        stats[f"pa{i}_norm"] = n_
        stats[f"pa{i}_max"]  = mx
    return stats


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
    Batched gradient update via jax.vmap over B samples — fast path.

    Does NOT compute gradient statistics.  Use ``train_step_debug`` when
    gradient norms are needed (debug=True).
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


@eqx.filter_jit
def train_step_debug(
    model:       Model,
    opt_state,
    tx,
    xs_batch:    jax.Array,   # (B, n_encoders, 1, n, n)
    y_batch:     jax.Array,   # (B, fc_out)
    problem_idx: int,         # Python int — traced as a compile-time constant
):
    """
    Batched gradient update with gradient norm statistics.

    Only called when ``debug=True`` — the extra norm computations and
    concatenations over all parameter leaves add measurable overhead.
    """
    def batch_loss(model):
        per_sample = jax.vmap(
            lambda xs, y: _single_loss(model, xs, y, problem_idx)
        )(xs_batch, y_batch)   # (B,)
        return jnp.mean(per_sample)

    loss, grads = eqx.filter_value_and_grad(batch_loss)(model)
    grad_stats = _grad_norms(grads)
    updates, opt_state_new = tx.update(
        grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
    )
    model_new = eqx.apply_updates(model, updates)
    return model_new, opt_state_new, loss, grad_stats


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

# ── Node-activity debug (first step only) ───────────────────────────────────────

def _print_node_activity(model: Model, xs: jax.Array, problem_idx: int) -> None:
    """
    Run a single non-JIT forward pass and print the active/inactive status of
    every decoder node in every layer.  Called only on global_step == 1.
    """
    flags = jnp.array(model.encoder_masks[problem_idx])
    enc_out, enc_flags = model.encoder_layer(xs[0], flags)   # single sample [0]

    _, _, layer_flags = model.decoder_cluster.forward_debug(enc_out, enc_flags)

    n_layers   = len(layer_flags)
    total_nodes = sum(len(f) for f in layer_flags)
    total_active = sum(int(f.sum()) for f in layer_flags)

    print(f"  [DEBUG] Decoder node activity  (problem_idx={problem_idx})")
    print(f"  {'Layer':<8}  {'Nodes':>6}  {'Active':>7}  {'Inactive':>9}  Activation map")
    print(f"  {'-'*8}  {'-'*6}  {'-'*7}  {'-'*9}  {'-'*40}")
    for li, f in enumerate(layer_flags):
        n_nodes    = len(f)
        n_active   = int(f.sum())
        n_inactive = n_nodes - n_active
        bar        = "".join("1" if b else "0" for b in f.tolist())
        # break bar into groups of 16 for readability
        groups = " ".join(bar[i:i+16] for i in range(0, len(bar), 16))
        n_out_nodes = model.decoder_cluster.output_node_indices[li].size
        tag = f" [out:{n_out_nodes}]" if n_out_nodes > 0 else ""
        print(f"  L{li:<7}  {n_nodes:>6}  {n_active:>7}  {n_inactive:>9}  {groups}{tag}")
    print(f"  {'TOTAL':<8}  {total_nodes:>6}  {total_active:>7}  {total_nodes-total_active:>9}")
    print()


# ── Gradient printing ─────────────────────────────────────────────────────────────

_prev_grad_stats: dict | None = None   # module-level: tracks last printed stats


def _print_grad_stats(step: int, loss0, loss1, gs: dict) -> None:
    """Print a compact gradient-norm table with delta from previous log."""
    global _prev_grad_stats

    def _delta(key: str) -> str:
        if _prev_grad_stats is None:
            return "      n/a"
        d = gs[key] - _prev_grad_stats[key]
        sign = "+" if d >= 0 else "-"
        return f"{sign}{abs(d):8.4f}"

    n_pa = sum(1 for k in gs if k.endswith("_norm") and k.startswith("pa"))
    print(f"  step {step:5d}  |  loss0={float(loss0):.4f}  loss1={float(loss1):.4f}")
    print(f"  {'component':<16}  {'L2 norm':>10}  {'max |g|':>10}  {'delta norm':>10}")
    print(f"  {'-'*16}  {'-'*10}  {'-'*10}  {'-'*10}")
    rows = [
        ("total",   "total_norm", "total_max"),
        ("encoder",  "enc_norm",   "enc_max"),
        ("decoder",  "dec_norm",   "dec_max"),
    ] + [(f"port_adapter_{i}", f"pa{i}_norm", f"pa{i}_max") for i in range(n_pa)]
    for label, nk, mk in rows:
        print(f"  {label:<16}  {gs[nk]:>10.5f}  {gs[mk]:>10.5f}  {_delta(nk):>10}")
    print()
    _prev_grad_stats = dict(gs)

# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    train_cfg: TrainingConfig = TRAIN_CFG,
    data_cfg=DATA_CFG,
    flow_cfg=FLOW_DATA_CFG,
    model_cfg=MODEL_CFG,
    debug: bool = False,
    log: bool = False,
):
    # ── Mirror all stdout to a timestamped log file ──────────────────────
    tee = None
    log_path = None
    if log:
        log_dir  = Path("logs")
        log_path = log_dir / f"train_{time.strftime('%Y%m%d_%H%M%S')}.log"
        tee = _Tee(sys.stdout, log_path)
        sys.stdout = tee
        print(f"[log] Writing to {log_path}")
    print()
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

    # Pre-build full training data on device — eliminates per-batch
    # numpy allocation + host→device transfer inside the inner loop.
    xs0_train_dev = jnp.array(build_xs_batch(X0_train, slots0, n_encoders, n))
    y0_train_dev  = jnp.array(Y0_train)
    xs1_train_dev = jnp.array(build_xs_batch(X1_train, slots1, n_encoders, n))
    y1_train_dev  = jnp.array(Y1_train)

    # Select train_step variant once (avoids Python branch per iteration)
    _step = train_step_debug if debug else train_step

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

        epoch_loss_0 = jnp.float32(0.0)
        epoch_loss_1 = jnp.float32(0.0)
        nb0 = 0
        nb1 = 0
        last_grad_stats: dict | None = None   # most recent grad stats this epoch

        for it in range(n_iters):
            # ── Task 0 (Sudoku) ───────────────────────────────────────
            b0   = (it % n_batches_0) * B
            idx0 = perm0_tiled[b0 : b0 + B]           # always exactly B indices
            xs0  = xs0_train_dev[idx0]                 # device-side gather
            y0   = y0_train_dev[idx0]                  # device-side gather
            if debug:
                model, opt_state, loss0, gs0 = _step(model, opt_state, tx, xs0, y0, 0)
            else:
                model, opt_state, loss0 = _step(model, opt_state, tx, xs0, y0, 0)
            epoch_loss_0 = epoch_loss_0 + loss0        # stays on device
            nb0 += 1

            # ── Task 1 (Flow Free) ────────────────────────────────────
            b1   = (it % n_batches_1) * B
            idx1 = perm1_tiled[b1 : b1 + B]           # always exactly B indices
            xs1  = xs1_train_dev[idx1]                 # device-side gather
            y1   = y1_train_dev[idx1]                  # device-side gather
            if debug:
                model, opt_state, loss1, gs1 = _step(model, opt_state, tx, xs1, y1, 1)
            else:
                model, opt_state, loss1 = _step(model, opt_state, tx, xs1, y1, 1)
            epoch_loss_1 = epoch_loss_1 + loss1        # stays on device
            nb1 += 1

            # merge grad stats across both tasks (only when debugging)
            if debug:
                merged_gs = {k: 0.5 * (float(gs0[k]) + float(gs1[k])) for k in gs0}
                last_grad_stats = merged_gs

            # ── First-step node-activity debug (only when debug=True) ────
            if debug and global_step == 1:
                print("\n" + "=" * 60)
                print("  NODE ACTIVITY DEBUG  (step 1, after first weight update)")
                print("=" * 60)
                _print_node_activity(model, xs0, problem_idx=0)
                _print_node_activity(model, xs1, problem_idx=1)
                print("=" * 60 + "\n")

            global_step += 1

        avg_loss_0 = float(epoch_loss_0) / max(nb0, 1)   # single device→host sync
        avg_loss_1 = float(epoch_loss_1) / max(nb1, 1)   # single device→host sync
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

        # ── Gradient stats (every epoch, debug only) ─────────────────────
        if debug and last_grad_stats is not None:
            _print_grad_stats(global_step, loss0, loss1, last_grad_stats)

        if avg_val_loss < best_val_loss - 0.001:
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
    if log and tee is not None:
        print(f"[log] Full log saved to {log_path}")
        sys.stdout = tee._stream
        tee.close()

    return model


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trained_model = train(debug=False)
