"""
trainparams.py — Single source of truth for ALL run parameters.

Edit the constants at the top to change anything about the training run
or model architecture.  The config objects below are built from them
automatically.
"""

# ── Training hyperparameters ─────────────────────────────────────────────────
BATCH_SIZE     = 1024
N_EPOCHS       = 2500
LEARNING_RATE  = 5e-3
OPTIMISER      = "adam"
LR_SCHEDULE    = "constant"
GRAD_CLIP_NORM = 1.01
LOG_EVERY      = 100
SEED           = 41

# ── Task / data dimensionality ───────────────────────────────────────────────
N              = 4    # grid size (N×N cells)
N_CHANNELS_IN  = 5   # per-task one-hot input channels (0..4 for both tasks)
N_CHANNELS_OUT = 4   # per-task one-hot output channels (1..4 for both tasks)

# ── Encoder pool (shared backbone) ──────────────────────────────────────────
# 12 total slots shared across all tasks; each task randomly occupies 5.
# The remaining 7 slots stay inactive (zero-filled input + is_active=False).
N_ENCODERS = 8

# ── Paths & datasets ────────────────────────────────────────────────────────
# Task 0 — 4×4 Sudoku
DATASET_URL      = "https://raw.githubusercontent.com/dsainvg/SUDOKU/main/outputs/dataset_4x4.npz"
DATA_DIR         = "data"
DATASET_FILENAME = "dataset_4x4.npz"
CHECKPOINT_DIR   = "checkpoints"

# Task 1 — 4×4 Flow Free
FLOW_DATASET_URL      = "https://github.com/dsainvg/FLOW/raw/refs/heads/main/outputs/flow_4x4.npz"
FLOW_DATASET_FILENAME = "flow_4x4.npz"

# ── Model architecture ───────────────────────────────────────────────────────
N_DECODER_LAYERS  = 5
MAX_DECODER_NODES = 170
PA_ACTIVATION     = "sigmoid"   # output activation for the Port Adapter

# ── Encoder node ─────────────────────────────────────────────────────────────
ENCODER_IN_CHANNELS           = 1
ENCODER_EXPAND_CHANNELS       = 4
ENCODER_INTERMEDIATE_CHANNELS = 10
ENCODER_OUT_CHANNELS          = 8

# ── EncoderLayer ─────────────────────────────────────────────────────────────
ENCODER_STACK_DEPTH           = 2
ENCODER_INTER_STACK_CHANNELS  = 8   # channels out of stage-1 = stage-2 leaf count
ENCODER_STAGE2_COUNT          = 8   # stage-2 leaf nodes per input stack
ENCODER_STACK_OUT_CHANNELS    = 64  # ENCODER_STAGE2_COUNT × ENCODER_OUT_CHANNELS

# ── Decoder node ─────────────────────────────────────────────────────────────
DECODER_MAX_PARENTS           = 16
DECODER_ACTIVATION_THRESHOLD  = 9  # min parents active to fire (≥9/16)
DECODER_TOP_K_EXTRACT         = 12  # initial proxy-score candidates
DECODER_INTERACT_RANKS        = 8   # top-k used for pairwise combos (8C2=28)
DECODER_PRESERVE_RANKS        = 4   # ranks 9-12 kept raw
DECODER_INTERMEDIATE_CHANNELS = 32  # 28 combos + 4 preserved
DECODER_HIDDEN_CHANNELS       = 4
DECODER_OUT_CHANNELS          = 1

# ── DecoderCluster fan-out (Gaussian) ────────────────────────────────────────
FANOUT_FIRST_MU     = 18    # mean downstream connections (first half of network)
FANOUT_FIRST_SIGMA  = 3.0
FANOUT_FIRST_LO     = 8
FANOUT_FIRST_HI     = 24
FANOUT_SECOND_MU    = 12   # mean downstream connections (second half)
FANOUT_SECOND_SIGMA = 3.0
FANOUT_SECOND_LO    = 6
FANOUT_SECOND_HI    = 20

# ── Port Adapter head ────────────────────────────────────────────────────────
PA_DEFAULT_ACTIVATION = 'sigmoid'
FC_DEFAULT_ACTIVATION = 'relu'    # kept for fc_layer.py (backward compat)

# ── Config objects ───────────────────────────────────────────────────────────
from .training import TrainingConfig
from .data import DataConfig
from .model import ModelConfig, ProblemConfig

TRAIN_CFG = TrainingConfig(
    batch_size=BATCH_SIZE,
    n_epochs=N_EPOCHS,
    learning_rate=LEARNING_RATE,
    optimiser=OPTIMISER,
    lr_schedule=LR_SCHEDULE,
    grad_clip_norm=GRAD_CLIP_NORM,
    log_every=LOG_EVERY,
    seed=SEED,
    weight_decay=0.0,
    warmup_steps=0,
)

# Task 0: Sudoku
DATA_CFG = DataConfig(
    dataset_url=DATASET_URL,
    data_dir=DATA_DIR,
    dataset_filename=DATASET_FILENAME,
    checkpoint_dir=CHECKPOINT_DIR,
    n=N,
    n_channels_in=N_CHANNELS_IN,
    n_channels_out=N_CHANNELS_OUT,
)

# Task 1: Flow Free
FLOW_DATA_CFG = DataConfig(
    dataset_url=FLOW_DATASET_URL,
    data_dir=DATA_DIR,
    dataset_filename=FLOW_DATASET_FILENAME,
    checkpoint_dir=CHECKPOINT_DIR,
    n=N,
    n_channels_in=N_CHANNELS_IN,
    n_channels_out=N_CHANNELS_OUT,
)

MODEL_CFG = ModelConfig(
    n=N,
    n_encoders=N_ENCODERS,           # 12 total slots in the shared backbone
    n_decoder_layers=N_DECODER_LAYERS,
    max_decoder_nodes=MAX_DECODER_NODES,
    problems=(
        ProblemConfig(               # Task 0 — Sudoku
            n_encoders_used=N_CHANNELS_IN,   # 5 of 8 slots, randomly assigned
            pa_out_channels=N_CHANNELS_OUT,  # 4 channels; PA outputs (4, n, n)
            pa_activation=PA_ACTIVATION,
        ),
        ProblemConfig(               # Task 1 — Flow Free
            n_encoders_used=N_CHANNELS_IN,   # 5 of 8 slots, randomly assigned
            pa_out_channels=N_CHANNELS_OUT,  # 4 channels; PA outputs (4, n, n)
            pa_activation=PA_ACTIVATION,
        ),
    ),
)
