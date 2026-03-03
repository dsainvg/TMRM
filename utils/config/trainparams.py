"""
trainparams.py — Single source of truth for ALL run parameters.

Edit the constants at the top to change anything about the training run
or model architecture.  The config objects below are built from them
automatically.
"""

# ── Training hyperparameters ─────────────────────────────────────────────────
BATCH_SIZE     = 1024
N_EPOCHS       = 300
LEARNING_RATE  = 3e-3
OPTIMISER      = "adam"
LR_SCHEDULE    = "constant"
GRAD_CLIP_NORM = 0.99
LOG_EVERY      = 10
SEED           = 42

# ── Task / data dimensionality ───────────────────────────────────────────────
N              = 9    # grid size (N×N cells)
N_CHANNELS_IN  = 10   # digits 0-9 → 10 one-hot input channels
N_CHANNELS_OUT = 9    # digits 1-9 → 9 one-hot output channels

# ── Paths & dataset ─────────────────────────────────────────────────────────
DATASET_URL      = "https://raw.githubusercontent.com/dsainvg/SUDOKU/main/outputs/dataset_9x9.npz"
DATA_DIR         = "data"
DATASET_FILENAME = "dataset_9x9.npz"
CHECKPOINT_DIR   = "checkpoints"

# ── Model architecture ───────────────────────────────────────────────────────
N_DECODER_LAYERS  = 8
MAX_DECODER_NODES = 150
FC_ACTIVATION     = "sigmoid"   # output activation for the FC head

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
DECODER_ACTIVATION_THRESHOLD  = 12  # min parents active to fire (≥12/16)
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

# ── FC head ───────────────────────────────────────────────────────────────────
FC_DEFAULT_ACTIVATION = 'relu'

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

DATA_CFG = DataConfig(
    dataset_url=DATASET_URL,
    data_dir=DATA_DIR,
    dataset_filename=DATASET_FILENAME,
    checkpoint_dir=CHECKPOINT_DIR,
    n=N,
    n_channels_in=N_CHANNELS_IN,
    n_channels_out=N_CHANNELS_OUT,
)

MODEL_CFG = ModelConfig(
    n=N,
    n_encoders=N_CHANNELS_IN,
    n_decoder_layers=N_DECODER_LAYERS,
    max_decoder_nodes=MAX_DECODER_NODES,
    problems=(
        ProblemConfig(
            n_encoders_used=N_CHANNELS_IN,
            fc_out_features=N_CHANNELS_OUT * N * N,
            fc_activation=FC_ACTIVATION,
        ),
    ),
)
