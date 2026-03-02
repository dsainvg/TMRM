# Configuration for EncoderLayer

ENCODER_STACK_DEPTH = 2

# Channels produced by the stage-1 encoder (= number of stage-2 leaf nodes).
ENCODER_INTER_STACK_CHANNELS = 8

# Number of stage-2 leaf encoder nodes per input stack.
# Each leaf receives one of the 8 channels from stage-1.
ENCODER_STAGE2_COUNT = 8

# Output channels per input stack: ENCODER_STAGE2_COUNT × ENCODER_OUT_CHANNELS.
# With N inputs the full layer output is (N, 64, n, n).
ENCODER_STACK_OUT_CHANNELS = 64
