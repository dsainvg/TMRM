# Configuration for EncoderLayer

ENCODER_STACK_DEPTH = 2

# Channels flowing between stage-1 and stage-2 encoders within one stack.
# Equals ENCODER_OUT_CHANNELS of the stage-1 Encoder.
ENCODER_INTER_STACK_CHANNELS = 8

# Output channels per input stack (stage-2 Encoder output).
# With N inputs the full layer output is (N, 64, n, n).
ENCODER_STACK_OUT_CHANNELS = 64
