# Configuration for Decoder Modules

DECODER_MAX_PARENTS = 16
DECODER_ACTIVATION_THRESHOLD = 12

# Ranks to extract initially based on proxy scores
DECODER_TOP_K_EXTRACT = 12

# Feature breakdown from Top-K
DECODER_INTERACT_RANKS = 8  # Number of top ranks to generate batched combinations for (e.g. 8C2 = 28)
DECODER_PRESERVE_RANKS = 4  # Number of ranks after the interaction ranks to keep raw

# Dimension reductions
DECODER_INTERMEDIATE_CHANNELS = 32  # 28 (combinations from top 8) + 4 (features ranked 9-12)
DECODER_HIDDEN_CHANNELS = 4
DECODER_OUT_CHANNELS = 1
