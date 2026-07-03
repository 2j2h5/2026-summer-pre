# src/semantic_alignment/config.py

# Device
DEVICE = "cuda"  # "cuda" or "cpu"; falls back to CPU if CUDA is unavailable.
OUT_DIR = "outputs"

# Models
MODEL_NAMES = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-base-v2",
    "thenlper/gte-base",
]

SEED = 42

# Primary decomposition: G ≈ M S
K = 207

# GTX1650-friendly default.
# Use 5000 for stronger final runs.
STEPS = 1500
LR = 2e-3
LOG_EVERY = 500

# Softmax temperatures for factor matrices
M_TEMPERATURE = 0.8
S_TEMPERATURE = 0.8

# MS loss weights
RECON_WEIGHT = 1.0
M_ENTROPY_WEIGHT = 5e-4
S_ENTROPY_WEIGHT = 5e-4
M_USAGE_WEIGHT = 2e-3
S_USAGE_WEIGHT = 2e-3
S_ORTH_WEIGHT = 1e-3
M_L2_WEIGHT = 1e-6
S_L2_WEIGHT = 1e-6

# Relation-temperature search
COARSE_T_CANDIDATES = [0.05, 0.07, 0.10, 0.15, 0.25]
COARSE_T_SELECTION_SEEDS = [1]

FINE_RADIUS = 0.04
FINE_NUM_POINTS = 5
FINE_T_SELECTION_SEEDS = [1]

FINAL_SEED_STABILITY_SEEDS = [1, 2]

# If True, tries combinations of top T candidates across models.
# This is expensive. Keep False on GTX1650.
COMMON_S_CROSS_SEARCH = False
TOP_T_PER_MODEL = 2

# T score
SCORE_SAMEG_WEIGHT = 1.0
SCORE_JS_WEIGHT = 1.0
SCORE_UNIQUE_WEIGHT = 0.001
SCORE_RANK_WEIGHT = 0.001
SCORE_RECON_WEIGHT = 10.0

# Common S
COMMON_S_REFERENCE_MODEL = None  # None = first model

# Fit M under fixed common S*
FIT_M_TO_COMMON_S_STEPS = 1000
FIT_M_TO_COMMON_S_LR = 2e-3
FIT_M_TO_COMMON_S_TEMPERATURE = 0.8
FIT_M_TO_COMMON_S_ENTROPY_WEIGHT = 5e-4
FIT_M_TO_COMMON_S_USAGE_WEIGHT = 2e-3
FIT_M_TO_COMMON_S_L2_WEIGHT = 1e-6
