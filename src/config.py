"""
Configuration settings for the Multi-Genre Music Generation project.
"""
import os
import torch

# ─── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_MIDI_DIR = os.path.join(DATA_DIR, "raw_midi")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
SPLIT_DIR = os.path.join(DATA_DIR, "train_test_split")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
GENERATED_MIDI_DIR = os.path.join(OUTPUT_DIR, "generated_midis")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
SURVEY_DIR = os.path.join(OUTPUT_DIR, "survey_results")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

# ─── Device ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── MIDI Preprocessing ────────────────────────────────────────────────────
MIDI_RESOLUTION = 16          # Steps per bar (16th note resolution)
MAX_SEQUENCE_LENGTH = 256     # Maximum tokens per sequence
SEQUENCE_LENGTH = 64          # Fixed-length window for segmentation
NUM_PITCHES = 128             # MIDI pitch range (0-127)
NUM_VELOCITIES = 32           # Quantized velocity bins
MAX_DURATION_STEPS = 64       # Maximum duration in steps
PIANO_ROLL_FPS = 16           # Frames per second for piano roll

# ─── Vocabulary (tokenizer) ────────────────────────────────────────────────
VOCAB_SIZE = 512              # Total vocabulary size for token-based representation
PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
NOTE_ON_OFFSET = 3            # note-on tokens: 3..130 (128 pitches)
NOTE_OFF_OFFSET = 131         # note-off tokens: 131..258
VELOCITY_OFFSET = 259         # velocity tokens: 259..290 (32 bins)
TIME_SHIFT_OFFSET = 291       # time-shift tokens: 291..390 (100 steps)

# ─── Genre Mapping ──────────────────────────────────────────────────────────
GENRES = ["classical", "jazz", "rock", "pop", "electronic"]
GENRE_TO_ID = {g: i for i, g in enumerate(GENRES)}
NUM_GENRES = len(GENRES)

# ─── Task 1: LSTM Autoencoder ──────────────────────────────────────────────
AE_LATENT_DIM = 64
AE_HIDDEN_DIM = 128
AE_NUM_LAYERS = 2
AE_DROPOUT = 0.2
AE_LEARNING_RATE = 1e-3
AE_BATCH_SIZE = 64
AE_EPOCHS = 20

# ─── Task 2: VAE ───────────────────────────────────────────────────────────
VAE_LATENT_DIM = 64
VAE_HIDDEN_DIM = 128
VAE_NUM_LAYERS = 2
VAE_DROPOUT = 0.2
VAE_LEARNING_RATE = 1e-3
VAE_BATCH_SIZE = 64
VAE_EPOCHS = 20
VAE_BETA = 1.0                # KL-divergence weight (beta-VAE)
VAE_BETA_ANNEAL_EPOCHS = 15   # Epochs for beta warm-up

# ─── Task 3: Transformer ───────────────────────────────────────────────────
TF_D_MODEL = 64
TF_N_HEADS = 4
TF_NUM_LAYERS = 3
TF_D_FF = 256
TF_DROPOUT = 0.1
TF_LEARNING_RATE = 3e-4
TF_BATCH_SIZE = 32
TF_EPOCHS = 20
TF_WARMUP_STEPS = 500
TF_MAX_SEQ_LEN = 512
TF_TEMPERATURE = 1.0
TF_TOP_K = 40

# ─── Task 4: RLHF ──────────────────────────────────────────────────────────
RLHF_LEARNING_RATE = 5e-6
RLHF_ITERATIONS = 500
RLHF_BATCH_SIZE = 16
RLHF_GAMMA = 0.99             # Discount factor
RLHF_CLIP_EPSILON = 0.2       # PPO clip range

# ─── Train / Test Split ────────────────────────────────────────────────────
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42
