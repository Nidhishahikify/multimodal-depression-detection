"""
config.py - All settings in one place

YOUR DATASET (single folder, auto-split 80/20):
    dataset/audio/Depressed/       <- 600 wav files
    dataset/audio/Non-depressed/   <- 600 wav files
    dataset/image/Depressed/       <- 2000 images
    dataset/image/Non-depressed/   <- 2000 images

Audio and image are SEPARATE datasets (different people)
so audio and image models are trained independently.
Fusion uses simple probability averaging.
"""
from pathlib import Path
import os
ROOT_DIR    = Path(__file__).resolve().parent

# ── Raw dataset (your original files - never modified) ────────────────────────
DATASET_DIR       = ROOT_DIR / "dataset"
RAW_AUDIO_DEP     = DATASET_DIR / "audio" / "Depressed"
RAW_AUDIO_NONDEP  = DATASET_DIR / "audio" / "Non-depressed"
RAW_IMAGE_DEP     = DATASET_DIR / "image" / "Depressed"
RAW_IMAGE_NONDEP  = DATASET_DIR / "image" / "Non-depressed"

# ── Clean dataset (preprocessed, split into train/test) ───────────────────────
CLEAN_DIR = ROOT_DIR / "clean_dataset"
CLEAN_TRAIN_AUDIO_DEP    = CLEAN_DIR / "train" / "audio" / "Depressed"
CLEAN_TRAIN_AUDIO_NONDEP = CLEAN_DIR / "train" / "audio" / "Non-depressed"
CLEAN_TEST_AUDIO_DEP     = CLEAN_DIR / "test"  / "audio" / "Depressed"
CLEAN_TEST_AUDIO_NONDEP  = CLEAN_DIR / "test"  / "audio" / "Non-depressed"
CLEAN_TRAIN_IMAGE_DEP    = CLEAN_DIR / "train" / "image" / "Depressed"
CLEAN_TRAIN_IMAGE_NONDEP = CLEAN_DIR / "train" / "image" / "Non-depressed"
CLEAN_TEST_IMAGE_DEP     = CLEAN_DIR / "test"  / "image" / "Depressed"
CLEAN_TEST_IMAGE_NONDEP  = CLEAN_DIR / "test"  / "image" / "Non-depressed"

# ── Feature CSVs ──────────────────────────────────────────────────────────────
FEATURES_DIR          = ROOT_DIR / "features"
TRAIN_AUDIO_CSV       = FEATURES_DIR / "train_audio.csv"
TEST_AUDIO_CSV        = FEATURES_DIR / "test_audio.csv"
TRAIN_IMAGE_CSV       = FEATURES_DIR / "train_image.csv"
TEST_IMAGE_CSV        = FEATURES_DIR / "test_image.csv"

# ── Models / Results / Logs ───────────────────────────────────────────────────
MODELS_DIR  = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
LOGS_DIR    = ROOT_DIR / "logs"

# ── Labels ────────────────────────────────────────────────────────────────────
LABEL_MAP = {0: "Non-Depressed", 1: "Depressed"}

# ── Split ratio ───────────────────────────────────────────────────────────────
# 80% train, 20% test  (applied separately to each class so ratio is balanced)
TRAIN_RATIO   = 0.80
RANDOM_SEED   = 42

# ── Audio processing settings ─────────────────────────────────────────────────
AUDIO_CONFIG = {
    "sample_rate"   : 22050,
    "duration"      : 10,
    "n_mfcc"        : 40,
    "n_mels"        : 128,
    "n_chroma"      : 12,
    "hop_length"    : 512,
    "n_fft"         : 2048,
"supported_ext" : {".wav", ".mp3", ".flac", ".ogg", ".mp4", ".m4a", ".webm"},
}

# ── Image processing settings ─────────────────────────────────────────────────
IMAGE_CONFIG = {
    "target_size"   : (224, 224),
    "supported_ext" : {".jpg", ".jpeg", ".png", ".bmp", ".webp"},
}

# ── Training settings ─────────────────────────────────────────────────────────
TRAIN_CONFIG = {
    "random_state"  : 42,
    "cv_folds"      : 5,
    "smote_k"       : 5,
}

# ── Fusion weights (audio + image averaged) ───────────────────────────────────
# Since audio and image are from different people, fusion is done per-file
# at inference time only (not evaluated on matched pairs)
FUSION_WEIGHTS = {"audio": 0.50, "image": 0.50}

# ── Flask API settings ────────────────────────────────────────────────────────
FLASK_CONFIG = {
    "host"                  : "0.0.0.0",
     "port"                 : int(os.environ.get("PORT", 5000)),
    "debug"                 : False,
    "max_content_length_mb" : 50,
}