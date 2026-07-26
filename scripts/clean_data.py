"""
scripts/clean_data.py

What this does:
1. Reads your single dataset folder
2. Shuffles files randomly
3. Splits 80% -> train, 20% -> test  (per class, so balance is kept)
4. Cleans each file (audio: resample/normalize, image: resize)
5. Saves to clean_dataset/train/ and clean_dataset/test/

Audio: 600 dep + 600 nondep -> 480 train + 120 test each
Image: 2000 dep + 2000 nondep -> 1600 train + 400 test each
"""
import sys
import random
import shutil
import logging
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    RAW_AUDIO_DEP, RAW_AUDIO_NONDEP,
    RAW_IMAGE_DEP, RAW_IMAGE_NONDEP,
    CLEAN_TRAIN_AUDIO_DEP, CLEAN_TRAIN_AUDIO_NONDEP,
    CLEAN_TEST_AUDIO_DEP,  CLEAN_TEST_AUDIO_NONDEP,
    CLEAN_TRAIN_IMAGE_DEP, CLEAN_TRAIN_IMAGE_NONDEP,
    CLEAN_TEST_IMAGE_DEP,  CLEAN_TEST_IMAGE_NONDEP,
    AUDIO_CONFIG, IMAGE_CONFIG,
    TRAIN_RATIO, RANDOM_SEED, LOGS_DIR,
)

LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.FileHandler(LOGS_DIR / "clean_data.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)
random.seed(RANDOM_SEED)


# ── Audio cleaning ────────────────────────────────────────────────────────────
def clean_audio(src: Path, dst: Path) -> bool:
    try:
        sr  = AUDIO_CONFIG["sample_rate"]
        dur = AUDIO_CONFIG["duration"]
        y, _ = librosa.load(str(src), sr=sr, mono=True, duration=dur)
        y, _ = librosa.effects.trim(y, top_db=25)
        required = sr * dur
        if len(y) < required:
            y = np.pad(y, (0, required - len(y)))
        else:
            y = y[:required]
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dst.with_suffix(".wav")), y, sr)
        return True
    except Exception as e:
        log.warning(f"Audio skip {src.name}: {e}")
        return False


# ── Image cleaning ────────────────────────────────────────────────────────────
def clean_image(src: Path, dst: Path) -> bool:
    try:
        img = cv2.imread(str(src))
        if img is None:
            raise ValueError("Cannot read image")
        img = cv2.resize(img, IMAGE_CONFIG["target_size"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst.with_suffix(".jpg")), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        return True
    except Exception as e:
        log.warning(f"Image skip {src.name}: {e}")
        return False


# ── Split + clean one class ───────────────────────────────────────────────────
def process_class(src_dir: Path, train_dst: Path, test_dst: Path,
                  supported_ext: set, clean_fn, label: str, modality: str):
    if not src_dir.exists():
        print(f"  [WARN] Folder not found: {src_dir}")
        print(f"  Make sure your files are in: {src_dir}")
        return 0, 0

    files = sorted([f for f in src_dir.rglob("*")
                    if f.suffix.lower() in supported_ext])
    if not files:
        print(f"  [WARN] No files found in {src_dir}")
        return 0, 0

    random.shuffle(files)
    split_idx  = int(len(files) * TRAIN_RATIO)
    train_files = files[:split_idx]
    test_files  = files[split_idx:]

    print(f"  {modality}/{label}: {len(files)} total -> "
          f"{len(train_files)} train + {len(test_files)} test")

    ok = skip = 0
    for f in tqdm(train_files, desc=f"  Train/{label}", leave=False):
        if clean_fn(f, train_dst / f.name): ok += 1
        else: skip += 1
    for f in tqdm(test_files, desc=f"  Test/{label}", leave=False):
        if clean_fn(f, test_dst  / f.name): ok += 1
        else: skip += 1

    return len(train_files), len(test_files)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  STEP 1: DATA CLEANING + TRAIN/TEST SPLIT (80/20)")
    print("=" * 55)

    audio_ext = AUDIO_CONFIG["supported_ext"] | {".wav"}
    image_ext = IMAGE_CONFIG["supported_ext"] | {".jpg"}

    print("\n  -- Audio --")
    a_tr_d, a_te_d = process_class(
        RAW_AUDIO_DEP, CLEAN_TRAIN_AUDIO_DEP, CLEAN_TEST_AUDIO_DEP,
        audio_ext, clean_audio, "Depressed", "Audio")
    a_tr_n, a_te_n = process_class(
        RAW_AUDIO_NONDEP, CLEAN_TRAIN_AUDIO_NONDEP, CLEAN_TEST_AUDIO_NONDEP,
        audio_ext, clean_audio, "Non-depressed", "Audio")

    print("\n  -- Image --")
    i_tr_d, i_te_d = process_class(
        RAW_IMAGE_DEP, CLEAN_TRAIN_IMAGE_DEP, CLEAN_TEST_IMAGE_DEP,
        image_ext, clean_image, "Depressed", "Image")
    i_tr_n, i_te_n = process_class(
        RAW_IMAGE_NONDEP, CLEAN_TRAIN_IMAGE_NONDEP, CLEAN_TEST_IMAGE_NONDEP,
        image_ext, clean_image, "Non-depressed", "Image")

    print("\n" + "=" * 55)
    print("  SPLIT SUMMARY")
    print("=" * 55)
    print(f"  Audio Train : {a_tr_d} Depressed + {a_tr_n} Non-depressed = {a_tr_d+a_tr_n}")
    print(f"  Audio Test  : {a_te_d} Depressed + {a_te_n} Non-depressed = {a_te_d+a_te_n}")
    print(f"  Image Train : {i_tr_d} Depressed + {i_tr_n} Non-depressed = {i_tr_d+i_tr_n}")
    print(f"  Image Test  : {i_te_d} Depressed + {i_te_n} Non-depressed = {i_te_d+i_te_n}")
    print("=" * 55)
    print("  CLEANING COMPLETE")


if __name__ == "__main__":
    main()