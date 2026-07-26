"""
scripts/extract_audio_features.py

Extracts 240 features from each audio file.
Runs on TRAIN and TEST folders separately.
Saves two CSVs: train_audio.csv and test_audio.csv
"""
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import os
os.environ["PATH"] += r";C:\Users\ASUS\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin"
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    CLEAN_TRAIN_AUDIO_DEP, CLEAN_TRAIN_AUDIO_NONDEP,
    CLEAN_TEST_AUDIO_DEP,  CLEAN_TEST_AUDIO_NONDEP,
    TRAIN_AUDIO_CSV, TEST_AUDIO_CSV,
    AUDIO_CONFIG, LOGS_DIR, FEATURES_DIR,
)

LOGS_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.FileHandler(LOGS_DIR / "extract_audio.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)


def extract_features(path: Path):
    """
    Returns a 240-dim numpy array:
      40 MFCC means + 40 MFCC stds         = 80
      12 Chroma means + 12 Chroma stds      = 24
      Centroid mean+std                     =  2
      Bandwidth mean+std                    =  2
      Rolloff mean+std                      =  2
      ZCR mean+std                          =  2
      128 Mel spectrogram means             = 128
      TOTAL                                 = 240
    """
    try:
        sr    = AUDIO_CONFIG["sample_rate"]
        y, _  = librosa.load(str(path), sr=sr)

        # Trim leading/trailing silence. Dataset clips tend to be clean speech
        # segments; a live mic recording usually has silence before/after the
        # person talks, which otherwise skews the MFCC/mel statistics.
        y, _ = librosa.effects.trim(y, top_db=25)

        # Enforce a fixed clip length so every sample (train, test, and live
        # recordings) is compared over the same time window. AUDIO_CONFIG had
        # a "duration" setting that was never actually applied - this applies it.
        target_len = int(AUDIO_CONFIG["duration"] * sr)
        if len(y) > target_len:
            y = y[:target_len]
        elif len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))

        if y.size == 0 or not np.any(y):
            log.warning(f"Empty/silent audio after trim: {path.name}")
            return None

        hop   = AUDIO_CONFIG["hop_length"]
        n_fft = AUDIO_CONFIG["n_fft"]
        feats = []

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr,
                                     n_mfcc=AUDIO_CONFIG["n_mfcc"],
                                     hop_length=hop, n_fft=n_fft)
        feats.extend(np.mean(mfcc, axis=1))
        feats.extend(np.std(mfcc,  axis=1))

        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr,
                                              n_chroma=AUDIO_CONFIG["n_chroma"],
                                              hop_length=hop, n_fft=n_fft)
        feats.extend(np.mean(chroma, axis=1))
        feats.extend(np.std(chroma,  axis=1))

        # Spectral features
        for feat in (
            librosa.feature.spectral_centroid( y=y, sr=sr, hop_length=hop, n_fft=n_fft),
            librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop, n_fft=n_fft),
            librosa.feature.spectral_rolloff(  y=y, sr=sr, hop_length=hop, n_fft=n_fft),
            librosa.feature.zero_crossing_rate(y, hop_length=hop),
        ):
            feats.append(float(np.mean(feat)))
            feats.append(float(np.std(feat)))

        # Mel spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr,
                                              n_mels=AUDIO_CONFIG["n_mels"],
                                              hop_length=hop, n_fft=n_fft)
        feats.extend(np.mean(librosa.power_to_db(mel, ref=np.max), axis=1))

        return np.array(feats, dtype=np.float32)

    except Exception as e:
        log.warning(f"Failed [{path.name}]: {e}")
        return None


def build_column_names():
    cols = []
    for i in range(AUDIO_CONFIG["n_mfcc"]):    cols.append(f"mfcc_mean_{i+1}")
    for i in range(AUDIO_CONFIG["n_mfcc"]):    cols.append(f"mfcc_std_{i+1}")
    for i in range(AUDIO_CONFIG["n_chroma"]):  cols.append(f"chroma_mean_{i+1}")
    for i in range(AUDIO_CONFIG["n_chroma"]):  cols.append(f"chroma_std_{i+1}")
    for name in ("centroid", "bandwidth", "rolloff", "zcr"):
        cols += [f"{name}_mean", f"{name}_std"]
    for i in range(AUDIO_CONFIG["n_mels"]):    cols.append(f"mel_{i+1}")
    return cols


def extract_split(dep_dir: Path, nondep_dir: Path,
                  out_csv: Path, split_name: str) -> int:
    print(f"\n  Extracting {split_name} audio features...")
    cols = build_column_names()
    rows = []
    ext  = AUDIO_CONFIG["supported_ext"] | {".wav"}

    for label_val, label_name, src_dir in [
        (1, "Depressed",    dep_dir),
        (0, "Non-depressed", nondep_dir),
    ]:
        if not src_dir.exists():
            print(f"  [WARN] Not found: {src_dir}")
            continue
        files = [f for f in src_dir.rglob("*") if f.suffix.lower() in ext]
        print(f"    {label_name}: {len(files)} files")
        ok = skip = 0
        for f in tqdm(files, desc=f"    {label_name}", leave=False):
            feat = extract_features(f)
            if feat is not None:
                row = {"filename": f.name, "label": label_val}
                for c, v in zip(cols, feat):
                    row[c] = v
                rows.append(row)
                ok += 1
            else:
                skip += 1
        print(f"    -> OK: {ok}  Skipped: {skip}")

    if not rows:
        print(f"  [ERROR] No features extracted for {split_name}!")
        return 0

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    counts = df["label"].value_counts().to_dict()
    print(f"  Saved {len(df)} rows -> {out_csv.name}")
    print(f"  Class counts: Depressed={counts.get(1,0)}, Non-depressed={counts.get(0,0)}")
    return len(df)


def main():
    print("=" * 55)
    print("  STEP 2a: AUDIO FEATURE EXTRACTION")
    print("=" * 55)

    n_train = extract_split(
        CLEAN_TRAIN_AUDIO_DEP, CLEAN_TRAIN_AUDIO_NONDEP,
        TRAIN_AUDIO_CSV, "TRAIN"
    )
    n_test = extract_split(
        CLEAN_TEST_AUDIO_DEP, CLEAN_TEST_AUDIO_NONDEP,
        TEST_AUDIO_CSV, "TEST"
    )

    print(f"\n  DONE: Train={n_train} | Test={n_test}")


if __name__ == "__main__":
    main()