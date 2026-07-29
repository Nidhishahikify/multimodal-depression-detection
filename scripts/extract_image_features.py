"""
scripts/extract_image_features.py

Extracts ~2140 features from each image.
Runs on TRAIN and TEST folders separately.
Saves two CSVs: train_image.csv and test_image.csv

NOTE: Face detection is now MANDATORY. If no face is found in an image
(e.g. a screenshot, blank/black frame, or non-face photo), that image is
skipped entirely — it is NOT analyzed as depressed/non-depressed, and no
row is written for it. This prevents non-face content from polluting the
feature CSVs.
"""
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from skimage.feature import hog, local_binary_pattern
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    CLEAN_TRAIN_IMAGE_DEP, CLEAN_TRAIN_IMAGE_NONDEP,
    CLEAN_TEST_IMAGE_DEP,  CLEAN_TEST_IMAGE_NONDEP,
    TRAIN_IMAGE_CSV, TEST_IMAGE_CSV,
    IMAGE_CONFIG, LOGS_DIR, FEATURES_DIR,
)

LOGS_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.FileHandler(LOGS_DIR / "extract_image.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)

# ── Face detector ──────────────────────────────────────────────────────────────
# Loaded once at import time and reused for every image. This is what makes a
# live webcam capture (which includes background/hair/shoulders) get normalized
# to roughly the same framing as a pre-cropped training-dataset face photo.
_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def _crop_to_face(img: np.ndarray):
    """
    Detects the largest face in the image and crops to it (with a small margin).

    Returns the cropped face image, or None if no face is detected.
    Face detection is REQUIRED — images with no detectable face (screenshots,
    blank frames, non-face photos, etc.) must NOT be analyzed, so callers
    should skip the image entirely when this returns None.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return None

    # Largest detected face by area (in case of multiple faces in frame)
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Add ~20% margin around the face box so we don't cut off chin/forehead
    m = int(0.2 * w)
    x0, y0 = max(x - m, 0), max(y - m, 0)
    x1, y1 = min(x + w + m, img.shape[1]), min(y + h + m, img.shape[0])

    return img[y0:y1, x0:x1]


def extract_features(path: Path):
    """
    Returns feature vector:
      HOG descriptor      ~1764 values  (shape/edges)
      LBP histogram         256 values  (texture)
      HSV color histogram    96 values  (color)
      Region stats           24 values  (6 zones x 4 stats)
      TOTAL               ~2140 values

    Returns None (and logs why) if the image can't be read OR if no face
    is detected in it. No-face images are intentionally NOT analyzed.
    """
    try:
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError("Cannot read image")

        face = _crop_to_face(img)
        if face is None:
            log.info(f"No face detected, skipping [{path.name}]")
            return None

        img  = cv2.resize(face, IMAGE_CONFIG["target_size"])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # HOG - shape and edge structure
        hog_feat = hog(
            gray,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        ).astype(np.float32)

        # LBP - texture
        lbp = local_binary_pattern(gray, 24, 3, method="uniform")
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=256,
                                    range=(0, 256), density=True)
        lbp_hist = lbp_hist.astype(np.float32)

        # HSV color histogram
        hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        color = []
        for ch, (lo, hi) in enumerate([(0, 180), (0, 256), (0, 256)]):
            h = cv2.calcHist([hsv], [ch], None, [32], [lo, hi])
            color.extend(cv2.normalize(h, h).flatten())
        color = np.array(color, dtype=np.float32)

        # Region stats - 6 face zones
        h, w = gray.shape
        zones = [
            gray[:h//3,    :],               # upper (forehead/brow)
            gray[h//3:2*h//3, :],            # middle (eyes/nose)
            gray[2*h//3:,  :],               # lower (mouth/chin)
            gray[:, :w//2],                  # left half
            gray[:, w//2:],                  # right half
            gray[h//4:3*h//4, w//4:3*w//4], # centre crop
        ]
        region_feats = []
        for z in zones:
            region_feats += [
                float(np.mean(z)),
                float(np.std(z)),
                float(np.percentile(z, 25)),
                float(np.percentile(z, 75)),
            ]
        region_feats = np.array(region_feats, dtype=np.float32)

        return np.concatenate([hog_feat, lbp_hist, color, region_feats])

    except Exception as e:
        log.warning(f"Failed [{path.name}]: {e}")
        return None


def extract_split(dep_dir: Path, nondep_dir: Path,
                  out_csv: Path, split_name: str) -> int:
    print(f"\n  Extracting {split_name} image features...")
    rows = []
    ext  = IMAGE_CONFIG["supported_ext"] | {".jpg"}

    for label_val, label_name, src_dir in [
        (1, "Depressed",     dep_dir),
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
                for i, v in enumerate(feat):
                    row[f"img_{i}"] = v
                rows.append(row)
                ok += 1
            else:
                skip += 1
        print(f"    -> OK: {ok}  Skipped (unreadable or no face detected): {skip}")

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
    print("  STEP 2b: IMAGE FEATURE EXTRACTION")
    print("=" * 55)

    n_train = extract_split(
        CLEAN_TRAIN_IMAGE_DEP, CLEAN_TRAIN_IMAGE_NONDEP,
        TRAIN_IMAGE_CSV, "TRAIN"
    )
    n_test = extract_split(
        CLEAN_TEST_IMAGE_DEP, CLEAN_TEST_IMAGE_NONDEP,
        TEST_IMAGE_CSV, "TEST"
    )

    print(f"\n  DONE: Train={n_train} | Test={n_test}")


if __name__ == "__main__":
    main()