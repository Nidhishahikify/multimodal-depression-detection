"""
threshold_utils.py - shared logic for modality scores, calibration,
thresholds and threshold-aware fusion.

Used by:
  * app.py                          (inference)
  * scripts/calibrate_thresholds.py (threshold selection / validation)

NOTE: scores are *model scores*, not clinical probabilities of depression.
"""
import json
import math
from pathlib import Path

import numpy as np

INDICATOR_LABELS = {0: "Non-Depressed indicator", 1: "Depressed indicator"}
DEFAULT_THRESHOLD = 0.5

DISCLAIMER = (
    "Experimental machine-learning screening prototype. Scores are model "
    "outputs, not clinical probabilities, and this is not a medical "
    "diagnosis. Consult a qualified professional for any mental-health concern."
)


# ── Scoring ──────────────────────────────────────────────────────────────────
def get_scores(model, X):
    """
    Return (scores, kind) where scores is a 1-D array for the positive
    class (label 1 = Depressed).

    kind:
        "probability"    -> from predict_proba (0..1)
        "decision_score" -> from decision_function (uncalibrated, any range)
        "none"           -> model only gives hard labels (no usable score)
    """
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        classes = list(getattr(model, "classes_", [0, 1]))
        col = classes.index(1) if 1 in classes else proba.shape[1] - 1
        return proba[:, col], "probability"
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X)).ravel(), "decision_score"
    return None, "none"


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def apply_calibration(score, calibration):
    """Platt calibration: sigmoid(a*score + b). calibration=None -> identity."""
    if not calibration:
        return float(score)
    if calibration.get("type") == "platt":
        return sigmoid(calibration["a"] * float(score) + calibration["b"])
    return float(score)


# ── Threshold file ───────────────────────────────────────────────────────────
def load_thresholds(path):
    """
    Load thresholds.json. If it does not exist, fall back to 0.5 and mark
    the thresholds as NOT validated so the UI/API can say so honestly.
    """
    path = Path(path)
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    return data


def modality_cfg(all_cfg, name):
    entry = (all_cfg or {}).get(name) or {}
    validated = "threshold" in entry
    return {
        "threshold": float(entry.get("threshold", DEFAULT_THRESHOLD)),
        "calibration": entry.get("calibration"),
        "score_type": entry.get("score_type", "probability"),
        "strategy": entry.get("strategy", "default 0.5 (not validated)"),
        "validated": validated,
    }


def classify(score, threshold):
    return int(score >= threshold)


# ── Fusion ───────────────────────────────────────────────────────────────────
def normalize_to_threshold(score, threshold):
    """
    Piecewise-linear map so each modality's OWN threshold becomes 0.5:
        score == threshold -> 0.5
        score == 0         -> 0.0
        score == 1         -> 1.0
    This lets two modalities with different thresholds be averaged on a
    common 'decision-relative' scale. Scores must be in [0, 1].
    """
    t = min(max(float(threshold), 0.01), 0.99)
    s = min(max(float(score), 0.0), 1.0)
    if s < t:
        return 0.5 * s / t
    return 0.5 + 0.5 * (s - t) / (1.0 - t)


def fuse(audio_score, audio_thr, image_score, image_thr, w_audio=0.5, w_image=0.5,
         fusion_threshold=0.5):
    total = (w_audio + w_image) or 1.0
    wa, wi = w_audio / total, w_image / total
    na = normalize_to_threshold(audio_score, audio_thr)
    ni = normalize_to_threshold(image_score, image_thr)
    fused = wa * na + wi * ni
    return {
        "fused_score": fused,
        "audio_normalized": na,
        "image_normalized": ni,
        "weights": {"audio": wa, "image": wi},
        "threshold": fusion_threshold,
        "prediction": int(fused >= fusion_threshold),
    }