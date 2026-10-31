import json
import os
import sys
# Only add ffmpeg to PATH on Windows development environment
# Render (Linux) has ffmpeg available in the system PATH
if sys.platform == "win32" and os.path.exists(r"C:\Users\ASUS\AppData\Local\Microsoft\WinGet\Packages"):
    os.environ["PATH"] += r";C:\Users\ASUS\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin"
import tempfile
import logging
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from config import (
    MODELS_DIR, RESULTS_DIR, LOGS_DIR,
    LABEL_MAP, AUDIO_CONFIG, IMAGE_CONFIG,
    FLASK_CONFIG, FUSION_WEIGHTS,
    THRESHOLDS_FILE, FUSION_REQUIRE_SESSION_CONFIRMATION,
)
from scripts.extract_audio_features import extract_features as extract_audio
from scripts.extract_image_features import extract_features_with_reason as extract_image
from threshold_utils import (
    DISCLAIMER, INDICATOR_LABELS,
    get_scores, apply_calibration, load_thresholds, modality_cfg,
    classify, fuse,
)

LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "app.log", encoding="utf-8"),
        logging.StreamHandler()  # Also log to console for Render logs
    ],
)
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = (
    FLASK_CONFIG["max_content_length_mb"] * 1024 * 1024
)

def load_model(path: Path):
    if path.exists():
        try:
            m = joblib.load(path)
            log.info(f"Loaded: {path.name}")
            return m
        except Exception as e:
            log.warning(f"Could not load {path.name}: {e}")
    return None

# Lazy load models — only load when first needed to stay within 512MB RAM
# Models are cached after first load so subsequent requests are fast
_AUDIO_MODEL   = None
_IMAGE_MODEL   = None
_FUSION_BUNDLE = None

def get_audio_model():
    global _AUDIO_MODEL
    if _AUDIO_MODEL is None:
        _AUDIO_MODEL = load_model(MODELS_DIR / "model_audio.pkl")
    return _AUDIO_MODEL

def get_image_model():
    global _IMAGE_MODEL
    if _IMAGE_MODEL is None:
        _IMAGE_MODEL = load_model(MODELS_DIR / "model_image.pkl")
    return _IMAGE_MODEL

def get_fusion_bundle():
    global _FUSION_BUNDLE
    if _FUSION_BUNDLE is None:
        _FUSION_BUNDLE = load_model(MODELS_DIR / "model_fusion.pkl")
    return _FUSION_BUNDLE

# ── Thresholds (re-read on every request so editing thresholds.json takes effect
#    immediately without restarting the server) ──────────────────────────────────
def get_threshold_cfg(name: str) -> dict:
    return modality_cfg(load_thresholds(THRESHOLDS_FILE), name)

ALLOWED_AUDIO = {".wav", ".mp3", ".flac", ".ogg", ".mp4", ".m4a", ".webm", ".aac"}
TRUE_VALUES = {"1", "true", "yes", "on"}


def analyze_modality(name: str, model, features: np.ndarray) -> dict:
    """
    Score one modality and apply ITS OWN threshold.
    Keeps the original response keys (prediction/label/confidence/probabilities)
    so the existing UI keeps working, and adds the threshold fields.
    """
    X = features.reshape(1, -1)
    n_in = getattr(model, "n_features_in_", None)
    if n_in is not None and X.shape[1] != n_in:
        raise ValueError(
            f"{name} feature size mismatch: extractor produced {X.shape[1]} "
            f"features but the model expects {n_in}."
        )

    cfg = get_threshold_cfg(name)
    raw, kind = get_scores(model, X)
    if raw is None:
        raise ValueError(f"The {name} model cannot produce a score (hard labels only).")
    raw = float(raw[0])
    score = apply_calibration(raw, cfg["calibration"])
    pred = classify(score, cfg["threshold"])

    out = {
        "prediction"        : pred,
        "label"             : LABEL_MAP[pred],
        "classification"    : INDICATOR_LABELS[pred],
        "score"             : round(score, 4),
        "raw_score"         : round(raw, 4),
        "score_type"        : kind,
        "calibrated"        : bool(cfg["calibration"]),
        "threshold"         : round(cfg["threshold"], 4),
        "threshold_validated": cfg["validated"],
        "threshold_strategy": cfg["strategy"],
        "margin"            : round(score - cfg["threshold"], 4),
        "disclaimer"        : DISCLAIMER,
    }
    if kind == "probability":
        # kept for UI backward-compatibility; these are model scores, not clinical risk
        out["confidence"] = round(max(score, 1 - score), 4)
        out["probabilities"] = {
            LABEL_MAP[0]: round(1 - score, 4),
            LABEL_MAP[1]: round(score, 4),
        }
    out["explanation"] = explain_modality(name, out)
    return out


def explain_modality(name: str, r: dict) -> str:
    side = "at or above" if r["prediction"] == 1 else "below"
    txt = (f"The {name} model score is {r['score']} and the {name} threshold is "
           f"{r['threshold']}. The score is {side} the threshold, so the {name} "
           f"input is classified as a \"{r['classification']}\".")
    if not r["threshold_validated"]:
        txt += (" Note: no validated threshold was found (thresholds.json missing), "
                "so the default 0.5 was used.")
    return txt


def save_upload(file_storage, suffix: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    file_storage.save(tmp.name)
    return Path(tmp.name)

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/health")
def health():
    return jsonify({
        "status"            : "ok",
        "audio_model_file"  : (MODELS_DIR / "model_audio.pkl").exists(),
        "image_model_file"  : (MODELS_DIR / "model_image.pkl").exists(),
        "fusion_model_file" : (MODELS_DIR / "model_fusion.pkl").exists(),
        "thresholds_file"   : THRESHOLDS_FILE.exists(),
        "audio_model_loaded": _AUDIO_MODEL is not None,
        "image_model_loaded": _IMAGE_MODEL is not None,
        "fusion_model_loaded": _FUSION_BUNDLE is not None,
    })

@app.get("/thresholds")
def thresholds():
    """Shows the thresholds currently in use (handy for the presentation)."""
    return jsonify({
        "audio": get_threshold_cfg("audio"),
        "image": get_threshold_cfg("image"),
        "fusion_weights": FUSION_WEIGHTS,
        "disclaimer": DISCLAIMER,
    })

@app.get("/metrics")
def metrics():
    data = {}
    for p in RESULTS_DIR.glob("metrics_*.json"):
        with open(p, encoding="utf-8") as fh:
            data[p.stem.replace("metrics_", "")] = json.load(fh)
    summary = RESULTS_DIR / "accuracy_summary.csv"
    if summary.exists():
        import pandas as pd
        data["summary"] = pd.read_csv(summary).to_dict(orient="records")
    return jsonify(data)

@app.post("/predict/audio")
def predict_audio():
    model = get_audio_model()
    if model is None:
        return jsonify({"error": "Audio model not loaded. Run main.py first."}), 503
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Use key: 'file'"}), 400

    f   = request.files["file"]
    ext = Path(f.filename).suffix.lower()

    if ext not in ALLOWED_AUDIO:
        return jsonify({"error": f"Unsupported format: {ext}. Use wav/mp3/mp4/flac/m4a"}), 400

    tmp = save_upload(f, ext)
    try:
        feat, reason = extract_audio(tmp, return_reason=True)
        if feat is None:
            return jsonify({"error": reason or "Could not extract features. Install ffmpeg for mp4/m4a support."}), 422
        try:
            return jsonify(analyze_modality("audio", model, feat))
        except ValueError as e:
            return jsonify({"error": str(e)}), 500
    finally:
        tmp.unlink(missing_ok=True)

@app.post("/predict/image")
def predict_image():
    model = get_image_model()
    if model is None:
        return jsonify({"error": "Image model not loaded. Run main.py first."}), 503
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Use key: 'file'"}), 400

    f   = request.files["file"]
    ext = Path(f.filename).suffix.lower()
    if ext not in IMAGE_CONFIG["supported_ext"]:
        return jsonify({"error": f"Unsupported format: {ext}. Use jpg/png"}), 400

    tmp = save_upload(f, ext)
    try:
        feat, reason = extract_image(tmp)
        if feat is None:
            return jsonify({"error": "Could not extract features from image", "detail": reason or "Unknown image processing error"}), 422
        try:
            return jsonify(analyze_modality("image", model, feat))
        except ValueError as e:
            return jsonify({"error": str(e)}), 500
    finally:
        tmp.unlink(missing_ok=True)

@app.post("/predict/fusion")
def predict_fusion():
    """
    Combined audio + image prediction.

    Both files must come from the SAME person in the SAME recording session.
    Send the form field  same_session=true  to confirm this (optionally also
    session_id=<any string>). If FUSION_REQUIRE_SESSION_CONFIRMATION is True in
    config.py the request is rejected without it; otherwise it is accepted but
    the response is flagged  pairing_verified=false.
    """
    audio_model = get_audio_model()
    image_model = get_image_model()
    if audio_model is None or image_model is None:
        return jsonify({"error": "Audio/image model not loaded. Run main.py first."}), 503
    if "audio" not in request.files or "image" not in request.files:
        return jsonify({"error": "Provide both 'audio' and 'image' files"}), 400

    paired = request.form.get("same_session", "").strip().lower() in TRUE_VALUES
    session_id = request.form.get("session_id")
    if FUSION_REQUIRE_SESSION_CONFIRMATION and not paired:
        return jsonify({
            "error": "Fusion requires audio and image from the same person and the same "
                     "recording session. Send form field same_session=true to confirm.",
        }), 400

    # Fusion weights: use the existing fusion bundle if present, else config.
    bundle  = get_fusion_bundle()
    weights = (bundle or {}).get("weights", FUSION_WEIGHTS) if isinstance(bundle, dict) else FUSION_WEIGHTS

    a_ext = Path(request.files["audio"].filename).suffix.lower()
    i_ext = Path(request.files["image"].filename).suffix.lower()
    a_tmp = save_upload(request.files["audio"], a_ext)
    i_tmp = save_upload(request.files["image"], i_ext)

    try:
        a_feat, a_reason = extract_audio(a_tmp, return_reason=True)
        i_feat, reason = extract_image(i_tmp)
        if a_feat is None:
            return jsonify({"error": a_reason or "Could not extract audio features."}), 422
        if i_feat is None:
            return jsonify({"error": "Feature extraction failed", "detail": reason or "Unknown image processing error"}), 422

        try:
            audio = analyze_modality("audio", audio_model, a_feat)
            image = analyze_modality("image", image_model, i_feat)
        except ValueError as e:
            return jsonify({"error": str(e)}), 500
        if audio["score_type"] != "probability" or image["score_type"] != "probability":
            return jsonify({"error": "Fusion needs probability-type scores from both models."}), 422

        f = fuse(audio["score"], audio["threshold"], image["score"], image["threshold"],
                 w_audio=weights["audio"], w_image=weights["image"])
        pred = f["prediction"]
        fs = f["fused_score"]
        explanation = (
            f"Audio: score {audio['score']} vs threshold {audio['threshold']} -> "
            f"{audio['classification']}. Image: score {image['score']} vs threshold "
            f"{image['threshold']} -> {image['classification']}. Each score is first "
            f"rescaled so its own threshold equals 0.5, then the two are combined with "
            f"weights audio={f['weights']['audio']:.2f} / image={f['weights']['image']:.2f}. "
            f"The combined score is {fs:.4f} (decision point 0.5), giving a combined "
            f"\"{INDICATOR_LABELS[pred]}\"."
        )
        if audio["prediction"] != image["prediction"]:
            explanation += " The two modalities disagree, so this result is less certain."
        if not paired:
            explanation += (" WARNING: it was not confirmed that the audio and image come "
                            "from the same person/session, so this combination is not reliable.")

        return jsonify({
            "prediction"      : pred,
            "label"           : LABEL_MAP[pred],
            "classification"  : INDICATOR_LABELS[pred],
            "confidence"      : round(max(fs, 1 - fs), 4),   # distance-from-boundary style score, not clinical
            "probabilities"   : {
                LABEL_MAP[0]: round(1 - fs, 4),
                LABEL_MAP[1]: round(fs, 4),
            },
            "fused_score"     : round(fs, 4),
            "fusion_threshold": f["threshold"],
            "fusion_method"   : "threshold-normalised weighted average",
            "fusion_validated_on_paired_data": False,
            "pairing_verified": paired,
            "session_id"      : session_id,
            "modality_scores" : {          # kept for UI backward-compatibility
                "audio": audio["score"],
                "image": image["score"],
            },
            "audio"           : audio,
            "image"           : image,
            "explanation"     : explanation,
            "disclaimer"      : DISCLAIMER,
        })
    finally:
        a_tmp.unlink(missing_ok=True)
        i_tmp.unlink(missing_ok=True)

if __name__ == "__main__":
    print("=" * 50)
    print("  MindScan - Depression Detection")
    print(f"  http://127.0.0.1:{FLASK_CONFIG['port']}")
    print(f"  Audio model  : {'FOUND' if (MODELS_DIR / 'model_audio.pkl').exists() else 'NOT FOUND - run main.py'}")
    print(f"  Image model  : {'FOUND' if (MODELS_DIR / 'model_image.pkl').exists() else 'NOT FOUND - run main.py'}")
    print(f"  Fusion model : {'FOUND' if (MODELS_DIR / 'model_fusion.pkl').exists() else 'NOT FOUND - run main.py'}")
    print(f"  Thresholds   : {'FOUND' if THRESHOLDS_FILE.exists() else 'NOT FOUND - run scripts/calibrate_thresholds.py'}")
    print("=" * 50)
    app.run(
        host=FLASK_CONFIG["host"],
        port=FLASK_CONFIG["port"],
        debug=False,
        use_reloader=False,
    )
else:
    # When run via gunicorn, log startup info
    log.info("=" * 50)
    log.info("MindScan starting via gunicorn")
    log.info(f"Audio model exists: {(MODELS_DIR / 'model_audio.pkl').exists()}")
    log.info(f"Image model exists: {(MODELS_DIR / 'model_image.pkl').exists()}")
    log.info(f"Fusion model exists: {(MODELS_DIR / 'model_fusion.pkl').exists()}")
    log.info(f"Thresholds file exists: {THRESHOLDS_FILE.exists()}")
    log.info(f"Host: {FLASK_CONFIG['host']}, Port: {FLASK_CONFIG['port']}")
    log.info("=" * 50)