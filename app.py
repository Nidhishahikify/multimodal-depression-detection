# ...existing code...
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
)
from scripts.extract_audio_features import extract_features as extract_audio
from scripts.extract_image_features import extract_features_with_reason as extract_image

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

# ...existing code...
ALLOWED_AUDIO = {".wav", ".mp3", ".flac", ".ogg", ".mp4", ".m4a", ".webm", ".aac"}

def make_prediction(model, features: np.ndarray) -> dict:
    X     = features.reshape(1, -1)
    pred  = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0].tolist()
    return {
        "prediction"    : pred,
        "label"         : LABEL_MAP[pred],
        "confidence"    : round(max(proba), 4),
        "probabilities" : {
            LABEL_MAP[0]: round(proba[0], 4),
            LABEL_MAP[1]: round(proba[1], 4),
        },
    }

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
        "audio_model_loaded": _AUDIO_MODEL is not None,
        "image_model_loaded": _IMAGE_MODEL is not None,
        "fusion_model_loaded": _FUSION_BUNDLE is not None,
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
        feat = extract_audio(tmp)
        if feat is None:
            return jsonify({"error": "Could not extract features. Install ffmpeg for mp4/m4a support."}), 422
        return jsonify(make_prediction(model, feat))
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
        return jsonify(make_prediction(model, feat))
    finally:
        tmp.unlink(missing_ok=True)

@app.post("/predict/fusion")
def predict_fusion():
    bundle = get_fusion_bundle()
    if bundle is None:
        return jsonify({"error": "Fusion model not loaded. Run main.py first."}), 503
    if "audio" not in request.files or "image" not in request.files:
        return jsonify({"error": "Provide both 'audio' and 'image' files"}), 400

    a_pipe  = bundle["audio"]
    i_pipe  = bundle["image"]
    weights = bundle["weights"]

    a_ext = Path(request.files["audio"].filename).suffix.lower()
    i_ext = Path(request.files["image"].filename).suffix.lower()
    a_tmp = save_upload(request.files["audio"], a_ext)
    i_tmp = save_upload(request.files["image"], i_ext)

    try:
        a_feat = extract_audio(a_tmp)
        i_feat, reason = extract_image(i_tmp)
        if a_feat is None or i_feat is None:
            return jsonify({"error": "Feature extraction failed", "detail": reason or "Unknown image processing error"}), 422

        p_a = float(a_pipe.predict_proba(a_feat.reshape(1, -1))[0][1])
        p_i = float(i_pipe.predict_proba(i_feat.reshape(1, -1))[0][1])
        p_f = weights["audio"] * p_a + weights["image"] * p_i
        pred = int(p_f >= 0.5)

        return jsonify({
            "prediction"      : pred,
            "label"           : LABEL_MAP[pred],
            "confidence"      : round(max(p_f, 1 - p_f), 4),
            "probabilities"   : {
                LABEL_MAP[0]: round(1 - p_f, 4),
                LABEL_MAP[1]: round(p_f, 4),
            },
            "modality_scores" : {
                "audio": round(p_a, 4),
                "image": round(p_i, 4),
            },
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
    log.info(f"Host: {FLASK_CONFIG['host']}, Port: {FLASK_CONFIG['port']}")
    log.info("=" * 50)
# ...existing code...