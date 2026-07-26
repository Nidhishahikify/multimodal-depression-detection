# ── MindScan deployment image ────────────────────────────────────────────────
FROM python:3.11-slim

# System dependencies:
#   ffmpeg      -> needed by librosa for mp4/m4a/webm audio decoding
#   libgomp1    -> needed by scikit-learn / scikit-image at runtime
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=5000
EXPOSE 5000

# gunicorn instead of the dev server. Long timeout because audio/image feature
# extraction (librosa + HOG + face detection) can take a few seconds per request.
CMD gunicorn -b 0.0.0.0:${PORT} --timeout 120 --workers 2 app:app
