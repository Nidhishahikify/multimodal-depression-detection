# MindScan Deployment Guide for Render

## Issues Fixed

### 1. **Windows-Specific PATH Configuration**
- **Problem**: Hardcoded Windows ffmpeg path failed on Linux servers
- **Fix**: Added platform check to only use Windows path in local development

### 2. **Missing Deployment Configuration**
- **Problem**: No Procfile or render.yaml for Render deployment
- **Fix**: Created both files with proper gunicorn configuration

### 3. **Gunicorn Version Pinning**
- **Problem**: Unversioned gunicorn dependency
- **Fix**: Pinned to `gunicorn==21.2.0`

### 4. **Logging for Production**
- **Problem**: Logs only went to file, not visible in Render dashboard
- **Fix**: Added StreamHandler to send logs to console

## Deployment Steps

### Pre-Deployment Checklist
1. Ensure all three model files exist in `models/` directory:
   - `model_audio.pkl`
   - `model_image.pkl`
   - `model_fusion.pkl`

2. Commit all changes to Git:
   ```bash
   git add .
   git commit -m "Fix Render deployment configuration"
   git push
   ```

### Deploy to Render

1. **Go to Render Dashboard**: https://dashboard.render.com
2. **Select your service**: `multimodal-depression-detection-66iq`
3. **Check Settings**:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
   - **Auto-Deploy**: Enabled (will redeploy on git push)

4. **Trigger Manual Deploy**:
   - Click "Manual Deploy" → "Deploy latest commit"
   - Or push to your repository to auto-deploy

### Verify Deployment

After deployment completes (5-10 minutes), check these in order:

#### 1. Check Logs Tab in Render Dashboard
Look for these SUCCESS indicators:
```
==> Starting service with 'gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120'
[INFO] Starting gunicorn 21.2.0
[INFO] Listening at: http://0.0.0.0:10000
MindScan starting via gunicorn
Audio model exists: True
Image model exists: True
Fusion model exists: True
```

Look for these FAILURE indicators (and fixes):
```
❌ "ModuleNotFoundError: No module named 'cv2'"
   → Fix: opencv-python-headless is in requirements.txt, rebuild should work

❌ "FileNotFoundError: models/model_audio.pkl"
   → Fix: Models not committed to git. Ensure models/ directory is tracked

❌ "MemoryError" or "Killed"
   → Fix: Models too large for free tier (512MB RAM limit)
   → Solution: Upgrade to Starter plan ($7/mo, 512MB → 1GB RAM)
   → OR: Reduce model size by using simpler algorithms

❌ "ImportError: librosa" or "soundfile"
   → Fix: Missing system dependencies for audio processing
   → Solution: Add to render.yaml buildCommand: 
     "apt-get update && apt-get install -y libsndfile1 ffmpeg && pip install -r requirements.txt"
```

#### 2. Test Health Endpoint
Open in browser or curl:
```bash
curl https://multimodal-depression-detection-66iq.onrender.com/health
```

Expected response:
```json
{
  "status": "ok",
  "audio_model_file": true,
  "image_model_file": true,
  "fusion_model_file": true,
  "audio_model_loaded": false,  # false until first use (lazy loading)
  "image_model_loaded": false,
  "fusion_model_loaded": false
}
```

#### 3. Test Frontend
Visit: https://multimodal-depression-detection-66iq.onrender.com
- Status pill should show "Audio + Image + Fusion Ready" with green dot
- Upload a test audio file
- Click "Analyse Voice"
- Should see results panel with prediction

#### 4. Test API Directly
```bash
# Test audio endpoint
curl -X POST \
  https://multimodal-depression-detection-66iq.onrender.com/predict/audio \
  -F "file=@test_audio.wav"
```

## Common Issues and Solutions

### Issue: "This site can't be reached"
- **Cause**: Service failed to start
- **Check**: Render Logs tab for startup errors
- **Fix**: Look for error messages and apply fixes above

### Issue: "502 Bad Gateway"
- **Cause**: App crashed or timed out
- **Check**: Logs tab for Python exceptions
- **Fix**: Usually memory limit exceeded on free tier

### Issue: Models fail to load
- **Cause**: Model files not in repository or too large
- **Check**: Logs show "FileNotFoundError" or "MemoryError"
- **Fix**: 
  1. Ensure `models/` directory is in git (check `.gitignore`)
  2. Check model file sizes: `ls -lh models/*.pkl`
  3. If >100MB each, may exceed free tier memory

### Issue: Audio processing fails
- **Cause**: Missing ffmpeg or audio libraries
- **Check**: Logs show librosa/soundfile errors
- **Fix**: Update render.yaml buildCommand:
```yaml
buildCommand: apt-get update && apt-get install -y libsndfile1 ffmpeg && pip install -r requirements.txt
```

### Issue: Image processing fails
- **Cause**: Missing OpenCV system dependencies
- **Check**: Logs show cv2 import errors
- **Fix**: opencv-python-headless should work, but if not:
```yaml
buildCommand: apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && pip install -r requirements.txt
```

## Environment Variables (if needed)

In Render Dashboard → Environment:
- `PORT`: Auto-set by Render (DO NOT change)
- `PYTHON_VERSION`: 3.11.0 (set in runtime.txt)

## Performance Notes

**Free Tier Limitations**:
- 512MB RAM (may be tight with 3 large models)
- Services spin down after 15 min inactivity
- First request after spin-down takes 30-60s

**To improve performance**:
1. Upgrade to Starter ($7/mo) for 1GB RAM + no spin-down
2. Implement model caching with redis
3. Use model compression techniques
4. Split audio/image/fusion into separate microservices

## Monitoring

**Key Metrics to Watch**:
- Response time (Render Dashboard → Metrics)
- Memory usage (should stay <80% of limit)
- Error rate in Logs tab
- Model loading time (first request is slower)

**Set up Alerts**:
- Render → Settings → Notifications
- Get alerted on deployment failures or crashes
