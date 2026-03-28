"""
API routes for the AI-Powered Multi-Modal Fake Content Detector.

Endpoints:
    POST /predict/image  – detect deepfakes in images
    POST /predict/video  – detect deepfakes in videos
    POST /predict/audio  – detect deepfakes in audio
    GET  /health         – service health check
    GET  /results        – list recent detection results (from MongoDB)
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from backend.utils.explainability import explain_audio, explain_image, explain_video
from backend.utils.preprocessing import validate_audio, validate_image, validate_video

logger = logging.getLogger(__name__)
router = APIRouter()


# ──────────────────────────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/health", tags=["system"])
async def health_check():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


# ──────────────────────────────────────────────────────────────────────────────
# Image prediction
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/predict/image", tags=["prediction"])
async def predict_image(request: Request, file: UploadFile = File(...)):
    """
    Analyse an uploaded image for deepfake content.

    Returns:
        prediction  : "real" | "fake"
        confidence  : float in [0, 1]
        heatmap_url : base64-encoded Grad-CAM overlay PNG (data URI)
    """
    content_type = file.content_type or ""
    contents = await file.read()
    try:
        validate_image(content_type, len(contents))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    image_model = request.app.state.image_model
    try:
        result = image_model.predict(contents)
        heatmap = explain_image(image_model, contents)
    except Exception as exc:
        logger.exception("Image prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    response = {
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "raw_score": result["raw_score"],
        "heatmap_url": heatmap,
        "filename": file.filename,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    await _save_result(request, "image", response)
    return response


# ──────────────────────────────────────────────────────────────────────────────
# Video prediction
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/predict/video", tags=["prediction"])
async def predict_video(request: Request, file: UploadFile = File(...)):
    """
    Analyse an uploaded video for deepfake content (frame-level CNN + LSTM).

    Returns:
        prediction  : "real" | "fake"
        confidence  : float in [0, 1]
        heatmap_url : base64-encoded Grad-CAM heatmap of representative frame
        frame_count : number of frames analysed
    """
    content_type = file.content_type or ""
    contents = await file.read()
    try:
        validate_video(content_type, len(contents))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    video_model = request.app.state.video_model
    try:
        result = video_model.predict(contents)
        heatmap = explain_video(video_model, contents)
    except Exception as exc:
        logger.exception("Video prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    response = {
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "raw_score": result["raw_score"],
        "frame_count": result.get("frame_count"),
        "heatmap_url": heatmap,
        "filename": file.filename,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    await _save_result(request, "video", response)
    return response


# ──────────────────────────────────────────────────────────────────────────────
# Audio prediction
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/predict/audio", tags=["prediction"])
async def predict_audio(request: Request, file: UploadFile = File(...)):
    """
    Analyse an uploaded audio file for deepfake content (spectrogram CNN).

    Returns:
        prediction  : "real" | "fake"
        confidence  : float in [0, 1]
        heatmap_url : base64-encoded spectrogram saliency map
    """
    content_type = file.content_type or ""
    contents = await file.read()
    try:
        validate_audio(content_type, len(contents))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    audio_model = request.app.state.audio_model
    try:
        result = audio_model.predict(contents)
        heatmap = explain_audio(audio_model, contents)
    except Exception as exc:
        logger.exception("Audio prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    response = {
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "raw_score": result["raw_score"],
        "heatmap_url": heatmap,
        "filename": file.filename,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    await _save_result(request, "audio", response)
    return response


# ──────────────────────────────────────────────────────────────────────────────
# Results listing
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/results", tags=["results"])
async def list_results(request: Request, limit: int = 20):
    """Return the most recent detection results stored in MongoDB."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        return {"results": [], "message": "Database not connected"}
    try:
        cursor = db.results.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
        results = await cursor.to_list(length=limit)
        return {"results": results}
    except Exception as exc:
        logger.exception("Failed to fetch results")
        raise HTTPException(status_code=500, detail="Database error") from exc


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

async def _save_result(request: Request, media_type: str, result: dict) -> None:
    """Persist a detection result to MongoDB (best-effort; errors are logged)."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        return
    try:
        doc = {"media_type": media_type, **result}
        # Strip heatmap data from the stored document to keep the DB lean
        doc.pop("heatmap_url", None)
        await db.results.insert_one(doc)
    except Exception as exc:
        logger.warning("Could not save result to DB: %s", exc)
