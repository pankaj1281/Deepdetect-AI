"""
FastAPI application entry point for the AI-Powered Multi-Modal Fake Content Detector.

Run with:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.models.audio_model import AudioModel
from backend.models.image_model import ImageModel
from backend.models.video_model import VideoModel
from backend.routes import router

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Deepdetect-AI",
    description="AI-Powered Multi-Modal Fake Content Detector (images, videos, audio)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow all origins in development; tighten in production via ALLOWED_ORIGINS env var
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


# ──────────────────────────────────────────────────────────────────────────────
# Lifespan (startup / shutdown)
# ──────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event() -> None:
    """Initialise models and optional database connection on startup."""
    logger.info("Loading deepfake detection models …")

    image_model_path = os.getenv("IMAGE_MODEL_PATH")
    video_model_path = os.getenv("VIDEO_MODEL_PATH")
    audio_model_path = os.getenv("AUDIO_MODEL_PATH")

    app.state.image_model = ImageModel(model_path=image_model_path)
    app.state.video_model = VideoModel(model_path=video_model_path)
    app.state.audio_model = AudioModel(model_path=audio_model_path)
    logger.info("Models loaded successfully.")

    # Optional MongoDB connection
    mongo_uri = os.getenv("MONGO_URI")
    if mongo_uri:
        try:
            import motor.motor_asyncio as motor  # type: ignore

            client = motor.AsyncIOMotorClient(mongo_uri)
            app.state.db = client[os.getenv("MONGO_DB", "deepdetect")]
            logger.info("Connected to MongoDB at %s", mongo_uri)
        except Exception as exc:
            logger.warning("MongoDB connection failed: %s — results will not be persisted", exc)
            app.state.db = None
    else:
        logger.info("MONGO_URI not set — results will not be persisted")
        app.state.db = None


@app.on_event("shutdown")
async def shutdown_event() -> None:
    logger.info("Shutting down Deepdetect-AI backend.")
