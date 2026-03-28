"""
Preprocessing utilities for image, video, and audio inputs.
"""

import io
import logging
import os
import tempfile

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Image utilities
# ──────────────────────────────────────────────

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/avi", "video/quicktime", "video/x-matroska"}
ALLOWED_AUDIO_TYPES = {"audio/wav", "audio/mpeg", "audio/x-wav", "audio/ogg", "audio/flac"}

MAX_IMAGE_SIZE = 20 * 1024 * 1024   # 20 MB
MAX_VIDEO_SIZE = 200 * 1024 * 1024  # 200 MB
MAX_AUDIO_SIZE = 50 * 1024 * 1024   # 50 MB


def validate_image(content_type: str, file_size: int) -> None:
    """Raise ValueError when the image upload is invalid."""
    if content_type not in ALLOWED_IMAGE_TYPES:
        raise ValueError(
            f"Unsupported image type '{content_type}'. "
            f"Allowed: {', '.join(ALLOWED_IMAGE_TYPES)}"
        )
    if file_size > MAX_IMAGE_SIZE:
        raise ValueError(f"Image exceeds 20 MB limit ({file_size} bytes)")


def validate_video(content_type: str, file_size: int) -> None:
    """Raise ValueError when the video upload is invalid."""
    if content_type not in ALLOWED_VIDEO_TYPES:
        raise ValueError(
            f"Unsupported video type '{content_type}'. "
            f"Allowed: {', '.join(ALLOWED_VIDEO_TYPES)}"
        )
    if file_size > MAX_VIDEO_SIZE:
        raise ValueError(f"Video exceeds 200 MB limit ({file_size} bytes)")


def validate_audio(content_type: str, file_size: int) -> None:
    """Raise ValueError when the audio upload is invalid."""
    if content_type not in ALLOWED_AUDIO_TYPES:
        raise ValueError(
            f"Unsupported audio type '{content_type}'. "
            f"Allowed: {', '.join(ALLOWED_AUDIO_TYPES)}"
        )
    if file_size > MAX_AUDIO_SIZE:
        raise ValueError(f"Audio exceeds 50 MB limit ({file_size} bytes)")


def resize_image(image_bytes: bytes, size: tuple = (224, 224)) -> bytes:
    """Resize an image to the given (width, height) and return as JPEG bytes."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def extract_video_thumbnail(video_bytes: bytes) -> bytes:
    """Return the first frame of a video as JPEG bytes."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    try:
        cap = cv2.VideoCapture(tmp_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            return buf.getvalue()
        return b""
    finally:
        os.unlink(tmp_path)
