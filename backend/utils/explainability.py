"""
Explainability utilities (Grad-CAM) for the deepfake detection models.

For images  : Grad-CAM highlights manipulated regions via a saliency heatmap.
For videos  : Grad-CAM is applied to the most influential frame.
For audio   : A saliency map is generated over the mel-spectrogram.
"""

import base64
import io
import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Generic Grad-CAM helper
# ──────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Computes Grad-CAM for any PyTorch model given a target convolutional layer.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Compute the Grad-CAM heatmap for the given input.

        Args:
            input_tensor: (1, C, H, W)
        Returns:
            heatmap: numpy array (H, W) with values in [0, 1]
        """
        self.model.eval()
        output = self.model(input_tensor)
        self.model.zero_grad()
        output.backward(torch.ones_like(output))

        if self.gradients is None or self.activations is None:
            return np.zeros((input_tensor.shape[-2], input_tensor.shape[-1]))

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[-2], input_tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ──────────────────────────────────────────────────────────────────────────────
# Heatmap overlay helpers
# ──────────────────────────────────────────────────────────────────────────────

def cam_to_heatmap_overlay(original_image_bytes: bytes, cam: np.ndarray) -> str:
    """
    Overlay a Grad-CAM heatmap onto the original image.

    Args:
        original_image_bytes: raw image bytes
        cam: (H, W) normalised heatmap in [0, 1]
    Returns:
        Base64-encoded PNG data URI
    """
    img = Image.open(io.BytesIO(original_image_bytes)).convert("RGB")
    img_array = np.array(img.resize((224, 224)))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.6 * img_array + 0.4 * heatmap_rgb).astype(np.uint8)

    out = Image.fromarray(overlay)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def spectrogram_heatmap(cam: np.ndarray) -> str:
    """
    Render the Grad-CAM heatmap for a spectrogram as a coloured PNG.

    Args:
        cam: (H, W) normalised heatmap in [0, 1]
    Returns:
        Base64-encoded PNG data URI
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_PLASMA)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(heatmap_rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


# ──────────────────────────────────────────────────────────────────────────────
# Per-modality explainability entry points
# ──────────────────────────────────────────────────────────────────────────────

def explain_image(image_model, image_bytes: bytes) -> str:
    """
    Generate a Grad-CAM heatmap for an image prediction.

    Args:
        image_model: ImageModel instance
        image_bytes: raw image bytes
    Returns:
        Base64 data URI string
    """
    try:
        # Locate last convolutional layer in EfficientNet backbone
        target_layer = image_model.model.backbone.features[-1][0]
        grad_cam = GradCAM(image_model.model, target_layer)
        tensor = image_model.preprocess(image_bytes).requires_grad_(True)
        cam = grad_cam.generate(tensor)
        return cam_to_heatmap_overlay(image_bytes, cam)
    except Exception as exc:
        logger.warning("Grad-CAM for image failed: %s", exc)
        return ""


def explain_video(video_model, video_bytes: bytes) -> str:
    """
    Generate a Grad-CAM heatmap for the most representative video frame.

    Args:
        video_model: VideoModel instance
        video_bytes: raw video bytes
    Returns:
        Base64 data URI string
    """
    try:
        frames = video_model.extract_frames(video_bytes)
        # Use the middle frame for explanation
        mid_frame = frames[len(frames) // 2]
        frame_bytes = _tensor_to_bytes(mid_frame)

        # Re-use image-level Grad-CAM on the encoder
        target_layer = video_model.model.encoder.features[-1]
        grad_cam = GradCAM(video_model.model.encoder, target_layer)
        # Encode single frame
        tensor = mid_frame.unsqueeze(0).to(video_model.device).requires_grad_(True)
        cam = grad_cam.generate(tensor)
        return cam_to_heatmap_overlay(frame_bytes, cam)
    except Exception as exc:
        logger.warning("Grad-CAM for video failed: %s", exc)
        return ""


def explain_audio(audio_model, audio_bytes: bytes) -> str:
    """
    Generate a saliency heatmap over the mel-spectrogram of the audio.

    Args:
        audio_model: AudioModel instance
        audio_bytes: raw audio bytes
    Returns:
        Base64 data URI string
    """
    try:
        target_layer = audio_model.model.features[-4]  # last conv block
        grad_cam = GradCAM(audio_model.model, target_layer)
        tensor = audio_model.preprocess(audio_bytes).requires_grad_(True)
        cam = grad_cam.generate(tensor)
        return spectrogram_heatmap(cam)
    except Exception as exc:
        logger.warning("Grad-CAM for audio failed: %s", exc)
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Convert a (3, H, W) normalised tensor back to JPEG bytes."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = tensor.cpu() * std + mean
    img_tensor = img_tensor.clamp(0, 1)
    img_array = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img_array).save(buf, format="JPEG")
    return buf.getvalue()
